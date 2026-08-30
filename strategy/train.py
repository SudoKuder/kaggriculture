"""Self-play training loop with Monte-Carlo value network training.

Core idea: at each day boundary (hour == 0), we:
1. Generate candidate strategic plans
2. Use an epsilon-greedy strategy to select a plan (evaluating candidates with a value network V(s, plan))
3. Record (encoded_features, plan_choice) state at the decision point
4. Run the game to completion
5. Pair the recorded state features with the actual final money delta
6. Train V(s, plan) via MSE on the accumulated data
"""

import os
import sys
import random
import time
import math
from copy import deepcopy
import glob
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kaggle_environments import make
from strategy.features import extract_features, FEATURE_DIM, encode_plan, _log_money
from strategy.candidates import generate_candidates
from strategy.value_net import ValueNetTorch, ValueNetNumpy, torch_to_numpy, save_weights
from strategy.opponents import OpponentPool


# ---------------------------------------------------------------------------
# Strategic agent wrapper for training
# ---------------------------------------------------------------------------

class StrategicTrainingAgent:
    """Wraps the heuristic agent with strategic plan selection during
    training episodes.  Records state features at each decision point.

    During training, this agent uses simulator rollouts to pick
    candidates (when rollout_env is provided), falling back to the value
    network score for selection.
    """

    def __init__(self, value_net_numpy, player_id=0, rollout_depth_days=2):
        self.value_net = value_net_numpy
        self.player_id = player_id
        self.rollout_depth = rollout_depth_days
        self.recorded_states = []  # list of feature vectors
        self.current_plan = None

        # Import the heuristic agent — we'll call it directly
        import main
        main._strategic_layer = None
        main.step_counter = 2
        
        self._heuristic = main.agent
        self._build_market = main.build_market
        self._GameState = main.GameState

    def __call__(self, obs, config=None):
        """Called by the kaggle_environments runner each turn."""
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)

        # At hour 0 of each day, make a strategic decision
        if hour == 0:
            features = extract_features(obs)

            # Generate and evaluate candidate plans
            candidates = generate_candidates(obs)
            best_plan, best_score = self._evaluate_candidates(
                obs, candidates, features
            )
            self.current_plan = best_plan
            
            # Record the actual plan chosen for training
            encoded_feats = self._encode_plan(features, best_plan)
            self.recorded_states.append(encoded_feats)
        
        # Inject the current plan into the heuristic agent's state
        import main
        main._current_plan[obs["player"]] = self.current_plan
        
        # Run the heuristic agent, which will now automatically budget and apply the overrides
        action = self._heuristic(obs)
        return action

    def _evaluate_candidates(self, obs, candidates, features):
        """Score each candidate plan using the value network.

        During training, we also use the value network to score the
        *current* position (not rollouts, since rollouts require env
        access that we don't have inside the agent callback — the env
        is controlled externally).  The value function is trained
        post-hoc on game outcomes.
        """
        best_plan = candidates[0]  # baseline
        best_score = -float("inf")

        for plan in candidates:
            # Score = V(features with plan context)
            # We augment the feature vector with a plan encoding
            plan_features = self._encode_plan(features, plan)
            score = self.value_net.predict(plan_features)

            if score > best_score:
                best_score = score
                best_plan = plan

        return best_plan, best_score

    def _encode_plan(self, features, plan):
        """Augment feature vector with plan information."""
        return encode_plan(features, plan)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_value_network(
    num_episodes=500,
    batch_size=32,
    lr=1e-3,
    rollout_depth_days=2,
    checkpoint_interval=100,
    output_dir="strategy_checkpoints",
    verbose=True,
):
    """Main training loop.

    Runs self-play episodes, collects (features, money_delta) samples at
    every day boundary, and trains the value network via MSE loss.

    Args:
        num_episodes: Total training episodes.
        batch_size: Mini-batch size for SGD.
        lr: Learning rate.
        rollout_depth_days: Days to roll forward per candidate evaluation.
        checkpoint_interval: Save weights every N episodes.
        output_dir: Directory for checkpoints.
        verbose: Print progress.

    Returns:
        Trained ValueNetTorch model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialise model + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    model = ValueNetTorch(input_dim=FEATURE_DIM).to(device)
    
    start_episode = 0
    final_path = os.path.join(output_dir, "value_net_final.npz")
    
    # 1. Determine start_episode from checkpoints or logs
    checkpoints = glob.glob(os.path.join(output_dir, "value_net_ep*.npz"))
    def extract_ep(p):
        match = re.search(r"value_net_ep(\d+)\.npz", p)
        return int(match.group(1)) if match else -1
        
    if checkpoints:
        latest_ckpt = max(checkpoints, key=extract_ep)
        start_episode = extract_ep(latest_ckpt)
        
    log_file = os.path.join(output_dir, "logs", "training_metrics.jsonl")
    if os.path.exists(log_file):
        try:
            import json
            with open(log_file, "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.strip():
                        last_log = json.loads(line)
                        if "episode" in last_log:
                            start_episode = max(start_episode, int(last_log["episode"]))
                        break
        except Exception:
            pass

    # 2. Load model weights
    loaded_path = None
    if os.path.exists(final_path):
        loaded_path = final_path
    elif checkpoints:
        loaded_path = max(checkpoints, key=extract_ep)
        
    if loaded_path:
        if verbose:
            print(f"Resuming training from weights at {loaded_path} (Starting at Episode {start_episode})")
        from strategy.value_net import load_weights
        layers = load_weights(loaded_path)
        model.import_numpy_weights(layers)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Opponent pool
    pool = OpponentPool(
        include_starter=True,
        include_heuristic=True,
        include_passive=True,
        noise_variants=2,
    )

    # Load existing snapshots into the pool
    if checkpoints:
        checkpoints_sorted = sorted([c for c in checkpoints if extract_ep(c) > 0], key=extract_ep)
        if checkpoints_sorted and verbose:
            print(f"Loading {len(checkpoints_sorted)} previous snapshots into opponent pool...")
            
        from strategy.value_net import load_weights
        for ckpt in checkpoints_sorted:
            ep = extract_ep(ckpt)
            try:
                dummy = ValueNetTorch(input_dim=FEATURE_DIM).to(device)
                dummy.import_numpy_weights(load_weights(ckpt))
                snap_vnet = torch_to_numpy(dummy)
                snap_agent = StrategicTrainingAgent(value_net_numpy=snap_vnet, player_id=1)
                pool.add_snapshot(snap_agent, label=f"snapshot_ep{ep}")
            except Exception as e:
                if verbose:
                    print(f"Failed to load snapshot {ckpt}: {e}")

    # Training data buffer (sliding window)
    data_features = []
    data_targets = []
    max_buffer = 50000

    total_games = 0
    running_loss = 0.0
    loss_count = 0

    # Exploration schedule: start with random plan selection, anneal to
    # greedy over training.
    epsilon_start = 0.5
    epsilon_end = 0.05

    t_start = time.time()

    for episode in range(start_episode, num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, num_episodes - 1))
        seed = random.randint(0, 100000)

        # Create the value net in numpy form for the agent
        model.eval()
        vnet_np = torch_to_numpy(model)

        # Create strategic training agent
        agent0 = StrategicTrainingAgent(
            value_net_numpy=vnet_np,
            player_id=0,
            rollout_depth_days=rollout_depth_days,
        )

        # Epsilon-greedy exploration: with probability epsilon, pick a
        # random candidate instead of the value-net's choice.
        original_evaluate = agent0._evaluate_candidates

        def explore_evaluate(obs, candidates, features, _orig=original_evaluate, _eps=epsilon):
            if random.random() < _eps:
                plan = random.choice(candidates)
                return plan, 0.0
            return _orig(obs, candidates, features)

        agent0._evaluate_candidates = explore_evaluate

        # Sample opponent
        opp_agent, opp_label = pool.sample()

        # Run the game
        try:
            env = make("kaggriculture", configuration={"seed": seed})
            env.run([agent0, opp_agent])
        except KeyboardInterrupt:
            if verbose:
                print("\nTraining interrupted by user (Ctrl+C). Stopping early and saving final weights...")
            break
        except Exception as e:
            if verbose:
                print(f"  Episode {episode}: ERROR ({e}), skipping")
            continue

        # Extract final money delta
        final_obs = env.state[0]["observation"]
        m0 = final_obs["farms"][0]["money"]
        m1 = final_obs["farms"][1]["money"]
        money_delta = m0 - m1

        # Record training samples: (features, money_delta) at each day
        for feat in agent0.recorded_states:
            data_features.append(feat)
            data_targets.append(money_delta)

        # Trim buffer
        if len(data_features) > max_buffer:
            data_features = data_features[-max_buffer:]
            data_targets = data_targets[-max_buffer:]

        total_games += 1

        # Train on accumulated data
        if len(data_features) >= batch_size:
            model.train()

            # Perform multiple batch updates to utilize the newly collected samples effectively
            num_updates = max(1, len(agent0.recorded_states) // (batch_size // 4))
            for _ in range(num_updates):
                # Sample a mini-batch
                indices = random.sample(range(len(data_features)), batch_size)
                batch_x = torch.tensor(
                    np.array([data_features[i] for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                batch_y = torch.tensor(
                    np.array([_log_money(data_targets[i]) for i in indices]),
                    dtype=torch.float32,
                ).to(device)
    
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
    
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
                running_loss += loss.item()
                loss_count += 1

        # Logging
        if verbose and (episode + 1) % 10 == 0:
            avg_loss = running_loss / max(1, loss_count) if loss_count > 0 else 0
            elapsed = time.time() - t_start
            eps_per_sec = (episode + 1) / elapsed
            print(
                f"  Episode {episode + 1}/{num_episodes} | "
                f"loss={avg_loss:.2f} | "
                f"delta={money_delta:+.0f} | "
                f"vs={opp_label} | "
                f"eps={epsilon:.2f} | "
                f"buf={len(data_features)} | "
                f"{eps_per_sec:.1f} ep/s"
            )
            
            # Log metrics to file
            import json
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "training_metrics.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "episode": episode + 1,
                    "loss": avg_loss,
                    "money_delta": money_delta,
                    "opponent": opp_label,
                    "epsilon": epsilon,
                    "buffer_size": len(data_features),
                    "eps_per_sec": eps_per_sec
                }
                f.write(json.dumps(log_entry) + "\n")
                
            running_loss = 0.0
            loss_count = 0
            gc.collect()

        # Checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"value_net_ep{episode + 1}.npz")
            layers = model.export_numpy_weights()
            save_weights(layers, ckpt_path)
            if verbose:
                print(f"  → Checkpoint saved: {ckpt_path}")

            # Add a snapshot to the opponent pool for diversity
            snapshot_vnet = torch_to_numpy(model)
            snapshot_agent = StrategicTrainingAgent(
                value_net_numpy=snapshot_vnet, player_id=1
            )
            pool.add_snapshot(snapshot_agent, label=f"snapshot_ep{episode + 1}")

    # Final save
    final_path = os.path.join(output_dir, "value_net_final.npz")
    layers = model.export_numpy_weights()
    save_weights(layers, final_path)
    if verbose:
        elapsed = time.time() - t_start
        print(f"\nTraining complete: {total_games} games in {elapsed:.1f}s")
        print(f"Final weights saved to {final_path}")
        print(f"Buffer size: {len(data_features)} samples")

    return model
