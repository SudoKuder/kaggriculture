"""Self-play training loop with REINFORCE policy gradient.

Core idea: at each day boundary (hour == 0), we:
1. Extract state features
2. Sample an action vector from the Actor network
3. Record (features, sampled_actions) at the decision point
4. Run the game to completion
5. Pair the recorded states/actions with the final money delta
6. Train the Actor via REINFORCE on the accumulated data
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
from strategy.features import extract_features, FEATURE_DIM, _log_money
from strategy.actor_net import ActorNetTorch, ActorNetNumpy, torch_to_numpy, save_weights
from strategy.value_net import ValueNetTorch
from strategy.opponents import OpponentPool


# ---------------------------------------------------------------------------
# Strategic agent wrapper for training
# ---------------------------------------------------------------------------

class StrategicTrainingAgent:
    """Wraps the heuristic agent with strategic plan selection during
    training episodes. Records state features and actions.
    """

    def __init__(self, actor_net_torch, player_id=0, epsilon=0.0):
        self.actor_net = actor_net_torch
        self.player_id = player_id
        self.epsilon = epsilon
        self.recorded_transitions = []  # list of (features, action_dict)
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

            # Sample action from the PyTorch model
            with torch.no_grad():
                device = next(self.actor_net.parameters()).device if hasattr(self.actor_net, "parameters") else "cpu"
                feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Epsilon-greedy (random action) vs sampled action
                if random.random() < self.epsilon:
                    # Biased random actions for exploration
                    a_land = float(random.random() > 0.8)
                    a_hire = random.randint(0, 2)
                    a_sell = [float(random.random() < 0.1) for _ in range(9)]
                    if random.random() < 0.7:
                        a_seed_c = 0
                        a_seed_f = random.random() * 0.5
                    else:
                        a_seed_c = random.randint(0, 4)
                        a_seed_f = 0.6 + random.random() * 0.4
                    if random.random() < 0.9:
                        a_anim_t = 0
                        a_anim_f = random.random() * 0.5
                    else:
                        a_anim_t = random.randint(0, 2)
                        a_anim_f = 0.6 + random.random() * 0.4
                    a_weed = 5.0
                    a_maint = 21.0
                    a_panic = 22.0
                    a_seed_m = 2.0
                    a_land_b = 500.0
                    
                    action_dict = {
                        "buy_land": torch.tensor([a_land]),
                        "hire_target": torch.tensor([a_hire]),
                        "sell_hold": torch.tensor([a_sell]),
                        "buy_seed_crop": torch.tensor([a_seed_c]),
                        "buy_seed_frac": torch.tensor([a_seed_f]),
                        "buy_animal_type": torch.tensor([a_anim_t]),
                        "buy_animal_frac": torch.tensor([a_anim_f]),
                        "weed_penalty": torch.tensor([a_weed]),
                        "maint_water_hour": torch.tensor([a_maint]),
                        "panic_drop_hour": torch.tensor([a_panic]),
                        "seed_threshold_mult": torch.tensor([a_seed_m]),
                        "land_unlock_buffer": torch.tensor([a_land_b]),
                    }
                else:
                    action_dict, _, _ = self.actor_net.get_action(feat_tensor, deterministic=False)

            # Build plan dict from action_dict
            plan = {}
            plan["buy_land"] = bool(action_dict["buy_land"].item() > 0.5)
            plan["hire_target"] = int(action_dict["hire_target"].item()) + 1
            
            PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "EGG", "MILK", "WOOL", "FERTILIZER"]
            sell_hold = {}
            for i, prod in enumerate(PRODUCT_ORDER):
                if action_dict["sell_hold"][0, i].item() > 0.8:  # Requires strong signal to hold
                    sell_hold[prod] = "hold"
                else:
                    sell_hold[prod] = "sell"
            plan["sell_hold"] = sell_hold
            
            CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
            crop = CROP_ORDER[int(action_dict["buy_seed_crop"].item())]
            raw_frac = action_dict["buy_seed_frac"].item()
            frac = (raw_frac - 0.6) / 0.4 if raw_frac > 0.6 else 0.0
            money = obs["farms"][obs["player"]]["money"]
            
            # Need to get seed price, import from kaggle_environments
            from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS
            seed_price = CROPS[crop]["seed"]
            qty = int((money * frac) / seed_price) if seed_price > 0 else 0
            if qty > 0:
                plan["buy_seed"] = {"crop": crop, "qty": qty}
            else:
                plan["buy_seed"] = None
                
            ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
            anim = ANIMAL_ORDER[int(action_dict["buy_animal_type"].item())]
            raw_anim_frac = action_dict["buy_animal_frac"].item()
            frac_anim = (raw_anim_frac - 0.6) / 0.4 if raw_anim_frac > 0.6 else 0.0
            anim_price = ANIMALS[anim]["cost"]
            qty_anim = int((money * frac_anim) / anim_price) if anim_price > 0 else 0
            if qty_anim > 0:
                plan["buy_animal"] = {"type": anim, "qty": qty_anim}
            else:
                plan["buy_animal"] = None
                
            plan["label"] = "learned_action"
            self.current_plan = plan
            
            # Save for REINFORCE (store the un-batched tensors/arrays)
            saved_action = {k: v.squeeze(0).cpu().numpy() for k, v in action_dict.items()}
            self.recorded_transitions.append((features, saved_action))
        
        # Inject the current plan into the heuristic agent's state
        import main
        main.IS_TRAINING = True
        main._current_plan[obs["player"]] = self.current_plan
        
        action = self._heuristic(obs)
        return action


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_actor_network(
    num_episodes=500,
    batch_size=32,
    lr=1e-3,
    checkpoint_interval=100,
    output_dir="strategy_checkpoints",
    verbose=True,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
    critic = ValueNetTorch(input_dim=FEATURE_DIM).to(device)
    
    start_episode = 0
    final_path = os.path.join(output_dir, "actor_net_final.npz")
    
    checkpoints = glob.glob(os.path.join(output_dir, "actor_net_ep*.npz"))
    def extract_ep(p):
        match = re.search(r"actor_net_ep(\d+)\.npz", p)
        return int(match.group(1)) if match else -1
        
    if checkpoints:
        latest_ckpt = max(checkpoints, key=extract_ep)
        start_episode = extract_ep(latest_ckpt)
        
    loaded_path = None
    loaded_path = None
    loaded_critic_path = None
    if os.path.exists(final_path):
        loaded_path = final_path
        critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
        if os.path.exists(critic_final_path):
            loaded_critic_path = critic_final_path
    elif checkpoints:
        loaded_path = max(checkpoints, key=extract_ep)
        loaded_critic_path = os.path.join(output_dir, f"critic_net_ep{start_episode}.npz")
        
    if loaded_path:
        if verbose:
            print(f"Resuming training from weights at {loaded_path} (Starting at Episode {start_episode})")
        from strategy.actor_net import load_weights
        layers = load_weights(loaded_path)
        model.import_numpy_weights(layers)
        
        if loaded_critic_path and os.path.exists(loaded_critic_path):
            from strategy.value_net import load_weights as load_critic_weights
            c_layers = load_critic_weights(loaded_critic_path)
            critic.import_numpy_weights(c_layers)

    optimizer = optim.Adam(list(model.parameters()) + list(critic.parameters()), lr=lr)

    pool = OpponentPool(
        include_starter=True,
        include_heuristic=True,
        include_passive=True,
        noise_variants=2,
    )
    
    # Load past checkpoints into the pool
    if checkpoints:
        for ckpt in checkpoints:
            ep_num = extract_ep(ckpt)
            if ep_num > 0:
                snapshot_model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
                from strategy.actor_net import load_weights
                snapshot_model.import_numpy_weights(load_weights(ckpt))
                snapshot_agent = StrategicTrainingAgent(
                    actor_net_torch=snapshot_model,
                    player_id=1,
                    epsilon=0.0
                )
                pool.add_snapshot(snapshot_agent, label=f"ep{ep_num}")
        if verbose:
            print(f"Loaded {len(checkpoints)} past checkpoints into OpponentPool (size: {len(pool)})")

    data_features = []
    data_actions = []
    data_targets = []
    max_buffer = 50000

    total_games = 0
    running_loss = 0.0
    loss_count = 0

    epsilon_start = 0.2
    epsilon_end = 0.01

    t_start = time.time()

    for episode in range(start_episode, num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, num_episodes - 1))
        seed = random.randint(0, 100000)

        model.eval()

        agent0 = StrategicTrainingAgent(
            actor_net_torch=model,
            player_id=0,
            epsilon=epsilon,
        )

        opp_agent, opp_label = pool.sample()

        try:
            env = make("kaggriculture", configuration={"seed": seed})
            env.run([agent0, opp_agent])
        except KeyboardInterrupt:
            if verbose:
                print("\nTraining interrupted by user. Saving final weights...")
            break
        except Exception as e:
            if verbose:
                print(f"  Episode {episode}: ERROR ({e}), skipping")
            continue

        final_obs = env.state[0]["observation"]
        m0 = final_obs["farms"][0]["money"]
        m1 = final_obs["farms"][1]["money"]
        
        # Reward Shaping: Give credit for unsold seeds for both players
        from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
        priv0 = env.state[0]["observation"].get("private", {})
        asset_value0 = sum(qty * CROPS[crop]["seed"] for crop, qty in priv0.get("seeds", {}).items())
        
        priv1 = env.state[1]["observation"].get("private", {})
        asset_value1 = sum(qty * CROPS[crop]["seed"] for crop, qty in priv1.get("seeds", {}).items())
        
        money_delta = (m0 + asset_value0) - (m1 + asset_value1)

        for feat, act in agent0.recorded_transitions:
            data_features.append(feat)
            data_actions.append(act)
            data_targets.append(money_delta)

        if len(data_features) > max_buffer:
            data_features = data_features[-max_buffer:]
            data_actions = data_actions[-max_buffer:]
            data_targets = data_targets[-max_buffer:]

        total_games += 1

        if len(data_features) >= batch_size:
            model.train()

            num_updates = max(1, len(agent0.recorded_transitions) // (batch_size // 4))
            for _ in range(num_updates):
                indices = random.sample(range(len(data_features)), batch_size)
                
                batch_x = torch.tensor(
                    np.array([data_features[i] for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                
                # Re-compute log probs for the recorded actions
                _, _, logits = model.get_action(batch_x)
                
                # Collect actual actions taken
                a_land = torch.tensor(np.array([data_actions[i]["buy_land"] for i in indices])).to(device)
                a_hire = torch.tensor(np.array([data_actions[i]["hire_target"] for i in indices])).to(device)
                a_sell = torch.tensor(np.array([data_actions[i]["sell_hold"] for i in indices])).to(device)
                a_seed_c = torch.tensor(np.array([data_actions[i]["buy_seed_crop"] for i in indices])).to(device)
                a_seed_f = torch.tensor(np.array([data_actions[i]["buy_seed_frac"] for i in indices])).to(device)
                a_anim_t = torch.tensor(np.array([data_actions[i]["buy_animal_type"] for i in indices])).to(device)
                a_anim_f = torch.tensor(np.array([data_actions[i]["buy_animal_frac"] for i in indices])).to(device)
                a_weed = torch.tensor(np.array([data_actions[i]["weed_penalty"] for i in indices])).to(device)
                a_maint = torch.tensor(np.array([data_actions[i]["maint_water_hour"] for i in indices])).to(device)
                a_panic = torch.tensor(np.array([data_actions[i]["panic_drop_hour"] for i in indices])).to(device)
                a_seed_m = torch.tensor(np.array([data_actions[i]["seed_threshold_mult"] for i in indices])).to(device)
                a_land_b = torch.tensor(np.array([data_actions[i]["land_unlock_buffer"] for i in indices])).to(device)
                
                # Compute log probs and entropies
                from torch.distributions import Categorical, Bernoulli, Normal
                dist_land = Bernoulli(torch.sigmoid(logits[..., 0]))
                dist_hire = Categorical(logits=logits[..., 1:7])
                dist_sell = Bernoulli(torch.sigmoid(logits[..., 7:16]))
                dist_seed_c = Categorical(logits=logits[..., 16:21])
                dist_seed_f = Normal(torch.sigmoid(logits[..., 21]), 0.2)
                dist_anim_t = Categorical(logits=logits[..., 22:25])
                dist_anim_f = Normal(torch.sigmoid(logits[..., 25]), 0.2)
                dist_weed = Normal(torch.sigmoid(logits[..., 26]) * 10.0, 1.0)
                dist_maint = Normal(torch.sigmoid(logits[..., 27]) * 23.0, 2.0)
                dist_panic = Normal(torch.sigmoid(logits[..., 28]) * 23.0, 2.0)
                dist_seed_m = Normal(torch.sigmoid(logits[..., 29]) * 5.0, 0.5)
                dist_land_b = Normal(torch.sigmoid(logits[..., 30]) * 2000.0, 200.0)

                lp_land = dist_land.log_prob(a_land)
                lp_hire = dist_hire.log_prob(a_hire)
                lp_sell = dist_sell.log_prob(a_sell).sum(dim=-1)
                lp_seed_c = dist_seed_c.log_prob(a_seed_c)
                lp_seed_f = dist_seed_f.log_prob(a_seed_f)
                lp_anim_t = dist_anim_t.log_prob(a_anim_t)
                lp_anim_f = dist_anim_f.log_prob(a_anim_f)
                lp_weed = dist_weed.log_prob(a_weed)
                lp_maint = dist_maint.log_prob(a_maint)
                lp_panic = dist_panic.log_prob(a_panic)
                lp_seed_m = dist_seed_m.log_prob(a_seed_m)
                lp_land_b = dist_land_b.log_prob(a_land_b)
                
                total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f + lp_weed + lp_maint + lp_panic + lp_seed_m + lp_land_b
                
                total_entropy = dist_land.entropy() + dist_hire.entropy() + dist_sell.entropy().sum(dim=-1) + \
                                dist_seed_c.entropy() + dist_seed_f.entropy() + dist_anim_t.entropy() + \
                                dist_anim_f.entropy() + dist_weed.entropy() + dist_maint.entropy() + \
                                dist_panic.entropy() + dist_seed_m.entropy() + dist_land_b.entropy()
                
                batch_r = torch.tensor(
                    np.array([_log_money(data_targets[i]) for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                
                batch_v = critic(batch_x)
                advantages = batch_r - batch_v.detach()
                
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                actor_loss = -(total_log_prob * advantages).mean()
                critic_loss = nn.functional.mse_loss(batch_v, batch_r)
                entropy_loss = -0.01 * total_entropy.mean()  # 0.01 is the entropy coefficient
                
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
    
                running_loss += loss.item()
                loss_count += 1

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
            # Log metrics to JSONL for visualize_training.py
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
            
            log_entry = {
                "episode": episode + 1,
                "loss": avg_loss,
                "money_delta": float(money_delta),
                "epsilon": epsilon,
                "buffer_size": len(data_features),
                "eps_per_sec": eps_per_sec,
                "opponent": opp_label
            }
            
            import json
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            running_loss = 0.0
            loss_count = 0
            gc.collect()

        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"actor_net_ep{episode + 1}.npz")
            layers = model.export_numpy_weights()
            save_weights(layers, ckpt_path)
            if verbose:
                print(f"  -> Checkpoint saved: {ckpt_path}")
                
            # Add snapshot to opponent pool
            snapshot_model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
            snapshot_model.import_numpy_weights(layers)
            snapshot_agent = StrategicTrainingAgent(
                actor_net_torch=snapshot_model,
                player_id=1,
                epsilon=0.0
            )
            pool.add_snapshot(snapshot_agent, label=f"ep{episode + 1}")
            if verbose:
                print(f"  -> Added ep{episode + 1} to OpponentPool (size: {len(pool)})")

    final_path = os.path.join(output_dir, "actor_net_final.npz")
    critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
    layers = model.export_numpy_weights()
    save_weights(layers, final_path)
    
    from strategy.value_net import save_weights as save_critic_weights
    c_layers = critic.export_numpy_weights()
    save_critic_weights(c_layers, critic_final_path)
    if verbose:
        elapsed = time.time() - t_start
        print(f"\nTraining complete: {total_games} games in {elapsed:.1f}s")
        print(f"Final weights saved to {final_path}")

    return model
