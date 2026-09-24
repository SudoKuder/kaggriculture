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

    def __init__(self, actor_net_torch, player_id=0, deterministic=False):
        self.actor_net = actor_net_torch
        self.player_id = player_id
        self.deterministic = deterministic
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
                
                # Epsilon-greedy removed: rely on the network's own stochastic sampling
                action_dict, _, _ = self.actor_net.get_action(feat_tensor, deterministic=self.deterministic)

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


class NumpySnapshotAgent:
    """Lightweight numpy-only opponent for self-play snapshots.
    
    Uses ActorNetNumpy for inference (no gradients, no GPU memory).
    Much cheaper than StrategicTrainingAgent for opponent-only use.
    
    NOTE: This agent is intentionally deterministic (argmax/threshold,
    no sampling) — that's a deliberate tradeoff for memory savings, not
    a bug.  Self-play opponents don't need exploration noise.
    """

    def __init__(self, actor_net_numpy, player_id=1):
        self.actor_net_np = actor_net_numpy
        self.player_id = player_id
        self.current_plan = None
        import main
        main._strategic_layer = None
        main.step_counter = 2
        self._heuristic = main.agent

    def __call__(self, obs, config=None):
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)

        if hour == 0:
            features = extract_features(obs)
            action_dict = self.actor_net_np.predict(features)

            plan = {}
            plan["buy_land"] = bool(action_dict["buy_land"] > 0.5)
            plan["hire_target"] = int(action_dict["hire_target"]) + 1

            PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "EGG", "MILK", "WOOL", "FERTILIZER"]
            sell_hold = {}
            for i, prod in enumerate(PRODUCT_ORDER):
                if action_dict["sell_hold"][i] > 0.5:
                    sell_hold[prod] = "hold"
                else:
                    sell_hold[prod] = "sell"
            plan["sell_hold"] = sell_hold

            CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
            from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS
            crop = CROP_ORDER[int(action_dict["buy_seed_crop"])]
            seed_price = CROPS[crop]["seed"]
            raw_frac = action_dict["buy_seed_frac"]
            frac = (raw_frac - 0.6) / 0.4 if raw_frac > 0.6 else 0.0
            money = obs["farms"][obs["player"]]["money"]
            qty = int((money * frac) / seed_price) if seed_price > 0 else 0
            plan["buy_seed"] = {"crop": crop, "qty": qty} if qty > 0 else None

            ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
            anim = ANIMAL_ORDER[int(action_dict["buy_animal_type"])]
            raw_anim_frac = action_dict["buy_animal_frac"]
            frac_anim = (raw_anim_frac - 0.6) / 0.4 if raw_anim_frac > 0.6 else 0.0
            anim_price = ANIMALS[anim]["cost"]
            qty_anim = int((money * frac_anim) / anim_price) if anim_price > 0 else 0
            plan["buy_animal"] = {"type": anim, "qty": qty_anim} if qty_anim > 0 else None

            plan["label"] = "numpy_snapshot"
            self.current_plan = plan

        import main
        main.IS_TRAINING = True
        main._current_plan[obs["player"]] = self.current_plan
        return self._heuristic(obs)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_actor_network(
    num_episodes=500,
    batch_size=32,
    lr=1e-3,
    checkpoint_interval=100,
    eval_interval=500,
    eval_episodes_per_opponent=50,
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
    
    # We create a static eval pool with just the fixed opponents for pure evaluation
    eval_pool = OpponentPool(
        include_starter=True,
        include_heuristic=True,
        include_passive=True,
        noise_variants=2,
    )
    eval_opponents = eval_pool.all_agents()
    
    # Load ONLY the most recent N checkpoints into the pool to avoid
    # immediately flooding self-play with stale opponents (which was the
    # primary cause of the 98% self-play / collapsed reward signal bug).
    MAX_STARTUP_SNAPSHOTS = 5
    if checkpoints:
        # Sort newest first, take at most MAX_STARTUP_SNAPSHOTS
        sorted_ckpts = sorted(checkpoints, key=extract_ep, reverse=True)[:MAX_STARTUP_SNAPSHOTS]
        for ckpt in sorted_ckpts:
            ep_num = extract_ep(ckpt)
            if ep_num > 0:
                from strategy.actor_net import load_weights
                snapshot_np = ActorNetNumpy(load_weights(ckpt))
                snapshot_agent = NumpySnapshotAgent(
                    actor_net_numpy=snapshot_np,
                    player_id=1,
                )
                pool.add_snapshot(snapshot_agent, label=f"ep{ep_num}")
        if verbose:
            print(f"Loaded {len(sorted_ckpts)} most-recent checkpoints into OpponentPool (size: {len(pool)})")

    # Entropy coefficients with linear annealing.
    # Start high enough to force exploration, decay to a small floor so the
    # policy gradient can dominate once the agent has seen enough diversity.
    # Reduced from 0.02/30k after MIN_STD per-head fix and loss-level clamp
    # stabilized gradients that previously caused collapse at lower coefficients.
    ENTROPY_COEFF_DISCRETE_START = 0.015
    ENTROPY_COEFF_DISCRETE_END   = 0.003
    ENTROPY_COEFF_CONT_START     = 0.002
    ENTROPY_COEFF_CONT_END       = 0.0005
    ENTROPY_ANNEAL_STEPS         = 10_000   # episodes over which to anneal (was 30k)
    # Discount factor for reward-to-go
    GAMMA = 0.99

    data_features = []
    data_actions = []
    data_targets = []
    total_games = 0
    running_loss = 0.0
    running_actor_loss = 0.0
    running_critic_loss = 0.0
    running_entropy_loss_disc = 0.0
    running_entropy_loss_cont = 0.0
    
    # Tracking for new diagnostics
    running_lp_mean = 0.0
    running_lp_std = 0.0
    running_clamp_frac = 0.0
    running_ent_land = 0.0
    running_ent_hire = 0.0
    running_ent_sell = 0.0
    running_ent_seed_c = 0.0
    running_ent_anim_t = 0.0
    running_cont_entropy = 0.0
    
    # Rolling win-rate tracking per opponent category
    FIXED_LABELS = {"starter", "heuristic", "noisy_0.1", "noisy_0.2", "passive", "top_player_script"}
    batch_wins_fixed = 0
    batch_total_fixed = 0
    batch_wins_self = 0
    batch_total_self = 0
    
    loss_count = 0
    last_buffer_size = 0

    t_start = time.time()

    for episode in range(start_episode, num_episodes):
        seed = random.randint(0, 100000)

        model.eval()

        agent0 = StrategicTrainingAgent(
            actor_net_torch=model,
            player_id=0,
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
        
        # Use raw bank money delta — the competition scores bank money only.
        # The panic-liquidation logic in main.py already converts inventory
        # to cash on day 29-30, and leftover seeds score zero.
        money_delta = m0 - m1

        # Discounted returns: store plain money_delta * discount.
        # The sign-blended log-transform is applied exactly once at batch
        # time (see batch_r_raw below) — NOT here, to avoid double
        # log-transforming.
        n_transitions = len(agent0.recorded_transitions)
        for step_idx, (feat, act) in enumerate(agent0.recorded_transitions):
            discount = GAMMA ** (n_transitions - 1 - step_idx)
            data_features.append(feat)
            data_actions.append(act)
            data_targets.append(money_delta * discount)

        total_games += 1
        
        # Track win rates by opponent category
        won = money_delta > 0
        pool.report_result(opp_label, won)  # update challenge-based sampling
        if opp_label in FIXED_LABELS:
            batch_total_fixed += 1
            if won:
                batch_wins_fixed += 1
        else:
            batch_total_self += 1
            if won:
                batch_wins_self += 1

        # Train every 10 episodes on a fresh on-policy batch, then clear it
        if total_games > 0 and total_games % 10 == 0:
            model.train()

            num_updates = max(1, len(data_features) // batch_size)
            for _ in range(num_updates):
                indices = random.sample(range(len(data_features)), batch_size)
                
                batch_x = torch.tensor(
                    np.array([data_features[i] for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                
                # Re-compute log probs for the recorded actions
                # Use the shared get_distributions() helper so min-std
                # stays in sync with actor_net.py.
                _, _, logits = model.get_action(batch_x)
                discrete_dists, continuous_dists = model.get_distributions(logits)
                
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
                
                # Unpack distributions from get_distributions()
                dist_land   = discrete_dists[0][1]
                dist_hire   = discrete_dists[1][1]
                dist_sell   = discrete_dists[2][1]
                dist_seed_c = discrete_dists[3][1]
                dist_anim_t = discrete_dists[4][1]
                dist_seed_f = continuous_dists[0][1]
                dist_anim_f = continuous_dists[1][1]
                dist_weed   = continuous_dists[2][1]
                dist_maint  = continuous_dists[3][1]
                dist_panic  = continuous_dists[4][1]
                dist_seed_m = continuous_dists[5][1]
                dist_land_b = continuous_dists[6][1]

                # Compute log probs per action head
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
                
                total_log_prob = (lp_land + lp_hire + lp_sell + lp_seed_c +
                                  lp_seed_f + lp_anim_t + lp_anim_f +
                                  lp_weed + lp_maint + lp_panic +
                                  lp_seed_m + lp_land_b)
                
                # NOTE: We no longer clamp total_log_prob itself.  The
                # underlying cause of extreme magnitudes (min_std = 0.01)
                # has been fixed by raising MIN_STD to 0.05 in
                # ActorNetTorch.  Instead we clip the per-sample
                # advantage-weighted product at the *loss* level so
                # gradient flow is never zeroed out.


                # --- Separate entropy for discrete vs continuous heads ---
                discrete_entropy = (
                    dist_land.entropy() +
                    dist_hire.entropy() +
                    dist_sell.entropy().sum(dim=-1) +
                    dist_seed_c.entropy() +
                    dist_anim_t.entropy()
                )
                continuous_entropy = (
                    dist_seed_f.entropy() +
                    dist_anim_f.entropy() +
                    dist_weed.entropy() +
                    dist_maint.entropy() +
                    dist_panic.entropy() +
                    dist_seed_m.entropy() +
                    dist_land_b.entropy()
                )
                
                # Sign-blended log-transform applied exactly once here.
                # data_targets[i] = money_delta * discount (raw scale).
                # sign_bonus pushes gradient toward correct win/loss sign
                # in close games, without being crushed by the log.
                def _blend_target(raw):
                    sign_bonus = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)
                    return 0.7 * _log_money(raw) + 0.3 * sign_bonus

                batch_r_raw = torch.tensor(
                    np.array([_blend_target(data_targets[i]) for i in indices]),
                    dtype=torch.float32,
                ).to(device)

                # Normalize the raw returns themselves before critic regression.
                # When all returns are near-zero (self-play cancellation), the
                # unscaled targets give the critic nothing to learn from, which
                # collapses advantages → kills actor gradients.
                if batch_r_raw.std() > 1e-6:
                    batch_r = (batch_r_raw - batch_r_raw.mean()) / (batch_r_raw.std() + 1e-8)
                else:
                    batch_r = batch_r_raw  # all same value – critic can't help anyway
                
                batch_v = critic(batch_x)
                advantages = batch_r - batch_v.detach()
                
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # DIAGNOSTIC: fraction of samples hitting the loss-level clamp.
                # Unlike the old log-prob clamp (which killed 100% of gradients),
                # clipping the product at [-20, 20] acts as gradient clipping
                # and typically only affects ~40% of samples.
                with torch.no_grad():
                    product = total_log_prob * advantages
                    would_clamp = ((product < -20.0) | (product > 20.0)).float()
                    clamp_frac = would_clamp.mean().item()
                
                # Clip at the loss level: cap the per-sample product
                # (log_prob * advantage) to [-20, 20] so no single sample
                # can produce an outsized gradient, but gradient still
                # flows for all samples (unlike the old log-prob clamp).
                # Linearly anneal entropy coefficient from start → end
                anneal_frac = min(1.0, (episode - start_episode) / max(1, ENTROPY_ANNEAL_STEPS))
                ec_disc = ENTROPY_COEFF_DISCRETE_START + anneal_frac * (ENTROPY_COEFF_DISCRETE_END - ENTROPY_COEFF_DISCRETE_START)
                ec_cont = ENTROPY_COEFF_CONT_START     + anneal_frac * (ENTROPY_COEFF_CONT_END     - ENTROPY_COEFF_CONT_START)

                actor_loss = -(total_log_prob * advantages).clamp(-20.0, 20.0).mean()
                critic_loss = nn.functional.mse_loss(batch_v, batch_r)
                entropy_loss_disc = -ec_disc * discrete_entropy.mean()
                entropy_loss_cont = -ec_cont * continuous_entropy.mean()
                
                loss = actor_loss + 0.5 * critic_loss + entropy_loss_disc + entropy_loss_cont
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
    
                running_loss += loss.item()
                running_actor_loss += actor_loss.item()
                running_critic_loss += critic_loss.item()
                running_entropy_loss_disc += entropy_loss_disc.item()
                running_entropy_loss_cont += entropy_loss_cont.item()
                
                running_lp_mean += total_log_prob.mean().item()
                running_lp_std += total_log_prob.std().item() if len(total_log_prob) > 1 else 0.0
                running_clamp_frac += clamp_frac
                running_ent_land += dist_land.entropy().mean().item()
                running_ent_hire += dist_hire.entropy().mean().item()
                running_ent_sell += dist_sell.entropy().sum(dim=-1).mean().item()
                running_ent_seed_c += dist_seed_c.entropy().mean().item()
                running_ent_anim_t += dist_anim_t.entropy().mean().item()
                running_cont_entropy += continuous_entropy.mean().item()
                
                loss_count += 1

            # CRITICAL FIX: Clear the on-policy buffer after training!
            last_buffer_size = len(data_features)
            data_features.clear()
            data_actions.clear()
            data_targets.clear()

        if verbose and (episode + 1) % 10 == 0:
            lc = max(1, loss_count)
            avg_loss = running_loss / lc
            avg_actor = running_actor_loss / lc
            avg_critic = running_critic_loss / lc
            avg_ent_disc = running_entropy_loss_disc / lc
            avg_ent_cont = running_entropy_loss_cont / lc
            avg_entropy = avg_ent_disc + avg_ent_cont
            
            avg_lp_mean = running_lp_mean / lc
            avg_lp_std = running_lp_std / lc
            avg_clamp_frac = running_clamp_frac / lc
            avg_ent_land = running_ent_land / lc
            avg_ent_hire = running_ent_hire / lc
            avg_ent_sell = running_ent_sell / lc
            avg_ent_seed_c = running_ent_seed_c / lc
            avg_ent_anim_t = running_ent_anim_t / lc
            avg_cont_entropy = running_cont_entropy / lc
            
            elapsed = time.time() - t_start
            eps_per_sec = (episode + 1) / elapsed
            print(
                f"  Episode {episode + 1}/{num_episodes} | "
                f"loss={avg_loss:.2f} (a:{avg_actor:.2f} c:{avg_critic:.2f} "
                f"e_d:{avg_ent_disc:.4f} e_c:{avg_ent_cont:.4f}) | "
                f"delta={money_delta:+.0f} | "
                f"vs={opp_label} | "
                f"buf={last_buffer_size} | "
                f"{eps_per_sec:.1f} ep/s"
            )
            print(
                f"    Diag: lp_mean={avg_lp_mean:.2f}, lp_std={avg_lp_std:.2f}, "
                f"clamp_frac={avg_clamp_frac:.3f} | "
                f"Discrete ents: land={avg_ent_land:.2f}, hire={avg_ent_hire:.2f}, "
                f"sell={avg_ent_sell:.2f}, seed_c={avg_ent_seed_c:.2f}, anim_t={avg_ent_anim_t:.2f} | "
                f"Cont ent={avg_cont_entropy:.2f}"
            )
            # Log metrics to JSONL for visualize_training.py
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
            
            # Compute batch win rates
            wr_fixed = batch_wins_fixed / max(1, batch_total_fixed)
            wr_self = batch_wins_self / max(1, batch_total_self)
            wr_overall = (batch_wins_fixed + batch_wins_self) / max(1, batch_total_fixed + batch_total_self)
            
            log_entry = {
                "episode": episode + 1,
                "loss": avg_loss,
                "actor_loss": avg_actor,
                "critic_loss": avg_critic,
                "entropy_loss_discrete": avg_ent_disc,
                "entropy_loss_continuous": avg_ent_cont,
                "entropy_loss": avg_entropy,
                "money_delta": float(money_delta),
                "buffer_size": last_buffer_size,
                "eps_per_sec": eps_per_sec,
                "opponent": opp_label,
                "lp_mean": avg_lp_mean,
                "lp_std": avg_lp_std,
                "clamp_frac": avg_clamp_frac,
                "discrete_entropy": avg_ent_land + avg_ent_hire + avg_ent_sell + avg_ent_seed_c + avg_ent_anim_t,
                "continuous_entropy": avg_cont_entropy,
                "win_rate_fixed": wr_fixed,
                "win_rate_self": wr_self,
                "win_rate_overall": wr_overall,
            }
            
            import json
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Reset batch win-rate counters
            batch_wins_fixed = 0
            batch_total_fixed = 0
            batch_wins_self = 0
            batch_total_self = 0
                
            running_loss = 0.0
            running_actor_loss = 0.0
            running_critic_loss = 0.0
            running_entropy_loss_disc = 0.0
            running_entropy_loss_cont = 0.0
            running_lp_mean = 0.0
            running_lp_std = 0.0
            running_clamp_frac = 0.0
            running_ent_land = 0.0
            running_ent_hire = 0.0
            running_ent_sell = 0.0
            running_ent_seed_c = 0.0
            running_ent_anim_t = 0.0
            running_cont_entropy = 0.0
            loss_count = 0

        if (episode + 1) % eval_interval == 0:
            if verbose:
                print(f"\n--- Running deterministic evaluation at episode {episode + 1} ---")
            
            eval_agent0 = StrategicTrainingAgent(actor_net_torch=model, player_id=0, deterministic=True)
            model.eval()
            
            eval_metrics = []
            for opp_agent, eval_opp_label in eval_opponents:
                eval_money_deltas = []
                wins = 0
                for _ in range(eval_episodes_per_opponent):
                    eval_seed = random.randint(0, 100000)
                    try:
                        eval_env = make("kaggriculture", configuration={"seed": eval_seed})
                        eval_env.run([eval_agent0, opp_agent])
                        
                        e_final_obs = eval_env.state[0]["observation"]
                        e_m0 = e_final_obs["farms"][0]["money"]
                        e_m1 = e_final_obs["farms"][1]["money"]
                        
                        # Raw bank money delta (matching competition scoring)
                        md = e_m0 - e_m1
                        eval_money_deltas.append(md)
                        if md > 0:
                            wins += 1
                    except Exception as e:
                        if verbose:
                            print(f"  Eval error vs {eval_opp_label}: {e}")
                            
                if eval_money_deltas:
                    mean_md = sum(eval_money_deltas) / len(eval_money_deltas)
                    win_rate = wins / len(eval_money_deltas)
                else:
                    mean_md, win_rate = 0.0, 0.0
                    
                eval_metrics.append({
                    "episode": episode + 1,
                    "opponent": eval_opp_label,
                    "mean_money_delta": mean_md,
                    "win_rate": win_rate
                })
                
                if verbose:
                    print(f"  Eval vs {eval_opp_label:12s} | win_rate: {win_rate:.2f} | mean_delta: {mean_md:+.1f}")
                    
            eval_jsonl_path = os.path.join(output_dir, "logs", "eval_metrics.jsonl")
            with open(eval_jsonl_path, "a", encoding="utf-8") as f:
                for entry in eval_metrics:
                    f.write(json.dumps(entry) + "\n")
            if verbose:
                print("----------------------------------------------------------\n")

        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"actor_net_ep{episode + 1}.npz")
            layers = model.export_numpy_weights()
            save_weights(layers, ckpt_path)
            if verbose:
                print(f"  -> Checkpoint saved: {ckpt_path}")
                
            # Add snapshot to opponent pool (numpy-only, no gradients needed)
            snapshot_np = ActorNetNumpy(layers)
            snapshot_agent = NumpySnapshotAgent(
                actor_net_numpy=snapshot_np,
                player_id=1,
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
