"""Actor network for the strategic decision layer.

Small MLP mapping state features → continuous action space (38 dims).
Training uses PyTorch (REINFORCE); inference uses a pure-numpy forward pass
so the submission stays lightweight and has no cold-start penalty.
"""

import numpy as np
import math

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical, Bernoulli, Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import FEATURE_DIM from the single source of truth; fall back to 64
# for Kaggle submission context where strategy package may not exist.
try:
    from strategy.features import FEATURE_DIM
except ImportError:
    FEATURE_DIM = 64

ACTION_DIM = 45

# Output dimension indices:
# [0] buy_land logit (sigmoid)
# [1:14] hire_target logits (softmax over 1-13)
# [14:23] sell_hold logits (9 independent sigmoids)
# [23:28] buy_seed_crop logits (softmax over 5 crops)
# [28] buy_seed_frac logit (sigmoid)
# [29:32] buy_animal_type logits (softmax over 3 animals)
# [32] buy_animal_frac logit (sigmoid)
# [33] weed_penalty logit (sigmoid -> scale [0, 10])
# [34] maint_water_hour logit (sigmoid -> scale [0, 23])
# [35] panic_drop_hour logit (sigmoid -> scale [0, 23])
# [36] seed_threshold_mult logit (sigmoid -> scale [0, 5])
# [37] land_unlock_buffer logit (sigmoid -> scale [0, 2000])
# [38] buy_seed_frac log_std
# [39] buy_animal_frac log_std
# [40] weed_penalty log_std
# [41] maint_water_hour log_std
# [42] panic_drop_hour log_std
# [43] seed_threshold_mult log_std
# [44] land_unlock_buffer log_std

# ---------------------------------------------------------------------------
# PyTorch model (training only)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class ActorNetTorch(nn.Module):
        """64 → 64 → 32 → 45 MLP with ReLU."""

        # Per-head minimum stds for continuous action heads, set to ~5-10%
        # of each head's range.  A single MIN_STD doesn't work because
        # heads span vastly different scales (e.g. buy_seed_frac in [0,1]
        # vs maint_water_hour in [0,23] vs land_unlock_buffer in [0,2000]).
        # With the original 0.01 (or even a uniform 0.05), wide-range heads
        # produce extreme log-probs that collapse gradients.
        MIN_STD_SEED_F  = 0.05   # range [0, 1]
        MIN_STD_ANIM_F  = 0.05   # range [0, 1]
        MIN_STD_WEED    = 0.5    # range [0, 10]
        MIN_STD_MAINT   = 1.15   # range [0, 23]
        MIN_STD_PANIC   = 1.15   # range [0, 23]
        MIN_STD_SEED_M  = 0.25   # range [0, 5]
        MIN_STD_LAND_B  = 100.0  # range [0, 2000]

        def __init__(self, input_dim=FEATURE_DIM, output_dim=ACTION_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            return self.net(x)

        def get_distributions(self, logits):
            """Build all action distributions from raw network logits.

            Returns a dict of named distributions, grouped into two lists
            for separate entropy accounting:
              - discrete_dists: list of (name, dist) for discrete heads
              - continuous_dists: list of (name, dist) for continuous heads

            This is the single source of truth for distribution construction;
            both ``get_action`` and the training loop must use this method so
            distribution parameters (especially min-std) stay in sync.
            """
            # --- Discrete heads ---
            dist_land = Bernoulli(torch.sigmoid(logits[..., 0]))
            dist_hire = Categorical(logits=logits[..., 1:14])
            dist_sell = Bernoulli(torch.sigmoid(logits[..., 14:23]))
            dist_seed_c = Categorical(logits=logits[..., 23:28])
            dist_anim_t = Categorical(logits=logits[..., 29:32])

            discrete_dists = [
                ("land", dist_land),
                ("hire", dist_hire),
                ("sell", dist_sell),
                ("seed_c", dist_seed_c),
                ("anim_t", dist_anim_t),
            ]

            # --- Continuous heads (per-head min-std) ---
            dist_seed_f = Normal(
                torch.sigmoid(logits[..., 28]),
                torch.nn.functional.softplus(logits[..., 38]).clamp(self.MIN_STD_SEED_F, 1.0),
            )
            dist_anim_f = Normal(
                torch.sigmoid(logits[..., 32]),
                torch.nn.functional.softplus(logits[..., 39]).clamp(self.MIN_STD_ANIM_F, 1.0),
            )
            dist_weed = Normal(
                torch.sigmoid(logits[..., 33]) * 10.0,
                torch.nn.functional.softplus(logits[..., 40]).clamp(self.MIN_STD_WEED, 5.0),
            )
            dist_maint = Normal(
                torch.sigmoid(logits[..., 34]) * 23.0,
                torch.nn.functional.softplus(logits[..., 41]).clamp(self.MIN_STD_MAINT, 10.0),
            )
            dist_panic = Normal(
                torch.sigmoid(logits[..., 35]) * 23.0,
                torch.nn.functional.softplus(logits[..., 42]).clamp(self.MIN_STD_PANIC, 10.0),
            )
            dist_seed_m = Normal(
                torch.sigmoid(logits[..., 36]) * 5.0,
                torch.nn.functional.softplus(logits[..., 43]).clamp(self.MIN_STD_SEED_M, 2.5),
            )
            dist_land_b = Normal(
                torch.sigmoid(logits[..., 37]) * 2000.0,
                torch.nn.functional.softplus(logits[..., 44]).clamp(self.MIN_STD_LAND_B, 1000.0),
            )

            continuous_dists = [
                ("seed_f", dist_seed_f),
                ("anim_f", dist_anim_f),
                ("weed", dist_weed),
                ("maint", dist_maint),
                ("panic", dist_panic),
                ("seed_m", dist_seed_m),
                ("land_b", dist_land_b),
            ]

            return discrete_dists, continuous_dists

        def get_action(self, x, deterministic=False):
            """Returns (sampled_action_dict, log_prob, raw_logits)
            If deterministic=True, returns the mode/mean of distributions.
            """
            logits = self.forward(x)
            discrete_dists, continuous_dists = self.get_distributions(logits)

            # Unpack discrete distributions
            dist_land   = discrete_dists[0][1]
            dist_hire   = discrete_dists[1][1]
            dist_sell   = discrete_dists[2][1]
            dist_seed_c = discrete_dists[3][1]
            dist_anim_t = discrete_dists[4][1]

            # Unpack continuous distributions
            dist_seed_f = continuous_dists[0][1]
            dist_anim_f = continuous_dists[1][1]
            dist_weed   = continuous_dists[2][1]
            dist_maint  = continuous_dists[3][1]
            dist_panic  = continuous_dists[4][1]
            dist_seed_m = continuous_dists[5][1]
            dist_land_b = continuous_dists[6][1]

            # --- Sample actions ---
            # 1. buy_land (Bernoulli)
            p_land = dist_land.probs
            a_land = (p_land > 0.5).float() if deterministic else dist_land.sample()
            lp_land = dist_land.log_prob(a_land)

            # 2. hire_target (Categorical)
            a_hire = torch.argmax(logits[..., 1:14], dim=-1) if deterministic else dist_hire.sample()
            lp_hire = dist_hire.log_prob(a_hire)

            # 3. sell_hold (9 independent Bernoullis)
            p_sell = dist_sell.probs
            a_sell = (p_sell > 0.5).float() if deterministic else dist_sell.sample()
            lp_sell = dist_sell.log_prob(a_sell).sum(dim=-1)

            # 4. buy_seed_crop (Categorical)
            a_seed_c = torch.argmax(logits[..., 23:28], dim=-1) if deterministic else dist_seed_c.sample()
            lp_seed_c = dist_seed_c.log_prob(a_seed_c)

            # 5. buy_seed_frac (Normal)
            a_seed_f = dist_seed_f.mean if deterministic else dist_seed_f.sample().clamp(0.0, 1.0)
            lp_seed_f = dist_seed_f.log_prob(a_seed_f)

            # 6. buy_animal_type (Categorical)
            a_anim_t = torch.argmax(logits[..., 29:32], dim=-1) if deterministic else dist_anim_t.sample()
            lp_anim_t = dist_anim_t.log_prob(a_anim_t)

            # 7. buy_animal_frac (Normal)
            a_anim_f = dist_anim_f.mean if deterministic else dist_anim_f.sample().clamp(0.0, 1.0)
            lp_anim_f = dist_anim_f.log_prob(a_anim_f)

            # 8. Tactical parameters (Normal distributions)
            a_weed = dist_weed.mean if deterministic else dist_weed.sample().clamp(0.0, 10.0)
            lp_weed = dist_weed.log_prob(a_weed)

            a_maint = dist_maint.mean if deterministic else dist_maint.sample().clamp(0.0, 23.0)
            lp_maint = dist_maint.log_prob(a_maint)

            a_panic = dist_panic.mean if deterministic else dist_panic.sample().clamp(0.0, 23.0)
            lp_panic = dist_panic.log_prob(a_panic)

            a_seed_m = dist_seed_m.mean if deterministic else dist_seed_m.sample().clamp(0.0, 5.0)
            lp_seed_m = dist_seed_m.log_prob(a_seed_m)

            a_land_b = dist_land_b.mean if deterministic else dist_land_b.sample().clamp(0.0, 2000.0)
            lp_land_b = dist_land_b.log_prob(a_land_b)

            total_log_prob = (lp_land + lp_hire + lp_sell + lp_seed_c +
                              lp_seed_f + lp_anim_t + lp_anim_f +
                              lp_weed + lp_maint + lp_panic +
                              lp_seed_m + lp_land_b)

            action_dict = {
                "buy_land": a_land,
                "hire_target": a_hire,
                "sell_hold": a_sell,
                "buy_seed_crop": a_seed_c,
                "buy_seed_frac": a_seed_f,
                "buy_animal_type": a_anim_t,
                "buy_animal_frac": a_anim_f,
                "weed_penalty": a_weed,
                "maint_water_hour": a_maint,
                "panic_drop_hour": a_panic,
                "seed_threshold_mult": a_seed_m,
                "land_unlock_buffer": a_land_b,
            }
            return action_dict, total_log_prob, logits

        def export_numpy_weights(self):
            """Extract weights as a list of (W, b, has_relu) tuples."""
            layers = []
            params = list(self.parameters())
            for i in range(0, len(params), 2):
                W = params[i].detach().cpu().numpy()
                b = params[i + 1].detach().cpu().numpy()
                has_relu = (i + 2 < len(params))
                layers.append((W.T, b, has_relu))
            return layers

        def import_numpy_weights(self, layers):
            """Load weights from a list of (W, b, has_relu) tuples."""
            params = list(self.parameters())
            for i in range(0, len(params), 2):
                layer_idx = i // 2
                W, b, _ = layers[layer_idx]
                with torch.no_grad():
                    w_tensor = torch.from_numpy(W.T)
                    b_tensor = torch.from_numpy(b)
                    
                    # Handle shape mismatches (e.g., resuming 38-dim checkpoint into 45-dim model)
                    out_old, in_old = w_tensor.shape
                    out_new, in_new = params[i].shape
                    
                    min_out = min(out_old, out_new)
                    min_in = min(in_old, in_new)
                    
                    params[i][:min_out, :min_in].copy_(w_tensor[:min_out, :min_in])
                    params[i + 1][:min_out].copy_(b_tensor[:min_out])

# ---------------------------------------------------------------------------
# Numpy-only inference (submission)
# ---------------------------------------------------------------------------

class ActorNetNumpy:
    """Pure-numpy MLP for inference. Loaded from exported weights."""

    def __init__(self, layers=None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers = [
                (np.random.randn(FEATURE_DIM, 64).astype(np.float32) * 0.1, np.zeros(64, dtype=np.float32), True),
                (np.random.randn(64, 32).astype(np.float32) * 0.1, np.zeros(32, dtype=np.float32), True),
                (np.random.randn(32, ACTION_DIM).astype(np.float32) * 0.1, np.zeros(ACTION_DIM, dtype=np.float32), False),
            ]

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def predict(self, features):
        """Forward pass and return deterministic action."""
        x = features.astype(np.float32)
        for W, b, relu in self.layers:
            x = x @ W + b
            if relu:
                x = np.maximum(x, 0.0)
        
        logits = x.squeeze() if x.ndim > 1 else x
        
        return {
            "buy_land": self._sigmoid(logits[0]),
            "hire_target": np.argmax(logits[1:14]),
            "sell_hold": self._sigmoid(logits[14:23]),
            "buy_seed_crop": np.argmax(logits[23:28]),
            "buy_seed_frac": self._sigmoid(logits[28]),
            "buy_animal_type": np.argmax(logits[29:32]),
            "buy_animal_frac": self._sigmoid(logits[32]),
            "weed_penalty": self._sigmoid(logits[33]) * 10.0,
            "maint_water_hour": self._sigmoid(logits[34]) * 23.0,
            "panic_drop_hour": self._sigmoid(logits[35]) * 23.0,
            "seed_threshold_mult": self._sigmoid(logits[36]) * 5.0,
            "land_unlock_buffer": self._sigmoid(logits[37]) * 2000.0,
        }

# ---------------------------------------------------------------------------
# Weight serialization
# ---------------------------------------------------------------------------

def save_weights(layers, path):
    arrays = {}
    for i, (W, b, has_relu) in enumerate(layers):
        arrays[f"W_{i}"] = W
        arrays[f"b_{i}"] = b
        arrays[f"relu_{i}"] = np.array([has_relu])
    np.savez_compressed(path, **arrays)

def load_weights(path):
    data = np.load(path)
    layers = []
    i = 0
    while f"W_{i}" in data:
        W = data[f"W_{i}"]
        b = data[f"b_{i}"]
        has_relu = bool(data[f"relu_{i}"][0])
        layers.append((W, b, has_relu))
        i += 1
    return layers

def torch_to_numpy(model):
    layers = model.export_numpy_weights()
    return ActorNetNumpy(layers)
