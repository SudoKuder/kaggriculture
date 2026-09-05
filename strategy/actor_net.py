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

# 64 state features (without the 14-dim plan encoding)
FEATURE_DIM = 64
ACTION_DIM = 38

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

# ---------------------------------------------------------------------------
# PyTorch model (training only)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class ActorNetTorch(nn.Module):
        """64 → 64 → 32 → 38 MLP with ReLU."""

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

        def get_action(self, x, deterministic=False):
            """Returns (sampled_action_dict, log_prob, raw_logits)
            If deterministic=True, returns the mode/mean of distributions.
            """
            logits = self.forward(x)
            
            # 1. buy_land (Bernoulli)
            p_land = torch.sigmoid(logits[..., 0])
            dist_land = Bernoulli(p_land)
            a_land = (p_land > 0.5).float() if deterministic else dist_land.sample()
            lp_land = dist_land.log_prob(a_land)
            
            # 2. hire_target (Categorical 0-12 mapping to 1-13 workers)
            dist_hire = Categorical(logits=logits[..., 1:14])
            a_hire = torch.argmax(logits[..., 1:14], dim=-1) if deterministic else dist_hire.sample()
            lp_hire = dist_hire.log_prob(a_hire)
            
            # 3. sell_hold (9 independent Bernoullis)
            p_sell = torch.sigmoid(logits[..., 14:23])
            dist_sell = Bernoulli(p_sell)
            a_sell = (p_sell > 0.5).float() if deterministic else dist_sell.sample()
            lp_sell = dist_sell.log_prob(a_sell).sum(dim=-1)
            
            # 4. buy_seed_crop (Categorical 0-4)
            dist_seed_c = Categorical(logits=logits[..., 23:28])
            a_seed_c = torch.argmax(logits[..., 23:28], dim=-1) if deterministic else dist_seed_c.sample()
            lp_seed_c = dist_seed_c.log_prob(a_seed_c)
            
            # 5. buy_seed_frac (Normal with fixed std or just deterministic sigmoid + noise)
            mean_seed_f = torch.sigmoid(logits[..., 28])
            std = 0.2
            dist_seed_f = Normal(mean_seed_f, std)
            if deterministic:
                a_seed_f = mean_seed_f
            else:
                a_seed_f = dist_seed_f.sample().clamp(0.0, 1.0)
            lp_seed_f = dist_seed_f.log_prob(a_seed_f)
            
            # 6. buy_animal_type (Categorical 0-2)
            dist_anim_t = Categorical(logits=logits[..., 29:32])
            a_anim_t = torch.argmax(logits[..., 29:32], dim=-1) if deterministic else dist_anim_t.sample()
            lp_anim_t = dist_anim_t.log_prob(a_anim_t)
            
            # 7. buy_animal_frac (Normal)
            mean_anim_f = torch.sigmoid(logits[..., 32])
            dist_anim_f = Normal(mean_anim_f, std)
            if deterministic:
                a_anim_f = mean_anim_f
            else:
                a_anim_f = dist_anim_f.sample().clamp(0.0, 1.0)
            lp_anim_f = dist_anim_f.log_prob(a_anim_f)
            
            # 8. Tactical parameters (Normal distributions)
            mean_weed = torch.sigmoid(logits[..., 33]) * 10.0
            dist_weed = Normal(mean_weed, 1.0)
            a_weed = mean_weed if deterministic else dist_weed.sample().clamp(0.0, 10.0)
            lp_weed = dist_weed.log_prob(a_weed)
            
            mean_maint = torch.sigmoid(logits[..., 34]) * 23.0
            dist_maint = Normal(mean_maint, 2.0)
            a_maint = mean_maint if deterministic else dist_maint.sample().clamp(0.0, 23.0)
            lp_maint = dist_maint.log_prob(a_maint)
            
            mean_panic = torch.sigmoid(logits[..., 35]) * 23.0
            dist_panic = Normal(mean_panic, 2.0)
            a_panic = mean_panic if deterministic else dist_panic.sample().clamp(0.0, 23.0)
            lp_panic = dist_panic.log_prob(a_panic)
            
            mean_seed_m = torch.sigmoid(logits[..., 36]) * 5.0
            dist_seed_m = Normal(mean_seed_m, 0.5)
            a_seed_m = mean_seed_m if deterministic else dist_seed_m.sample().clamp(0.0, 5.0)
            lp_seed_m = dist_seed_m.log_prob(a_seed_m)
            
            mean_land_b = torch.sigmoid(logits[..., 37]) * 2000.0
            dist_land_b = Normal(mean_land_b, 200.0)
            a_land_b = mean_land_b if deterministic else dist_land_b.sample().clamp(0.0, 2000.0)
            lp_land_b = dist_land_b.log_prob(a_land_b)
            
            total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f + lp_weed + lp_maint + lp_panic + lp_seed_m + lp_land_b
            
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
                    params[i].copy_(torch.from_numpy(W.T))
                    params[i + 1].copy_(torch.from_numpy(b))

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
