"""Value network for the strategic decision layer.

Small MLP mapping state features → scalar (predicted end-of-game money
delta).  Training uses PyTorch; inference uses a pure-numpy forward pass
so the submission stays lightweight and has no cold-start penalty.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .features import FEATURE_DIM

# ---------------------------------------------------------------------------
# PyTorch model (training only)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class ValueNetTorch(nn.Module):
        """64 → 64 → 32 → 1 MLP with ReLU."""

        def __init__(self, input_dim=FEATURE_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def export_numpy_weights(self):
            """Extract weights as a list of (W, b, has_relu) tuples."""
            layers = []
            params = list(self.parameters())
            # params come in pairs: weight, bias for each Linear
            for i in range(0, len(params), 2):
                W = params[i].detach().cpu().numpy()    # shape (out, in)
                b = params[i + 1].detach().cpu().numpy()  # shape (out,)
                has_relu = (i + 2 < len(params))  # relu after all but last
                layers.append((W.T, b, has_relu))  # transpose to (in, out)
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

class ValueNetNumpy:
    """Pure-numpy MLP for inference.  Loaded from exported weights."""

    def __init__(self, layers=None):
        """
        Args:
            layers: list of (W, b, has_relu) tuples where W is (in, out).
                    If None, initialises with random weights (for testing).
        """
        if layers is not None:
            self.layers = layers
        else:
            # Random init for testing
            self.layers = [
                (np.random.randn(FEATURE_DIM, 64).astype(np.float32) * 0.1,
                 np.zeros(64, dtype=np.float32), True),
                (np.random.randn(64, 32).astype(np.float32) * 0.1,
                 np.zeros(32, dtype=np.float32), True),
                (np.random.randn(32, 1).astype(np.float32) * 0.1,
                 np.zeros(1, dtype=np.float32), False),
            ]

    def predict(self, features):
        """Forward pass.  features: (FEATURE_DIM,) → scalar."""
        x = features.astype(np.float32)
        for W, b, relu in self.layers:
            x = x @ W + b
            if relu:
                x = np.maximum(x, 0.0)
        return float(x[0]) if x.ndim == 1 else float(x.squeeze())

    def predict_batch(self, features_batch):
        """Forward pass on a batch.  features_batch: (N, FEATURE_DIM) → (N,)."""
        x = features_batch.astype(np.float32)
        for W, b, relu in self.layers:
            x = x @ W + b
            if relu:
                x = np.maximum(x, 0.0)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Weight serialization
# ---------------------------------------------------------------------------

def save_weights(layers, path):
    """Save numpy weight layers to a .npz file."""
    arrays = {}
    for i, (W, b, has_relu) in enumerate(layers):
        arrays[f"W_{i}"] = W
        arrays[f"b_{i}"] = b
        arrays[f"relu_{i}"] = np.array([has_relu])
    np.savez_compressed(path, **arrays)


def load_weights(path):
    """Load numpy weight layers from a .npz file."""
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
    """Convert a PyTorch ValueNetTorch to a ValueNetNumpy."""
    layers = model.export_numpy_weights()
    return ValueNetNumpy(layers)
