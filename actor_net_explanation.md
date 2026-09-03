# Line-by-Line Explanation of `actor_net.py`

This document provides a detailed explanation of `c:\codes\kaggriculture\strategy\actor_net.py`. This file defines the neural network architectures (in both PyTorch and NumPy) used to make strategic decisions.

---

## 1. Docstrings and Imports (Lines 1-18)

```python
1: """Actor network for the strategic decision layer.
2: 
3: Small MLP mapping state features → continuous action space (26 dims).
4: Training uses PyTorch (REINFORCE); inference uses a pure-numpy forward pass
5: so the submission stays lightweight and has no cold-start penalty.
6: """
```
- **Lines 1-6**: Explains the purpose of the file. It implements a small Multi-Layer Perceptron (MLP). It explicitly notes a dual-architecture design: PyTorch is used for fast training with automatic differentiation, while a pure-NumPy equivalent is used during inference to avoid heavy dependencies and slow load times in the Kaggle environment.

```python
8: import numpy as np
9: import math
10: 
11: try:
12:     import torch
13:     import torch.nn as nn
14:     from torch.distributions import Categorical, Bernoulli, Normal
15:     HAS_TORCH = True
16: except ImportError:
17:     HAS_TORCH = False
```
- **Lines 8-17**: Imports NumPy and Math unconditionally. Tries to import PyTorch components (nn, distributions). If PyTorch is missing (like on the Kaggle submission servers), it catches the `ImportError` and sets `HAS_TORCH = False` instead of crashing. This allows the file to be safely imported during the final game.

---

## 2. Constants and Output Mapping (Lines 19-36)

```python
19: # 64 state features (without the 14-dim plan encoding)
20: FEATURE_DIM = 64
21: ACTION_DIM = 31
```
- **Lines 19-21**: Defines the fixed dimensions for the network. It takes 64 input features and outputs an array of 31 raw numbers (logits) that encode various actions.

```python
23: # Output dimension indices:
24: # [0] buy_land logit (sigmoid)
25: # [1:7] hire_target logits (softmax over 1-6)
...
35: # [30] land_unlock_buffer logit (sigmoid -> scale [0, 2000])
```
- **Lines 23-35**: Documentation mapping what each of the 31 output indices corresponds to. For example, index `0` represents the desire to buy land, `1:7` represent the probability distribution over hiring 1 to 6 workers, and `30` is scaled up to 2000 to represent a monetary threshold for land unlocking.

---

## 3. PyTorch Model Definition (Lines 41-174)

```python
41: if HAS_TORCH:
42:     class ActorNetTorch(nn.Module):
43:         """64 → 64 → 32 → 26 MLP with ReLU."""
```
- **Lines 41-43**: Only defines the PyTorch model if PyTorch was successfully imported. (Note: docstring says 26 outputs, but the code actually uses 31. This is a stale comment).

```python
45:         def __init__(self, input_dim=FEATURE_DIM, output_dim=ACTION_DIM):
46:             super().__init__()
47:             self.net = nn.Sequential(
48:                 nn.Linear(input_dim, 64),
49:                 nn.ReLU(),
50:                 nn.Linear(64, 32),
51:                 nn.ReLU(),
52:                 nn.Linear(32, output_dim),
53:             )
```
- **Lines 45-53**: The constructor. Calls the parent `nn.Module` constructor. Defines a simple feed-forward neural network using `nn.Sequential`. It has three linear layers (64 -> 64 -> 32 -> 31) separated by ReLU (Rectified Linear Unit) activation functions.

```python
55:         def forward(self, x):
56:             return self.net(x)
```
- **Lines 55-56**: The standard PyTorch forward pass, which simply runs input `x` through the Sequential network.

```python
58:         def get_action(self, x, deterministic=False):
...
62:             logits = self.forward(x)
```
- **Lines 58-62**: The `get_action` method wraps the forward pass. It takes state `x` and a `deterministic` flag. It pushes `x` through the network to get the raw `logits`.

```python
64:             # 1. buy_land (Bernoulli)
65:             p_land = torch.sigmoid(logits[..., 0])
66:             dist_land = Bernoulli(p_land)
67:             a_land = (p_land > 0.5).float() if deterministic else dist_land.sample()
68:             lp_land = dist_land.log_prob(a_land)
```
- **Lines 64-68**: Converts logit `[0]` into a probability using `sigmoid`. Wraps it in a Bernoulli (coin-flip) distribution. If `deterministic` is true, it strictly chooses True if probability > 50%. Otherwise, it rolls the weighted dice. It then records the log probability of that specific choice.

```python
70:             # 2. hire_target (Categorical 0-5 mapping to 1-6 workers)
71:             dist_hire = Categorical(logits=logits[..., 1:7])
72:             a_hire = torch.argmax(logits[..., 1:7], dim=-1) if deterministic else dist_hire.sample()
73:             lp_hire = dist_hire.log_prob(a_hire)
```
- **Lines 70-73**: Treats logits `[1:7]` as 6 mutually exclusive classes. Uses `Categorical` to calculate the softmax distribution. Selects the `argmax` (highest probability) if deterministic, or samples stochastically otherwise. Records log probability.

```python
75:             # 3. sell_hold (9 independent Bernoullis)
...
79:             lp_sell = dist_sell.log_prob(a_sell).sum(dim=-1)
```
- **Lines 75-79**: Evaluates logits `[7:16]` as 9 independent binary choices. Sums their log probabilities along the last dimension because they are treated as a single composite action.

```python
81:             # 4. buy_seed_crop (Categorical 0-4)
...
84:             lp_seed_c = dist_seed_c.log_prob(a_seed_c)
```
- **Lines 81-84**: Categorical selection for which crop type to buy out of 5 options.

```python
86:             # 5. buy_seed_frac (Normal with fixed std or just deterministic sigmoid + noise)
87:             mean_seed_f = torch.sigmoid(logits[..., 21])
88:             std = 0.2
89:             dist_seed_f = Normal(mean_seed_f, std)
90:             if deterministic:
91:                 a_seed_f = mean_seed_f
92:             else:
93:                 a_seed_f = dist_seed_f.sample().clamp(0.0, 1.0)
94:             lp_seed_f = dist_seed_f.log_prob(a_seed_f)
```
- **Lines 86-94**: Handles continuous values. Uses sigmoid to squash the logit to [0, 1], acting as the `mean`. Creates a Normal (Gaussian) distribution with a fixed standard deviation of 0.2. Samples from it and `clamp`s the result so it doesn't drop below 0 or above 1.

```python
96:             # 6. buy_animal_type (Categorical 0-2)
...
108:             lp_anim_f = dist_anim_f.log_prob(a_anim_f)
```
- **Lines 96-108**: Repeats the categorical and continuous logic for picking animal types and computing the fraction of money to spend on them.

```python
110:             # 8. Tactical parameters (Normal distributions)
111:             mean_weed = torch.sigmoid(logits[..., 26]) * 10.0
...
134:             lp_land_b = dist_land_b.log_prob(a_land_b)
```
- **Lines 110-134**: Parses the remaining continuous parameters (weed penalty, maintenance hour, panic hour, seed multipliers, land buffers). These work identically to the continuous fraction logic above, except the sigmoid outputs are multiplied by scalars (e.g., `* 10.0`, `* 23.0`, `* 2000.0`) to stretch the [0,1] range into the specific domain bounds needed by the heuristic logic.

```python
136:             total_log_prob = lp_land + lp_hire + ... + lp_land_b
...
152:             return action_dict, total_log_prob, logits
```
- **Lines 136-152**: Sums up the log probabilities of all independent decisions to get the total likelihood of this specific combinatorial action. Packages all sampled actions into a dictionary and returns it alongside the probabilities and raw network outputs.

```python
154:         def export_numpy_weights(self):
155:             """Extract weights as a list of (W, b, has_relu) tuples."""
156:             layers = []
157:             params = list(self.parameters())
158:             for i in range(0, len(params), 2):
159:                 W = params[i].detach().cpu().numpy()
160:                 b = params[i + 1].detach().cpu().numpy()
161:                 has_relu = (i + 2 < len(params))
162:                 layers.append((W.T, b, has_relu))
163:             return layers
```
- **Lines 154-163**: Implements serialization. Iterates over all PyTorch parameters (which are stored sequentially as Weight, Bias, Weight, Bias...). Transposes the weights `W.T` to match typical NumPy dot-product expectations. Detects if it's the last layer or not to determine if a ReLU activation should follow. Returns a clean list of NumPy tuples.

```python
165:         def import_numpy_weights(self, layers):
...
173:                     params[i].copy_(torch.from_numpy(W.T))
174:                     params[i + 1].copy_(torch.from_numpy(b))
```
- **Lines 165-174**: The reverse of export. Iterates over the provided NumPy layers and safely copies the raw data into the PyTorch parameters using `copy_` inside a `no_grad()` block to avoid messing up the optimizer state.

---

## 4. NumPy-only Inference Model (Lines 179-218)

```python
179: class ActorNetNumpy:
180:     """Pure-numpy MLP for inference. Loaded from exported weights."""
181: 
182:     def __init__(self, layers=None):
183:         if layers is not None:
184:             self.layers = layers
185:         else:
...
190:             ]
```
- **Lines 179-190**: Defines the dependency-free version of the network. If `layers` aren't provided, it initializes itself with random Gaussian noise to prevent crashing if loaded improperly.

```python
192:     def _sigmoid(self, x):
193:         return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
```
- **Lines 192-193**: A mathematically stable implementation of the Sigmoid function. It uses `np.clip` to prevent numeric overflow errors inside `np.exp` when values become too extremely positive or negative.

```python
195:     def predict(self, features):
196:         """Forward pass and return deterministic action."""
197:         x = features.astype(np.float32)
198:         for W, b, relu in self.layers:
199:             x = x @ W + b
200:             if relu:
201:                 x = np.maximum(x, 0.0)
202:         
203:         logits = x.squeeze() if x.ndim > 1 else x
```
- **Lines 195-203**: The NumPy forward pass. Iterates through the stored matrices. Computes `(x matrix-multiply W) + bias`. If the layer has a ReLU, applies `np.maximum(x, 0.0)`. Finally, squeezes out empty dimensions.

```python
205:         return {
206:             "buy_land": self._sigmoid(logits[0]),
207:             "hire_target": np.argmax(logits[1:7]),
...
217:             "land_unlock_buffer": self._sigmoid(logits[30]) * 2000.0,
218:         }
```
- **Lines 205-218**: Perfectly replicates the "deterministic" branch of the PyTorch `get_action` logic, utilizing raw NumPy functions (`_sigmoid`, `np.argmax`) to decode the logits into a usable action dictionary.

---

## 5. Weight Serialization Utilities (Lines 224-247)

```python
224: def save_weights(layers, path):
225:     arrays = {}
226:     for i, (W, b, has_relu) in enumerate(layers):
227:         arrays[f"W_{i}"] = W
228:         arrays[f"b_{i}"] = b
229:         arrays[f"relu_{i}"] = np.array([has_relu])
230:     np.savez_compressed(path, **arrays)
```
- **Lines 224-230**: Flattens the list of layer tuples into a single dictionary using naming conventions (e.g., `W_0`, `b_0`). Saves this dictionary securely to disk as a `.npz` ZIP archive using `np.savez_compressed()`.

```python
232: def load_weights(path):
233:     data = np.load(path)
234:     layers = []
235:     i = 0
236:     while f"W_{i}" in data:
237:         W = data[f"W_{i}"]
...
242:     return layers
```
- **Lines 232-242**: Reads an `.npz` file. Uses a `while` loop to dynamically determine how many layers are saved in the file, packing them back up into tuples and returning the reconstructed list.

```python
244: def torch_to_numpy(model):
245:     layers = model.export_numpy_weights()
246:     return ActorNetNumpy(layers)
```
- **Lines 244-247**: Convenience helper function. Takes a live PyTorch model, exports its weights to NumPy lists, and injects them into a new `ActorNetNumpy` instance, returning the converted model.
