# Line-by-Line Explanation of `train_strategy.py`

This document provides a detailed explanation of `c:\codes\kaggriculture\train_strategy.py`. This script serves as the top-level CLI entry point for initiating, testing, and managing the training of the agent's strategic layer.

---

## 1. Docstrings and Imports (Lines 1-30)

```python
1: """Top-level training script for the strategic decision layer.
2: 
3: Usage:
4:     # Quick smoke test (5 episodes)
5:     python train_strategy.py --episodes 5 --test
6: 
7:     # Short training run (500 episodes, ~20 min)
8:     python train_strategy.py --episodes 500
9: 
10:     # Full training run (2000 episodes, ~80 min)
11:     python train_strategy.py --episodes 2000
12: 
13:     # After training, export weights for submission:
14:     python -m strategy.export_weights strategy_checkpoints/value_net_final.npz -o strategy_weights_inline.py
15: """
```
- **Lines 1-15**: Module docstring documenting what the script does and providing example CLI usages for testing, short training, long training, and exporting weights into the final Python submission script.

```python
17: import argparse
18: import os
19: import sys
20: import time
```
- **Lines 17-20**: Standard Python imports. `argparse` parses command-line arguments, `os` and `sys` manage file paths and the Python path, and `time` handles tracking how long training takes.

```python
22: # Ensure repo root is on path
23: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
- **Lines 22-23**: Finds the absolute path to the directory containing this script (the repo root) and inserts it at the front of `sys.path`. This guarantees that Python can find the `strategy` package regardless of the current working directory in the terminal.

```python
25: from strategy.train import train_actor_network
26: from strategy.actor_net import (
27:     ActorNetTorch, torch_to_numpy, save_weights, load_weights, ActorNetNumpy,
28: )
29: from strategy.features import extract_features
```
- **Lines 25-29**: Imports the core logic from the `strategy` sub-package. Specifically: the main `train_actor_network` loop, the `actor_net` (both PyTorch and Numpy versions) and serialization functions, and the `extract_features` function used to prepare observations.

---

## 2. Smoke Test Setup & Feature Extraction (Lines 32-65)

```python
32: def smoke_test():
33:     """Quick sanity check that all components work end-to-end."""
34:     print("=" * 60)
35:     print("SMOKE TEST: verifying all components")
36:     print("=" * 60)
```
- **Lines 32-36**: Defines the `smoke_test` function, which is a hardcoded unit test verifying that every major part of the training pipeline functions without crashing.

```python
38:     # 1. Feature extraction
39:     print("\n1. Testing feature extraction...")
40:     from copy import deepcopy
41:     from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS, CROPS, ANIMALS, MARKET_PARAMS
```
- **Lines 38-41**: Starts test phase 1: Feature Extraction. Imports deepcopy and various environment constants directly from the Kaggle environment to mock an observation dictionary.

```python
43:     tiles = [["LOCKED" if (x >= 5 or y >= 5) else None for x in range(10)] for y in range(10)]
44:     farm = {
45:         "money": 3000.0, "farmer": [4, 4], "hands": [],
46:         "unlocked_quadrants": ["NW"], "hires_today": 0, "tiles": tiles,
47:     }
48:     obs = {
49:         "step": 0, "day": 0, "hour": 0, "player": 0,
50:         "farms": [farm, deepcopy(farm)],
51:         "market": {"inventory": {}, "prices": {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}},
52:         "town": {"unlocked_shops": []},
53:         "private": {
54:             "shed": {p: 0 for p in PRODUCTS + list(ANIMALS)},
55:             "seeds": {c: 0 for c in CROPS},
56:             "inventories": [{}],
57:         },
58:     }
```
- **Lines 43-58**: Manually constructs a mock observation state (`obs`) mirroring exactly what the Kaggle runner provides on Step 0. This involves creating a 10x10 tile grid, mocking initial player `money`, mocking the global `market`, and setting up `private` inventories (shed and seeds).

```python
59:     features = extract_features(obs)
60:     features = extract_features(obs)
61:     print(f"   Feature vector: shape={features.shape}, "
62:           f"range=[{features.min():.3f}, {features.max():.3f}]")
63:     assert features.shape == (64,), f"Expected (64,), got {features.shape}"
64:     print("   [OK] Feature extraction OK")
```
- **Lines 59-64**: Calls `extract_features` (called twice, likely an accidental duplication in code but harmless) to parse the mock `obs` into a NumPy array. Logs the shape and min/max values. Asserts the feature vector has exactly 64 elements. Prints success.

---

## 3. Smoke Test: Candidates, Actor Net, Serialization (Lines 66-107)

```python
66:     # 2. Candidate generation
67:     print("\n2. Testing candidate generation...")
68:     from strategy.candidates import generate_candidates
69:     candidates = generate_candidates(obs)
70:     print(f"   Generated {len(candidates)} candidates: "
71:           f"{[c['label'] for c in candidates]}")
72:     assert len(candidates) >= 2, "Need at least 2 candidates"
73:     assert candidates[0]["label"] == "baseline", "First candidate must be baseline"
74:     print("   [OK] Candidate generation OK")
```
- **Lines 66-74**: Starts test phase 2: Candidates. Tests `generate_candidates()` which produces possible hand-coded heuristic plans. Asserts that it generates at least 2 candidates and the first one is the "baseline" (doing nothing special).

```python
76:     # 3. Actor network
77:     print("\n3. Testing actor network...")
78:     import torch
79:     model = ActorNetTorch()
80:     x = torch.randn(1, 64)
81:     y = model(x)
82:     print(f"   PyTorch forward: input={x.shape} -> output={y.shape}")
```
- **Lines 76-82**: Starts test phase 3: Actor Network. Instantiates an untrained PyTorch `ActorNetTorch`. Pushes a random noise vector `x` of shape `(1, 64)` through the model to verify the forward pass doesn't throw a shape error.

```python
84:     # Numpy equivalence
85:     vnet_np = torch_to_numpy(model)
86:     action_np = vnet_np.predict(features)
87:     action_dict, _, _ = model.get_action(torch.tensor(features).unsqueeze(0), deterministic=True)
88:     diff = abs(action_np["buy_seed_frac"] - torch.sigmoid(model(torch.tensor(features).unsqueeze(0))[..., 21]).item())
89:     print(f"   Numpy forward: val={action_np['buy_seed_frac']:.4f}, PyTorch: diff={diff:.6f}")
90:     assert diff < 1e-4, f"Numpy/PyTorch mismatch: {diff}"
91:     print("   [OK] Actor network OK")
```
- **Lines 84-91**: Crucial validation step. Converts the PyTorch model to a raw Numpy model using `torch_to_numpy()`. Runs the same features through *both* models. Calculates the difference in output for a specific node (`buy_seed_frac`, which happens to be node 21 in the final tensor). Asserts the outputs match within `1e-4`, proving the Numpy port operates identically to the PyTorch original.

```python
93:     # 4. Weight save/load
94:     print("\n4. Testing weight serialization...")
95:     os.makedirs("strategy_checkpoints", exist_ok=True)
96:     test_path = "strategy_checkpoints/test_weights.npz"
97:     layers = model.export_numpy_weights()
98:     save_weights(layers, test_path)
99:     loaded = load_weights(test_path)
100:     vnet_loaded = ActorNetNumpy(loaded)
101:     action_loaded = vnet_loaded.predict(features)
102:     diff2 = abs(action_loaded["buy_seed_frac"] - action_np["buy_seed_frac"])
103:     print(f"   Save/load roundtrip: diff={diff2:.6f}")
104:     assert diff2 < 1e-6, f"Save/load mismatch: {diff2}"
105:     os.remove(test_path)
106:     print("   [OK] Weight serialization OK")
```
- **Lines 93-106**: Starts test phase 4: Serialization. Exports the model weights to Numpy, saves them to disk as an `.npz` file, loads them back off disk, instantiates a new Numpy network, and passes data through it. Asserts the loaded network's output perfectly matches the original, proving that save/load handles the dictionaries and array shapes properly. Deletes the test `.npz` file afterward.

---

## 4. Smoke Test: Opponent Pool (Lines 108-120)

```python
108:     # 5. Opponent pool
109:     print("\n5. Testing opponent pool...")
110:     from strategy.opponents import OpponentPool
111:     pool = OpponentPool()
112:     print(f"   Pool size: {len(pool)} opponents")
113:     agent, label = pool.sample()
114:     print(f"   Sampled: {label}")
115:     assert len(pool) >= 3, f"Expected >= 3 opponents, got {len(pool)}"
116:     print("   [OK] Opponent pool OK")
117: 
118:     print("\n" + "=" * 60)
119:     print("ALL SMOKE TESTS PASSED")
120:     print("=" * 60)
```
- **Lines 108-120**: Starts test phase 5: Opponent Pool. Initializes the `OpponentPool` which handles drawing opponents for self-play. Checks that it seeds properly (at least 3 default opponents: starter, passive, heuristic). Confirms sampling an opponent works. Concludes the smoke test function with success banners.

---

## 5. Main CLI Entry Point (Lines 123-159)

```python
123: def main():
124:     parser = argparse.ArgumentParser(
125:         description="Train the strategic decision layer"
126:     )
```
- **Lines 123-126**: Defines the script's `main()` entrypoint. Instantiates `argparse` to handle command line flags.

```python
127:     parser.add_argument(
128:         "--episodes", type=int, default=500,
129:         help="Number of training episodes (default: 500)"
130:     )
131:     parser.add_argument(
132:         "--batch-size", type=int, default=32,
133:         help="Mini-batch size (default: 32)"
134:     )
135:     parser.add_argument(
136:         "--lr", type=float, default=1e-3,
137:         help="Learning rate (default: 0.001)"
138:     )
139:     parser.add_argument(
140:         "--rollout-days", type=int, default=2,
141:         help="Rollout depth in days (default: 2)"
142:     )
143:     parser.add_argument(
144:         "--checkpoint-interval", type=int, default=100,
145:         help="Save checkpoint every N episodes (default: 100)"
146:     )
147:     parser.add_argument(
148:         "--output-dir", type=str, default="strategy_checkpoints",
149:         help="Checkpoint directory (default: strategy_checkpoints)"
150:     )
151:     parser.add_argument(
152:         "--test", action="store_true",
153:         help="Run smoke test instead of training"
154:     )
155:     args = parser.parse_args()
```
- **Lines 127-155**: Declares the CLI arguments. You can customize the number of episodes, batch size, learning rate (default 1e-3), rollout horizon (currently unused by train.py, but likely a legacy param or intended for tree search), how frequently to save the network to disk, and what directory to save it to. Adds a `--test` boolean flag to trigger the smoke test.

```python
157:     if args.test:
158:         smoke_test()
159:         return
```
- **Lines 157-159**: If `--test` was provided, executes `smoke_test()` and immediately exits.

---

## 6. Training Execution (Lines 161-192)

```python
161:     print("=" * 60)
162:     print("STRATEGIC LAYER TRAINING")
163:     print("=" * 60)
164:     print(f"Episodes:   {args.episodes}")
165:     print(f"Batch size: {args.batch_size}")
166:     print(f"LR:         {args.lr}")
167:     print(f"Rollout:    {args.rollout_days} days")
168:     print(f"Output:     {args.output_dir}")
169:     print("=" * 60)
```
- **Lines 161-169**: Prints a summary header into the console to visually confirm to the user what parameters the training run is actually using.

```python
171:     t0 = time.time()
172:     model = train_actor_network(
173:         num_episodes=args.episodes,
174:         batch_size=args.batch_size,
175:         lr=args.lr,
176:         checkpoint_interval=args.checkpoint_interval,
177:         output_dir=args.output_dir,
178:         verbose=True,
179:     )
180:     elapsed = time.time() - t0
```
- **Lines 171-180**: Records the start time. Calls the heavy-lifting `train_actor_network` function (imported from `strategy/train.py`), passing in all user-specified arguments. This function runs the long REINFORCE loop. Records the end time once it returns.

```python
182:     print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
183:     print(f"Weights saved to: {args.output_dir}/actor_net_final.npz")
184:     print("\nNext steps:")
185:     print(f"  1. Run benchmark: python benchmark.py")
186:     print(f"  2. Export for submission:")
187:     print(f"     python -m strategy.export_weights {args.output_dir}/actor_net_final.npz -o strategy_weights_inline.py")
```
- **Lines 182-187**: Computes total runtime in minutes. Prints helpful terminal instructions indicating what the user should do next (benchmarking the new model, and eventually exporting its weights to be bundled into the kaggle submission script).

```python
190: if __name__ == "__main__":
191:     main()
192: 
```
- **Lines 190-192**: Standard Python idiom to ensure `main()` is only executed if the script is run directly via command-line (e.g. `python train_strategy.py`), preventing it from triggering if the script is imported by another module.
