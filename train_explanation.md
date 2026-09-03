# Line-by-Line Explanation of `train.py`

This document provides a detailed explanation of `c:\codes\kaggriculture\strategy\train.py`. The file implements the self-play training loop using a REINFORCE policy gradient algorithm. 

---

## 1. Docstrings and Imports (Lines 1-35)

```python
1: """Self-play training loop with REINFORCE policy gradient.
2: 
3: Core idea: at each day boundary (hour == 0), we:
4: 1. Extract state features
5: 2. Sample an action vector from the Actor network
6: 3. Record (features, sampled_actions) at the decision point
7: 4. Run the game to completion
8: 5. Pair the recorded states/actions with the final money delta
9: 6. Train the Actor via REINFORCE on the accumulated data
10: """
```
- **Lines 1-10**: The file docstring explains the high-level algorithm. It trains an Actor network using the REINFORCE algorithm (a policy gradient method). The strategy is evaluated once a day (`hour == 0`), extracting state features, sampling actions, and storing them to later compare with the final money delta to apply gradient updates.

```python
12: import os
13: import sys
14: import random
15: import time
16: import math
17: from copy import deepcopy
18: import glob
19: import re
20: import gc
```
- **Lines 12-20**: Standard Python imports. `os`/`sys` for path management, `random` for sampling/epsilon-greedy exploration, `time` for logging durations, `math` for mathematical ops, `deepcopy` for copying state objects, `glob`/`re` for parsing checkpoint filenames, and `gc` for garbage collection to prevent memory leaks during long loops.

```python
22: import numpy as np
23: import torch
24: import torch.nn as nn
25: import torch.optim as optim
```
- **Lines 22-25**: Data science and deep learning imports. `numpy` for arrays. `torch`, `torch.nn`, and `torch.optim` to build the neural networks, define loss functions, and use the Adam optimizer.

```python
27: _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
28: if _REPO_ROOT not in sys.path:
29:     sys.path.insert(0, _REPO_ROOT)
```
- **Lines 27-29**: Dynamically calculates the absolute path to the repository root and adds it to `sys.path`. This ensures absolute imports from the root package (like `from strategy.features...`) work regardless of where the script is executed from.

```python
31: from kaggle_environments import make
32: from strategy.features import extract_features, FEATURE_DIM, _log_money
33: from strategy.actor_net import ActorNetTorch, ActorNetNumpy, torch_to_numpy, save_weights
34: from strategy.value_net import ValueNetTorch
35: from strategy.opponents import OpponentPool
```
- **Line 31**: Imports `make` from Kaggle's environment library to create game instances.
- **Line 32**: Imports the feature extraction functions, feature dimension constant, and a reward shaping function `_log_money`.
- **Line 33**: Imports the PyTorch and NumPy Actor network classes, plus utility functions to convert between them and save weights.
- **Line 34**: Imports the `ValueNetTorch` which acts as the Critic in the training process.
- **Line 35**: Imports `OpponentPool` to sample historical agent checkpoints for self-play.

---

## 2. StrategicTrainingAgent (Lines 42-172)

```python
42: class StrategicTrainingAgent:
43:     """Wraps the heuristic agent with strategic plan selection during
44:     training episodes. Records state features and actions.
45:     """
```
- **Lines 42-45**: Declares the wrapper class used during training. It replaces the logic of a normal agent, collecting transition data while deciding moves.

```python
47:     def __init__(self, actor_net_torch, player_id=0, epsilon=0.0):
48:         self.actor_net = actor_net_torch
49:         self.player_id = player_id
50:         self.epsilon = epsilon
51:         self.recorded_transitions = []  # list of (features, action_dict)
52:         self.current_plan = None
```
- **Line 47-52**: The constructor. It receives the PyTorch model (`actor_net`), assigns the `player_id`, and sets an `epsilon` for random exploration. It initializes an empty list `recorded_transitions` to store experiences for REINFORCE, and a `current_plan` to hold the current daily strategy.

```python
54:         # Import the heuristic agent — we'll call it directly
55:         import main
56:         main._strategic_layer = None
57:         main.step_counter = 2
58:         
59:         self._heuristic = main.agent
60:         self._build_market = main.build_market
61:         self._GameState = main.GameState
```
- **Lines 54-61**: Modifies global state in `main.py` before stealing its logic. It disables the hardcoded `_strategic_layer` in `main.py`, offsets the `step_counter`, and stores references to `main.agent`, `main.build_market`, and `main.GameState` to leverage the heuristic rules for low-level tasks.

```python
63:     def __call__(self, obs, config=None):
64:         """Called by the kaggle_environments runner each turn."""
65:         day = obs.get("day", 0)
66:         hour = obs.get("hour", 0)
```
- **Lines 63-66**: The entry point for Kaggle environments each tick. It extracts the current `day` and `hour` from the observation dictionary.

```python
68:         # At hour 0 of each day, make a strategic decision
69:         if hour == 0:
70:             features = extract_features(obs)
```
- **Lines 68-70**: Only calculates a new strategic plan at the beginning of the day (`hour == 0`), starting by extracting current state features.

```python
72:             # Sample action from the PyTorch model
73:             with torch.no_grad():
74:                 device = next(self.actor_net.parameters()).device if hasattr(self.actor_net, "parameters") else "cpu"
75:                 feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
```
- **Lines 72-75**: Ensures gradients aren't tracked (`torch.no_grad()`). Finds the target hardware device (CPU/GPU) from the `actor_net` weights. Converts the extracted feature array into a PyTorch tensor, adds a batch dimension (`unsqueeze(0)`), and moves it to the appropriate device.

```python
77:                 # Epsilon-greedy (random action) vs sampled action
78:                 if random.random() < self.epsilon:
79:                     # Biased random actions for exploration
80:                     a_land = float(random.random() > 0.8)
81:                     a_hire = random.randint(0, 2)
82:                     a_sell = [float(random.random() < 0.1) for _ in range(9)]
83:                     if random.random() < 0.7:
84:                         a_seed_c = 0
85:                         a_seed_f = random.random() * 0.5
86:                     else:
87:                         a_seed_c = random.randint(0, 4)
88:                         a_seed_f = 0.6 + random.random() * 0.4
89:                     if random.random() < 0.9:
90:                         a_anim_t = 0
91:                         a_anim_f = random.random() * 0.5
92:                     else:
93:                         a_anim_t = random.randint(0, 2)
94:                         a_anim_f = 0.6 + random.random() * 0.4
95:                     a_weed = 5.0
96:                     a_maint = 21.0
97:                     a_panic = 22.0
98:                     a_seed_m = 2.0
99:                     a_land_b = 500.0
```
- **Lines 77-99**: Implements `epsilon`-greedy exploration. If the random check passes, it generates a random action vector rather than using the model. The random generation is heavily biased (e.g., land purchase is 20% likely). These lines assign raw random values mimicking the bounds expected by the neural network format.

```python
101:                     action_dict = {
102:                         "buy_land": torch.tensor([a_land]),
103:                         "hire_target": torch.tensor([a_hire]),
104:                         "sell_hold": torch.tensor([a_sell]),
105:                         "buy_seed_crop": torch.tensor([a_seed_c]),
106:                         "buy_seed_frac": torch.tensor([a_seed_f]),
107:                         "buy_animal_type": torch.tensor([a_anim_t]),
108:                         "buy_animal_frac": torch.tensor([a_anim_f]),
109:                         "weed_penalty": torch.tensor([a_weed]),
110:                         "maint_water_hour": torch.tensor([a_maint]),
111:                         "panic_drop_hour": torch.tensor([a_panic]),
112:                         "seed_threshold_mult": torch.tensor([a_seed_m]),
113:                         "land_unlock_buffer": torch.tensor([a_land_b]),
114:                     }
```
- **Lines 101-114**: Packages the randomly generated parameters into a dictionary of PyTorch tensors to match the format outputted by the model.

```python
115:                 else:
116:                     action_dict, _, _ = self.actor_net.get_action(feat_tensor, deterministic=False)
```
- **Lines 115-116**: If not doing random exploration, passes the `feat_tensor` to `actor_net.get_action()` to sample an action dict stochastically (`deterministic=False`).

```python
118:             # Build plan dict from action_dict
119:             plan = {}
120:             plan["buy_land"] = bool(action_dict["buy_land"].item() > 0.5)
121:             plan["hire_target"] = int(action_dict["hire_target"].item()) + 1
```
- **Lines 118-121**: Converts the network's tensor outputs into a practical `plan` dictionary. Land buying becomes a boolean. The hire target category (0-2) maps to actual employee counts (1-3).

```python
123:             PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
124:                              "EGG", "MILK", "WOOL", "FERTILIZER"]
125:             sell_hold = {}
126:             for i, prod in enumerate(PRODUCT_ORDER):
127:                 if action_dict["sell_hold"][0, i].item() > 0.8:  # Requires strong signal to hold
128:                     sell_hold[prod] = "hold"
129:                 else:
130:                     sell_hold[prod] = "sell"
131:             plan["sell_hold"] = sell_hold
```
- **Lines 123-131**: Iterates over the 9 sellable products. Translates the 9-dimensional `sell_hold` tensor into a dictionary mapping each product name to "hold" (if probability > 0.8) or "sell". 

```python
133:             CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
134:             crop = CROP_ORDER[int(action_dict["buy_seed_crop"].item())]
135:             raw_frac = action_dict["buy_seed_frac"].item()
136:             frac = (raw_frac - 0.6) / 0.4 if raw_frac > 0.6 else 0.0
137:             money = obs["farms"][obs["player"]]["money"]
```
- **Lines 133-137**: Resolves the crop selection from its index. Translates the `raw_frac` (network output 0 to 1) into an actual fraction. Values under 0.6 mean 0% fraction. Values between 0.6 and 1.0 are scaled to be between 0% and 100%. Grabs the player's current money.

```python
139:             # Need to get seed price, import from kaggle_environments
140:             from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS
141:             seed_price = CROPS[crop]["seed"]
142:             qty = int((money * frac) / seed_price) if seed_price > 0 else 0
143:             if qty > 0:
144:                 plan["buy_seed"] = {"crop": crop, "qty": qty}
145:             else:
146:                 plan["buy_seed"] = None
```
- **Lines 139-146**: Dynamically imports game constants to lookup the cost of the chosen seed. Calculates the exact `qty` to buy based on the player's `money` and the chosen `frac`. Appends this to the `plan`.

```python
148:             ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
149:             anim = ANIMAL_ORDER[int(action_dict["buy_animal_type"].item())]
150:             raw_anim_frac = action_dict["buy_animal_frac"].item()
151:             frac_anim = (raw_anim_frac - 0.6) / 0.4 if raw_anim_frac > 0.6 else 0.0
152:             anim_price = ANIMALS[anim]["cost"]
153:             qty_anim = int((money * frac_anim) / anim_price) if anim_price > 0 else 0
154:             if qty_anim > 0:
155:                 plan["buy_animal"] = {"type": anim, "qty": qty_anim}
156:             else:
157:                 plan["buy_animal"] = None
```
- **Lines 148-157**: Repeats the same process used for seeds to determine animal purchases. Translates indices to names, scales fraction boundaries, looks up base cost, computes the `qty_anim`, and stores it in the `plan`.

```python
159:             plan["label"] = "learned_action"
160:             self.current_plan = plan
```
- **Lines 159-160**: Adds a debugging label and stores the generated plan on `self.current_plan`.

```python
162:             # Save for REINFORCE (store the un-batched tensors/arrays)
163:             saved_action = {k: v.squeeze(0).cpu().numpy() for k, v in action_dict.items()}
164:             self.recorded_transitions.append((features, saved_action))
```
- **Lines 162-164**: Iterates through `action_dict` and converts all PyTorch tensors back to raw CPU NumPy arrays, stripping the batch dimension. Saves the original `features` array and `saved_action` array tuple in `self.recorded_transitions` so it can be evaluated at the end of the game.

```python
166:         # Inject the current plan into the heuristic agent's state
167:         import main
168:         main.IS_TRAINING = True
169:         main._current_plan[obs["player"]] = self.current_plan
170:         
171:         action = self._heuristic(obs)
172:         return action
```
- **Lines 166-172**: Binds the newly created plan to the `main.py` global scope for this player, turns on the `IS_TRAINING` flag, and forwards the current observation to the `_heuristic` agent to compute actual turn-by-turn game steps. Finally, returns the lower-level game actions.

---

## 3. Training Loop Setup (Lines 179-240)

```python
179: def train_actor_network(
180:     num_episodes=500,
181:     batch_size=32,
182:     lr=1e-3,
183:     checkpoint_interval=100,
184:     output_dir="strategy_checkpoints",
185:     verbose=True,
186: ):
187:     os.makedirs(output_dir, exist_ok=True)
```
- **Lines 179-187**: Function definition with sensible defaults for episodes, learning rate, and checkpoints. Ensures the checkpoint directory exists.

```python
189:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
190:     if verbose:
191:         print(f"Using device: {device}")
192:     model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
193:     critic = ValueNetTorch(input_dim=FEATURE_DIM).to(device)
```
- **Lines 189-193**: Detects CUDA availability to use the GPU. Initializes both the Actor network (decides actions) and Critic network (estimates state value) and moves them to the device.

```python
195:     start_episode = 0
196:     final_path = os.path.join(output_dir, "actor_net_final.npz")
197:     
198:     checkpoints = glob.glob(os.path.join(output_dir, "actor_net_ep*.npz"))
199:     def extract_ep(p):
200:         match = re.search(r"actor_net_ep(\d+)\.npz", p)
201:         return int(match.group(1)) if match else -1
202:         
203:     if checkpoints:
204:         latest_ckpt = max(checkpoints, key=extract_ep)
205:         start_episode = extract_ep(latest_ckpt)
```
- **Lines 195-205**: Looks for existing `.npz` checkpoint weights in the `output_dir`. Uses a regular expression utility function `extract_ep` to find the checkpoint with the highest episode number, setting it as `start_episode` to resume training smoothly.

```python
207:     loaded_path = None
208:     loaded_critic_path = None
209:     if os.path.exists(final_path):
210:         loaded_path = final_path
211:         critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
212:         if os.path.exists(critic_final_path):
213:             loaded_critic_path = critic_final_path
214:     elif checkpoints:
215:         loaded_path = max(checkpoints, key=extract_ep)
216:         loaded_critic_path = os.path.join(output_dir, f"critic_net_ep{start_episode}.npz")
```
- **Lines 207-216**: Discovers which paths to actually load the network weights from. Favors `final.npz` files if a full run was previously completed; otherwise, falls back to the latest numbered checkpoint. Determines the companion `critic` path to load simultaneously.

```python
218:     if loaded_path:
219:         if verbose:
220:             print(f"Resuming training from weights at {loaded_path} (Starting at Episode {start_episode})")
221:         from strategy.actor_net import load_weights
222:         layers = load_weights(loaded_path)
223:         model.import_numpy_weights(layers)
224:         
225:         if loaded_critic_path and os.path.exists(loaded_critic_path):
226:             from strategy.value_net import load_weights as load_critic_weights
227:             c_layers = load_critic_weights(loaded_critic_path)
228:             critic.import_numpy_weights(c_layers)
```
- **Lines 218-228**: Actually imports the `.npz` NumPy weights and bridges them into the live PyTorch `model` and `critic` using their custom `import_numpy_weights()` methods.

```python
230:     optimizer = optim.Adam(list(model.parameters()) + list(critic.parameters()), lr=lr)
231: 
232:     pool = OpponentPool(
233:         include_starter=True,
234:         include_heuristic=True,
235:         include_passive=True,
236:         noise_variants=2,
237:     )
```
- **Line 230**: Instantiates the Adam optimizer, aggregating the parameters of both networks to train them simultaneously.
- **Lines 232-237**: Instantiates the `OpponentPool` from `opponents.py`, pre-seeding it with rule-based opponents (starter, heuristic, passive).

```python
239:     # Load past checkpoints into the pool
240:     if checkpoints:
241:         for ckpt in checkpoints:
242:             ep_num = extract_ep(ckpt)
243:             if ep_num > 0:
244:                 snapshot_model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
245:                 from strategy.actor_net import load_weights
246:                 snapshot_model.import_numpy_weights(load_weights(ckpt))
247:                 snapshot_agent = StrategicTrainingAgent(
248:                     actor_net_torch=snapshot_model,
249:                     player_id=1,
250:                     epsilon=0.0
251:                 )
252:                 pool.add_snapshot(snapshot_agent, label=f"ep{ep_num}")
253:         if verbose:
254:             print(f"Loaded {len(checkpoints)} past checkpoints into OpponentPool (size: {len(pool)})")
```
- **Lines 239-254**: Supports "self-play" by looping over all historical checkpoints, loading their weights into an isolated PyTorch model, wrapping it in `StrategicTrainingAgent`, and adding it to the `OpponentPool`. This ensures the agent trains against its past selves to avoid catastrophic forgetting and continuously improve.

---

## 4. Replay Buffer Setup & Episode Iteration (Lines 256-309)

```python
256:     data_features = []
257:     data_actions = []
258:     data_targets = []
259:     max_buffer = 50000
260: 
261:     total_games = 0
262:     running_loss = 0.0
263:     loss_count = 0
264: 
265:     epsilon_start = 0.2
266:     epsilon_end = 0.01
267: 
268:     t_start = time.time()
```
- **Lines 256-268**: Initializes variables for the replay buffer (`data_features`, `data_actions`, `data_targets`) with a max cap of 50,000 steps. Tracks statistics like games played and cumulative loss. Sets a decaying `epsilon` strategy (from 20% random action down to 1%).

```python
270:     for episode in range(start_episode, num_episodes):
271:         epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, num_episodes - 1))
272:         seed = random.randint(0, 100000)
```
- **Lines 270-272**: Begins the main training loop. Linearly anneals `epsilon` (probability of a random move) downwards as episodes progress. Generates a random seed for the Kaggle environment to ensure unique maps and item distributions.

```python
274:         model.eval()
275: 
276:         agent0 = StrategicTrainingAgent(
277:             actor_net_torch=model,
278:             player_id=0,
279:             epsilon=epsilon,
280:         )
281: 
282:         opp_agent, opp_label = pool.sample()
```
- **Lines 274-282**: Sets the model to `eval()` mode to suppress training layers like dropout. Wraps the model into `agent0`. Samples a random opponent `opp_agent` from the pool.

```python
284:         try:
285:             env = make("kaggriculture", configuration={"seed": seed})
286:             env.run([agent0, opp_agent])
287:         except KeyboardInterrupt:
288:             if verbose:
289:                 print("\nTraining interrupted by user. Saving final weights...")
290:             break
291:         except Exception as e:
292:             if verbose:
293:                 print(f"  Episode {episode}: ERROR ({e}), skipping")
294:             continue
```
- **Lines 284-294**: Safely instantiates the Kaggle game engine using `make()`. Calls `env.run()` which plays the game to completion. Wraps this inside a `try/except` block to cleanly handle `Ctrl+C` (KeyboardInterrupt) by breaking the loop, and to safely skip over bugs inside the game logic without crashing the entire long-running script.

```python
296:         final_obs = env.state[0]["observation"]
297:         m0 = final_obs["farms"][0]["money"]
298:         m1 = final_obs["farms"][1]["money"]
299:         
300:         # Reward Shaping: Give credit for unsold seeds for both players
301:         from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
302:         priv0 = env.state[0]["observation"].get("private", {})
303:         asset_value0 = sum(qty * CROPS[crop]["seed"] for crop, qty in priv0.get("seeds", {}).items())
304:         
305:         priv1 = env.state[1]["observation"].get("private", {})
306:         asset_value1 = sum(qty * CROPS[crop]["seed"] for crop, qty in priv1.get("seeds", {}).items())
307:         
308:         money_delta = (m0 + asset_value0) - (m1 + asset_value1)
```
- **Lines 296-308**: Game is finished. Extracts final money values for Player 0 (`m0`) and Player 1 (`m1`). Adds "reward shaping" by artificially evaluating their unsold seed inventories and converting that into equivalent cash (`asset_value`). Computes the final target metric `money_delta` — positive if Player 0 won the wealth battle, negative if they lost.

---

## 5. Transition Storage and Model Training (Lines 310-410)

```python
310:         for feat, act in agent0.recorded_transitions:
311:             data_features.append(feat)
312:             data_actions.append(act)
313:             data_targets.append(money_delta)
314: 
315:         if len(data_features) > max_buffer:
316:             data_features = data_features[-max_buffer:]
317:             data_actions = data_actions[-max_buffer:]
318:             data_targets = data_targets[-max_buffer:]
319: 
320:         total_games += 1
```
- **Lines 310-320**: Pulls all daily decisions made by `agent0` during the match. Pushes the features, actions, and the finalized `money_delta` (applied retroactively to *every* step in the match) into the replay buffers. If buffers exceed `max_buffer` (50,000 steps), trims older memories off the beginning.

```python
322:         if len(data_features) >= batch_size:
323:             model.train()
324: 
325:             num_updates = max(1, len(agent0.recorded_transitions) // (batch_size // 4))
326:             for _ in range(num_updates):
327:                 indices = random.sample(range(len(data_features)), batch_size)
```
- **Lines 322-327**: Once enough data exists, sets `model.train()` to enable backprop. Calculates how many mini-batch updates to run (`num_updates`). Randomly selects a `batch_size` set of step indices from the global memory buffer.

```python
329:                 batch_x = torch.tensor(
330:                     np.array([data_features[i] for i in indices]),
331:                     dtype=torch.float32,
332:                 ).to(device)
333:                 
334:                 # Re-compute log probs for the recorded actions
335:                 _, _, logits = model.get_action(batch_x)
```
- **Lines 329-335**: Re-assembles the `data_features` corresponding to those random `indices` into a `batch_x` PyTorch tensor. Passes this tensor into the model to recalculate the raw network output (`logits`).

```python
337:                 # Collect actual actions taken
338:                 a_land = torch.tensor(np.array([data_actions[i]["buy_land"] for i in indices])).to(device)
...
350:                 a_seed_m = torch.tensor(np.array([data_actions[i]["seed_threshold_mult"] for i in indices])).to(device)
351:                 a_land_b = torch.tensor(np.array([data_actions[i]["land_unlock_buffer"] for i in indices])).to(device)
```
- **Lines 337-351**: Re-assembles all 12 action arrays from memory into PyTorch tensors and moves them to the device. These are the *actions that were actually chosen* during the match (some randomly due to epsilon).

```python
352:                 # Compute log probs and entropies
353:                 from torch.distributions import Categorical, Bernoulli, Normal
354:                 dist_land = Bernoulli(torch.sigmoid(logits[..., 0]))
355:                 dist_hire = Categorical(logits=logits[..., 1:7])
...
365:                 dist_land_b = Normal(torch.sigmoid(logits[..., 30]) * 2000.0, 200.0)
```
- **Lines 352-365**: Uses the fresh `logits` to build PyTorch probability distributions for every single output parameter. Booleans use `Bernoulli` distributions. Multi-class integers use `Categorical` distributions. Continuous fractions/values use `Normal` distributions (scaling sigmoids to desired boundaries).

```python
367:                 lp_land = dist_land.log_prob(a_land)
...
378:                 lp_land_b = dist_land_b.log_prob(a_land_b)
379:                 
380:                 total_log_prob = lp_land + lp_hire + ... + lp_land_b
```
- **Lines 367-380**: Asks each Distribution what the *log probability* is of selecting the exact action `a_*` that was recorded in memory. Sums all these log probabilities together into `total_log_prob` — a metric of how likely the current network would be to repeat the historical actions.

```python
382:                 total_entropy = dist_land.entropy() + dist_hire.entropy() + ... + dist_land_b.entropy()
```
- **Lines 382-385**: Calculates `total_entropy` by summing the entropy of every distribution. Entropy encourages the network to remain uncertain/random. This prevents premature convergence on a single sub-optimal strategy.

```python
387:                 batch_r = torch.tensor(
388:                     np.array([_log_money(data_targets[i]) for i in indices]),
389:                     dtype=torch.float32,
390:                 ).to(device)
391:                 
392:                 batch_v = critic(batch_x)
393:                 advantages = batch_r - batch_v.detach()
```
- **Lines 387-393**: Processes the recorded `money_delta` targets through `_log_money` (which squashes massive money values to keep gradients stable), creating `batch_r`. The `critic` network evaluates the features to predict how good the state was (`batch_v`). Subtracting the predicted value from the actual reward gives the `advantage` — a measure of how much *better or worse* the action was than expected.

```python
395:                 if len(advantages) > 1:
396:                     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
397:                 
398:                 actor_loss = -(total_log_prob * advantages).mean()
399:                 critic_loss = nn.functional.mse_loss(batch_v, batch_r)
400:                 entropy_loss = -0.01 * total_entropy.mean()  # 0.01 is the entropy coefficient
401:                 
402:                 loss = actor_loss + 0.5 * critic_loss + entropy_loss
```
- **Lines 395-402**: Normalizes the advantages for stability. Calculates the core REINFORCE policy gradient: `actor_loss = -log_prob * advantage`. It pushes the network to increase probability of actions that resulted in positive advantages. The `critic_loss` trains the value predictor to be more accurate (Mean Squared Error). Total loss combines them all.

```python
404:                 optimizer.zero_grad()
405:                 loss.backward()
406:                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
407:                 torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
408:                 optimizer.step()
409:     
410:                 running_loss += loss.item()
411:                 loss_count += 1
```
- **Lines 404-411**: Zeroes old gradients, backpropagates `loss.backward()` to calculate new gradients, clips exploding gradients to a maximum of 1.0, and steps the optimizer. Updates rolling statistics for logging.

---

## 6. Logging and Checkpointing (Lines 413-482)

```python
413:         if verbose and (episode + 1) % 10 == 0:
414:             avg_loss = running_loss / max(1, loss_count) if loss_count > 0 else 0
415:             elapsed = time.time() - t_start
416:             eps_per_sec = (episode + 1) / elapsed
417:             print(
...
424:                 f"{eps_per_sec:.1f} ep/s"
425:             )
```
- **Lines 413-425**: Every 10 episodes, prints console statistics (loss, win delta, opponent, epsilons, speed).

```python
426:             # Log metrics to JSONL for visualize_training.py
427:             log_dir = os.path.join(output_dir, "logs")
428:             os.makedirs(log_dir, exist_ok=True)
429:             jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
430:             
431:             log_entry = {
432:                 "episode": episode + 1,
...
439:             }
440:             
441:             import json
442:             with open(jsonl_path, "a", encoding="utf-8") as f:
443:                 f.write(json.dumps(log_entry) + "\n")
444:                 
445:             running_loss = 0.0
446:             loss_count = 0
447:             gc.collect()
```
- **Lines 426-447**: Dumps that same metadata into a JSON Lines (`.jsonl`) file on disk so tools like `visualize_training.py` can render graphs later. Resets loss accumulators and forces garbage collection.

```python
449:         if (episode + 1) % checkpoint_interval == 0:
450:             ckpt_path = os.path.join(output_dir, f"actor_net_ep{episode + 1}.npz")
451:             layers = model.export_numpy_weights()
452:             save_weights(layers, ckpt_path)
453:             if verbose:
454:                 print(f"  -> Checkpoint saved: {ckpt_path}")
```
- **Lines 449-454**: Every `checkpoint_interval` (default 100 episodes), serializes the model weights back into NumPy array format (`export_numpy_weights`) and writes them to a new checkpoint `.npz` file.

```python
456:             # Add snapshot to opponent pool
457:             snapshot_model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)
458:             snapshot_model.import_numpy_weights(layers)
459:             snapshot_agent = StrategicTrainingAgent(
...
464:             pool.add_snapshot(snapshot_agent, label=f"ep{episode + 1}")
```
- **Lines 456-464**: Also injects this newly minted checkpoint into the active `OpponentPool` as a snapshot, dynamically widening the pool of opponents the agent can face.

```python
468:     final_path = os.path.join(output_dir, "actor_net_final.npz")
469:     critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
470:     layers = model.export_numpy_weights()
471:     save_weights(layers, final_path)
472:     
473:     from strategy.value_net import save_weights as save_critic_weights
474:     c_layers = critic.export_numpy_weights()
475:     save_critic_weights(c_layers, critic_final_path)
```
- **Lines 468-475**: Once the main `for` loop finishes all `num_episodes` (or if it was KeyboardInterrupted), it performs a final export of both the Actor and Critic networks, saving them to `final.npz` files.

```python
476:     if verbose:
477:         elapsed = time.time() - t_start
478:         print(f"\nTraining complete: {total_games} games in {elapsed:.1f}s")
479:         print(f"Final weights saved to {final_path}")
480: 
481:     return model
```
- **Lines 476-481**: Prints end-of-training summary stats and returns the initialized `model` to whatever function called `train_actor_network()`.
