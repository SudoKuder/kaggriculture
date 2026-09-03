# Line-by-Line Explanation of `features.py`

This document provides a detailed explanation of `c:\codes\kaggriculture\strategy\features.py`. This file is responsible for parsing the complex, nested JSON-like game state dictionary into a flat, fixed-size numerical vector that a Neural Network can understand.

---

## 1. Docstrings and Imports (Lines 1-26)

```python
1: """State feature extraction for the strategic decision layer.
2: 
3: Extracts a fixed-size numeric vector from the observation, capturing own state,
4: opponent visible state, and market conditions.  This vector feeds the value
5: network and the candidate-plan scorer.
6: """
```
- **Lines 1-6**: File docstring. Explains that the script's main job is state translation. The resulting fixed-size array (`float32`) is used as the input layer for the Actor and Value neural networks.

```python
8: import math
9: import numpy as np
10: 
11: # pyrefly: ignore [missing-import]
12: from kaggle_environments.envs.kaggriculture.kaggriculture import (
13:     CROPS, PRODUCTS, ANIMALS, MARKET_PARAMS,
14: )
```
- **Lines 8-14**: Imports `math` for log scaling and `numpy` for building the final array. Imports all the core constants directly from the Kaggle environment engine (`kaggriculture`). This ensures the agent is always in sync with the game's actual definitions of crops and products. The `pyrefly: ignore` comment silences an IDE linter warning because the kaggle module is injected dynamically.

```python
16: # Canonical orderings so feature indices are deterministic.
17: CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
18: PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
19:                   "EGG", "MILK", "WOOL", "FERTILIZER"]
20: ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
21: SHOP_TYPES = [
22:     "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
23:     "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
24: ]
25: 
26: FEATURE_DIM = 64  # 64 state features
```
- **Lines 16-26**: Neural networks require data to always be in the exact same position in the input array. These canonical lists lock the iteration order (e.g., Wheat is always parsed first, Melon last) so that feature #14 always means the exact same thing across different turns. Defines the constant `FEATURE_DIM = 64`.

---

## 2. Utility Functions (Lines 28-80)

```python
28: def _log_money(m):
29:     """Log-scaled money.  Handles negatives gracefully."""
30:     return math.copysign(math.log1p(abs(m)), m) / 10.0
```
- **Lines 28-30**: A utility to compress money values. Because money can grow exponentially (from $10 to $1,000,000), feeding raw money into a neural network breaks gradients. This uses `log1p(abs(m))` to squash it logarithmically, preserves the negative sign with `copysign` (if you go into debt), and divides by 10 to keep it roughly within the `[-1.0, 1.0]` range preferred by Neural Nets.

```python
33: def _count_tiles(tiles):
34:     """Count tile types on a 10x10 board.
...
38:     """
39:     counts = {
40:         "empty": 0, "locked": 0, "weed": 0,
41:     }
```
- **Lines 33-41**: Starts a helper function `_count_tiles` to summarize the 10x10 grid into aggregate numbers. Initializes a `counts` dictionary with base counters.

```python
42:     for crop in CROP_ORDER:
43:         counts[f"plant_{crop}"] = 0
44:         counts[f"yield_{crop}"] = 0
45:     counts["coop_occupied"] = 0
46:     counts["coop_empty"] = 0
...
50:     counts["total_animals"] = 0
```
- **Lines 42-50**: Dynamically adds zeroed-out counters for every specific crop type (both the number of plants, and their accumulated yield). Initializes counters for structures (coops and pastures) and total animals.

```python
52:     for row in tiles:
53:         for tile in row:
54:             if tile is None:
55:                 counts["empty"] += 1
56:             elif tile == "LOCKED":
57:                 counts["locked"] += 1
```
- **Lines 52-57**: Nested loop to iterate through every tile on the grid. If the tile is `None`, it's an empty grass tile. If it's the string `"LOCKED"`, the player hasn't bought that land yet. 

```python
58:             elif isinstance(tile, dict):
59:                 kind = tile.get("kind", "")
60:                 if kind == "WEED":
61:                     counts["weed"] += 1
```
- **Lines 58-61**: If the tile is a dictionary, it contains a game entity. First checks if the entity's `"kind"` is a `"WEED"`.

```python
62:                 elif kind == "PLANT":
63:                     crop = tile.get("crop", "WHEAT")
64:                     counts[f"plant_{crop}"] = counts.get(f"plant_{crop}", 0) + 1
65:                     counts[f"yield_{crop}"] = counts.get(f"yield_{crop}", 0) + tile.get("yield_units", 0)
```
- **Lines 62-65**: If the entity is a `"PLANT"`, figures out what crop it is. Increments that specific crop's plant counter, and adds the plant's current harvestable `yield_units` to the aggregate yield counter.

```python
66:                 elif kind == "COOP":
67:                     if "animal" in tile:
68:                         counts["coop_occupied"] += 1
...
80:     return counts
```
- **Lines 66-80**: Processes `"COOP"` and `"PASTURE"` buildings. Distinguishes between empty and occupied buildings. Tracks overall animal yield. Finally, returns the fully populated `counts` dictionary.

---

## 3. Main Feature Extraction Routine (Lines 83-168)

```python
83: def extract_features(obs):
84:     """Convert a raw observation dict into a numpy float32 feature vector.
...
87:     """
88:     player_id = obs["player"]
89:     opp_id = 1 - player_id
90:     me = obs["farms"][player_id]
91:     opp = obs["farms"][opp_id]
```
- **Lines 83-91**: The main extraction entry point. Finds the active `player_id` (0 or 1), calculates the opponent's ID, and unpacks the respective `"farms"` data into `me` (own farm) and `opp` (opponent's farm).

```python
92:     private = obs.get("private", {})
93:     market = obs.get("market", {})
94:     town = obs.get("town", {})
95: 
96:     day = obs.get("day", 0)
97:     hour = obs.get("hour", 0)
98: 
99:     feats = []
```
- **Lines 92-99**: Unpacks the `private` state (which holds shed inventory and seeds), the global `market`, the `town` unlocks, and the current `day` and `hour`. Initializes an empty list `feats` which will accumulate the numeric features.

### 3.1 Adding Normalized Features

Neural networks perform best when inputs are scaled between `0.0` and `1.0`. The following sections manually normalize every game variable.

```python
101:     # --- Temporal features (3) ---
102:     feats.append(day / 29.0)                    # season progress [0, 1]
103:     feats.append(hour / 23.0)                   # hour within day [0, 1]
104:     feats.append(max(0, 29 - day) / 29.0)       # remaining fraction
```
- **Lines 101-104**: Adds time variables. `day / 29.0` normalizes the day of the season. `hour / 23.0` normalizes the time of day. The last feature is redundant but often helps the network recognize the end-game approaching without having to learn to invert the day signal.

```python
106:     # --- Own money (1) ---
107:     feats.append(_log_money(me.get("money", 0)))
```
- **Lines 106-107**: Appends the log-scaled money calculated by the helper function.

```python
109:     # --- Own tile counts (12) ---
110:     my_tc = _count_tiles(me.get("tiles", []))
111:     feats.append(my_tc["empty"] / 25.0)
112:     feats.append(my_tc["weed"] / 25.0)
113:     for crop in CROP_ORDER:
114:         feats.append(my_tc.get(f"plant_{crop}", 0) / 25.0)
115:     feats.append(my_tc["coop_occupied"] / 6.0)
116:     feats.append(my_tc["pasture_occupied"] / 6.0)
117:     feats.append(my_tc["coop_empty"] / 6.0)
118:     feats.append(my_tc["pasture_empty"] / 6.0)
```
- **Lines 109-118**: Calls `_count_tiles` for the player's farm. Adds the counts of empty tiles, weeds, specific crops, and structures. Divides by `25.0` or `6.0` to roughly normalize the values (based on maximum expected thresholds).

```python
120:     # --- Own crop yields (5) ---
121:     for crop in CROP_ORDER:
122:         max_y = CROPS[crop]["max_yield"]
123:         feats.append(my_tc.get(f"yield_{crop}", 0) / max(1, max_y * 25))
124: 
125:     # --- Own animal yield + count (2) ---
126:     feats.append(my_tc["total_animals"] / 6.0)
127:     feats.append(my_tc["animal_yield"] / 24.0)
```
- **Lines 120-127**: Normalizes crop yields by dividing by the theoretical maximum yield that 25 plants could produce. Normalizes animal totals and yields similarly.

```python
129:     # --- Own shed inventory (9) ---
130:     shed = private.get("shed", {})
131:     for prod in PRODUCT_ORDER:
132:         feats.append(min(shed.get(prod, 0), 100) / 100.0)
133: 
134:     # --- Own seeds (5) ---
135:     seeds = private.get("seeds", {})
136:     for crop in CROP_ORDER:
137:         feats.append(min(seeds.get(crop, 0), 50) / 50.0)
```
- **Lines 129-137**: Iterates through canonical orders for products and seeds, grabbing counts from the `private` state. Caps counts at 100 or 50 using `min()` before dividing, ensuring the neural network never sees a value > 1.0 even if the player hoards massively.

```python
139:     # --- Own infrastructure (2) ---
140:     feats.append(len(me.get("unlocked_quadrants", ["NW"])) / 4.0)
141:     feats.append((1 + len(me.get("hands", []))) / 6.0)   # total workers
```
- **Lines 139-141**: Adds the number of unlocked land quadrants (1-4) normalized to `[0, 1]`. Adds total workforce (1 farmer + up to 5 hands) normalized to `[0, 1]`.

```python
143:     # --- Opponent visible state (5) ---
144:     feats.append(_log_money(opp.get("money", 0)))
145:     opp_tc = _count_tiles(opp.get("tiles", []))
146:     feats.append(opp_tc["total_animals"] / 6.0)
147:     total_opp_plants = sum(opp_tc.get(f"plant_{c}", 0) for c in CROP_ORDER)
148:     feats.append(total_opp_plants / 100.0)
149:     feats.append(len(opp.get("unlocked_quadrants", ["NW"])) / 4.0)
150:     feats.append((1 + len(opp.get("hands", []))) / 6.0)
```
- **Lines 143-150**: The agent can only see limited public information about the opponent (their money, their grid, their unlocked land, their workers). Extracts a smaller summary vector for the opponent to give the network tactical awareness of who is winning.

```python
152:     # --- Market prices (9) ---
153:     prices = market.get("prices", {})
154:     for prod in PRODUCT_ORDER:
155:         base = MARKET_PARAMS.get(prod, {}).get("base", 50)
156:         feats.append(prices.get(prod, base) / max(1, base * 2))
157: 
158:     # --- Shop unlocks (8) ---
159:     shops = town.get("unlocked_shops", [])
160:     for shop_type in SHOP_TYPES:
161:         feats.append(shops.count(shop_type) / 3.0)
```
- **Lines 152-161**: Adds global market data. Normalizes current selling `prices` by dividing by `base * 2`. Looks at the `town` unlocks and adds a feature for how heavily invested the town is in each shop type (max level is 3).

```python
163:     # --- Pad to 64 (original state dimension) ---
164:     while len(feats) < 64:
165:         feats.append(0.0)
166: 
167:     return np.array(feats[:64], dtype=np.float32)
```
- **Lines 163-167**: A safety fallback. If the extraction logic produced fewer than 64 features, it pads the rest with zeros. It converts the `feats` list into a NumPy `float32` array and aggressively slices `[:64]` to ensure it is *exactly* 64 dimensions long, satisfying the shape requirements of the neural network.
