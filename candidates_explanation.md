# Line-by-Line Explanation of `candidates.py`

This document provides a detailed explanation of `c:\codes\kaggriculture\strategy\candidates.py`. This script is likely designed for an alternative or complementary approach to the primary REINFORCE actor network—it generates a discrete list of tactical "plans" (candidates) that a Value Network could evaluate and choose from. 

---

## 1. Docstrings and Setup (Lines 1-17)

```python
1: """Candidate plan generator for the strategic decision layer.
2: 
3: Produces a small set of concrete market-order override plans that the
4: strategic layer evaluates.  Each plan specifies *overrides* to the
5: default heuristic's ``build_market()`` behaviour.
6: 
7: Plans are rule-based (not learned) — we enumerate the genuinely uncertain
8: strategic choices and let the value network pick the best one.
9: """
```
- **Lines 1-9**: Explains the core philosophy of this file. Instead of having a neural network output continuous numbers (like `actor_net.py`), this approach hardcodes specific strategic ideas (e.g., "Buy lots of land", "Hoard premium goods"). The neural network then simply scores which idea is best for the current situation. 

```python
11: # pyrefly: ignore [missing-import]
12: from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, MARKET_PARAMS
13: 
14: 
15: def _can_afford(money, cost, buffer=100):
16:     return money >= cost + buffer
```
- **Lines 11-16**: Imports game constants. Defines a small helper function `_can_afford` to ensure that buying an item doesn't drain the player's bank account entirely, preserving a `buffer` (default $100) for ongoing maintenance costs like water.

---

## 2. Generating Candidates: Initialization (Lines 19-61)

```python
19: def generate_candidates(obs):
20:     """Return a list of candidate plan dicts for the current observation.
...
37:     """
```
- **Lines 19-37**: The main function and its docstring. It explains that each plan is a dictionary. Missing keys in the dictionary imply the agent should just fallback to doing whatever the basic `main.py` heuristic would do.

```python
38:     player_id = obs["player"]
39:     me = obs["farms"][player_id]
40:     day = obs.get("day", 0)
41:     money = me.get("money", 0)
42:     unlocked = len(me.get("unlocked_quadrants", ["NW"]))
43:     shed = obs.get("private", {}).get("shed", {})
44:     seeds = obs.get("private", {}).get("seeds", {})
```
- **Lines 38-44**: Extracts key variables from the complex observation dictionary to make the rule-writing easier. It grabs the player's farm, current day, bank balance, how many quadrants are unlocked, and their private inventory.

```python
46:     # Count empty tiles
47:     empty_n = sum(1 for row in me["tiles"] for t in row if t is None)
48: 
49:     # Land costs
50:     land_cost = {1: 1000, 2: 2000, 3: 4000}.get(unlocked, float("inf"))
```
- **Lines 46-50**: Calculates `empty_n` (the number of `None` tiles) to know how many seeds can actually be planted. Looks up the cost of unlocking the next piece of land based on how many are already unlocked. If all 4 are unlocked, the cost becomes infinity.

```python
52:     # Animals on farm + shed
53:     animals_farm = 0
54:     for row in me["tiles"]:
55:         for t in row:
56:             if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and "animal" in t:
57:                 animals_farm += 1
58:     animals_shed = sum(shed.get(a, 0) for a in ["GOOSE", "COW", "SHEEP"])
59:     total_animals = animals_farm + animals_shed
60: 
61:     candidates = []
```
- **Lines 52-61**: Iterates through the farm to count active animals, and checks the shed for unplaced animals. Sums them into `total_animals` to throttle over-buying. Initializes the empty `candidates` list.

---

## 3. The Rules: Candidate Plans (Lines 63-161)

```python
63:     # --- Plan 0: BASELINE (heuristic as-is) ---
64:     candidates.append({
65:         "label": "baseline",
66:     })
```
- **Lines 63-66**: Always offers the baseline plan. Because it has no override keys, the heuristic will run exactly as programmed.

```python
68:     # --- Plan 1: AGGRESSIVE LAND ---
69:     if unlocked < 4 and _can_afford(money, land_cost, 200) and day <= 24:
70:         if unlocked < 3 or _can_afford(money, land_cost, 3000):
71:             candidates.append({
72:                 "buy_land": True,
73:                 "label": "aggressive_land",
74:             })
```
- **Lines 68-74**: Proposes buying land. It only does this if the player can afford it, and if it's early enough in the season (`day <= 24`) for the land to pay itself off. For the final, most expensive plot (`unlocked == 3`), it demands a much higher safety buffer ($3000) before suggesting it.

```python
76:     # --- Plan 2: HOLD PREMIUM GOODS ---
77:     # Only makes sense if we have premium goods in shed and it's not endgame
78:     premium = ["MELON", "STRAWBERRY", "MILK", "WOOL"]
79:     has_premium = any(shed.get(p, 0) > 0 for p in premium)
80:     if has_premium and day < 27:
81:         candidates.append({
82:             "sell_hold": {p: "hold" for p in premium},
83:             "label": "hold_premium",
84:         })
```
- **Lines 76-84**: Proposes hoarding valuable items. If the player owns any "premium" goods and it's not the very end of the game, it suggests overriding the heuristic's sell logic to `"hold"`, waiting for a better market price.

```python
86:     # --- Plan 3: SELL EVERYTHING NOW ---
87:     if has_premium and day < 28:
88:         candidates.append({
89:             "sell_hold": {p: "sell" for p in premium},
90:             "label": "sell_all_now",
91:         })
```
- **Lines 86-91**: The exact opposite of Plan 2. It forces the immediate sale of all premium goods. The neural network will compare Plan 2 and Plan 3 and decide if the current market price is high enough to justify selling now versus waiting.

```python
93:     # --- Plan 4: ANIMAL RUSH (buy geese aggressively) ---
94:     if total_animals < 30 and day <= 22 and _can_afford(money, 300, 200):
95:         n_buy = min(4, 30 - total_animals)
96:         if _can_afford(money, 300 * n_buy, 200):
97:             candidates.append({
98:                 "buy_animal": {"type": "GOOSE", "qty": n_buy},
99:                 "label": f"animal_rush_{n_buy}",
100:             })
```
- **Lines 93-100**: Proposes buying up to 4 Geese simultaneously, provided the player has the cash and hasn't hit the arbitrary 30 animal limit. Only proposed before day 22.

```python
102:     # --- Plan 5: MELON PIVOT ---
103:     # In the melon window, consider switching all seeds to melon
104:     if 14 <= day <= 19 and _can_afford(money, 80, 200):
105:         melon_qty = min(empty_n, int((money - 200) // 80))
106:         if melon_qty > 0:
107:             candidates.append({
108:                 "buy_seed": {"crop": "MELON", "qty": melon_qty},
109:                 "label": f"melon_pivot_{melon_qty}",
110:             })
```
- **Lines 102-110**: Melons take a long time to grow but pay off massively. This rule creates a specific "Melon Window" (days 14-19) where it suggests buying exactly enough melon seeds to fill every empty tile on the farm.

```python
112:     # --- Plan 6: CONSERVATIVE (skip all buys) ---
113:     if day >= 5:  # only meaningful after opening
114:         candidates.append({
115:             "buy_land": False,
116:             "buy_seed": None,
117:             "buy_animal": None,
118:             "hire_target": 1,  # farmer only
119:             "label": "conservative",
120:         })
```
- **Lines 112-120**: Suggests a "turtle" strategy. Stops buying land, fires all extra hands (reducing workforce to 1), and rides out the rest of the game minimizing expenses.

```python
122:     # --- Plan 7: HIRE HEAVY ---
123:     if day <= 20 and _can_afford(money, 15, 100):
124:         candidates.append({
125:             "hire_target": 14,
126:             "label": "hire_heavy",
127:         })
```
- **Lines 122-127**: A massive hiring spike, likely used to rapidly clear a heavily weeded field or harvest a massive crop yield. (Note: 14 is higher than the max hands allowed, suggesting the underlying heuristic clamps it to the max).

```python
129:     # --- Plan 8: COW INVESTMENT (if day allows ROI) ---
130:     if day <= 18 and total_animals < 30 and _can_afford(money, 400, 300):
131:         candidates.append({
132:             "buy_animal": {"type": "COW", "qty": 1},
133:             "label": "cow_invest",
134:         })
```
- **Lines 129-134**: Suggests buying a Cow (which requires an earlier Return-On-Investment horizon than Geese, hence `day <= 18`).

```python
136:     # --- Plan 9: WHEAT ECONOMY (fast cycling) ---
137:     seed_count = sum(seeds.values())
138:     if day <= 25 and empty_n > 5 and seed_count < 5:
139:         wheat_qty = min(empty_n, int((money - 100) // 10))
140:         if wheat_qty > 0:
141:             candidates.append({
142:                 "buy_seed": {"crop": "WHEAT", "qty": wheat_qty},
143:                 "label": f"wheat_economy_{wheat_qty}",
144:             })
```
- **Lines 136-144**: If the farm has lots of empty space (`> 5`), low seed inventory, and it's late in the season, it suggests buying cheap Wheat seeds to rapidly fill the field and get a fast payout before the season ends.

```python
146:     # --- Plan 10: SHEEP INVESTMENT (if day allows ROI) ---
147:     if day <= 16 and total_animals < 30 and _can_afford(money, 600, 300):
148:         candidates.append({
149:             "buy_animal": {"type": "SHEEP", "qty": 1},
150:             "label": "sheep_invest",
151:         })
```
- **Lines 146-151**: Suggests buying a Sheep, which has the most restrictive timeline (`day <= 16`) and highest cost ($600).

```python
153:     # Always have at least 2 candidates (baseline + one alternative)
154:     if len(candidates) < 2:
155:         candidates.append({
156:             "sell_hold": {},
157:             "label": "noop_alternative",
158:         })
159: 
160:     return candidates
161: 
```
- **Lines 153-161**: A safety catch. Deep learning models or evaluation loops can crash if given only a single option to evaluate. If none of the conditional rules above fired (meaning `candidates` only contains `baseline`), it injects a dummy `"noop_alternative"` to ensure there are always at least two options to choose between. Finally, returns the list.
