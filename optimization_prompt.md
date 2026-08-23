# Kaggriculture Profit-Maximizing Agent — AI Implementation Prompt

Copy everything below the line into any coding AI to have it apply the changes to `main.py` in this repository.

---

## ROLE

You are an expert competitive-programming agent architect for the Kaggle "Kaggriculture" farming-simulation competition. Your task is to overhaul the existing heuristic agent in `main.py` to **maximize end-of-season coins** (the money in the bank after day 29, hour 23).

## GAME RULES (critical for correctness)

- Season: 30 days (day 0–29), 24 turns per day = 720 turns total.
- Board: 10x10 = four 5x5 quadrants. Only NW starts unlocked; NE/SW/SE cost $1,000 / $2,000 / $4,000.
- Each turn you can perform one action per worker (main `farmer` + hired `hands`) and submit up to **10 market orders** (extras are silently dropped).
- Crops (seed cost, base sale price, days to max yield, max harvestable):
  - Wheat: $10, $25, 4 days, 6 (4 without fertilizer)
  - Carrot: $20, $35, 3 days, 4 (3 without fertilizer)
  - Tomato: $50, $60, 11 days, 4 (ongoing: yields every day, days 8–11)
  - Strawberry: $100, $120, 16 days, 4 (ongoing: every other day, days 10–16)
  - Melon: $80, $250, 12 days (cap reached at day 10, or day 8 with fertilizer), 6
- Animals (buy cost, base product price, production interval, required structure):
  - Goose: $300, Egg $50, 1/day (up to 4 held), coop
  - Cow: $400, Milk $160, every 2 days (up to 6 held), pasture
  - Sheep: $500, Wool $200, every 3 days (up to 6 held), pasture
- Water plants and feed animals every day (or every other). Two consecutive missed water/feed → plant becomes a WEED / animal escapes forever.
- `FERTILIZE` doubles the per-day watering bonus for 3 days. Fertilizer can be **bought** via `BUY_PRODUCT FERTILIZER` (base price $100) and is also produced by animals (1/day each).
- Market prices shift dynamically: staples (wheat, carrot, tomato, egg) absorb oversupply gently; premium goods (melon, strawberry, milk, wool) crash hard to the $1 floor with modest overproduction. Town shops unlock every 3 days and consume demand, which raises prices.
- Only WHEAT and FERTILIZER can be bought back from the market via `BUY_PRODUCT`.
- Hiring: `HIRE` costs Fibonacci sequence $1,1,2,3,5,8... per hand per day. Hands spawn at the shed at hour 0 and disappear at day end.
- Harvested items go to worker inventory, then shed via DROP (or automatically at day end if space). Shed capacity = 100.
- Seeds are stored separately and are consumed directly by `PLANT`.

## CURRENT AGENT KNOWN WEAKNESSES (fix these FIRST)

1. **Market order overflow bugs keep profitable buys from firing.**  
   With 9 SELL orders + HIRE + BUY_SEED + BUY_ANIMAL + BUY_LAND, the agent exceeds `maxMarketOrdersPerTurn = 10` and the env silently drops the tail. Restructuring this is critical. SELL orders should be consolidated to only products actually held, so garbage orders never fill the cap.

2. **Land purchase is effectively inactive.**  
   Current gate: `if unlocked < 4 and empty_n <= 4 and day <= 24 and money >= land_cost + 500` never fires in practice because (a) `empty_n <= 4` is rarely true when money is available, and (b) the `+500` cash buffer is unjustified.  
   **Math**: each quadrant adds 25 tiles. A carrot tile earns roughly $40/day gross when replanted on-cycle. 25 tiles × $20–35/day × ~20 remaining days ≈ $10k–17k from any $1k–4k quadrant. Land is the single highest-ROI investment in the game.

3. **Crop selection is wrong.** The current heuristic compares `profit_per_day = (price*max_yield − seed)/time_to_max` without accounting for:
   - **Cycle recycling**: fast crops can be planted/harvested multiple times per season (this dominates ROI). Wheat cycles ~6 times; carrot ~7; melon only once.
   - **Fertilizer compounding**: fertilized wheat/carrot yield +50–150% more per tile.
   - **Actual current market prices**: the base $35 carrot price can rise sharply when carrot shops unlock.
   - **Melon's $250 price vs $80 seed**: even a single cycle is ~$1,400 gross for 25 tiles.

4. **Other strategic gaps:** never buys melon even in its window, wastes money buying strawberry seeds on day 12+ that never finish, never buys cows/sheep even when profitable, and has no fertilizer-purchase strategy (it only uses animal-produced fertilizer).

5. **Market sell strategy is too naive.** It sells everything immediately regardless of the town's demand schedule. Selling staples during high-demand shop windows gets meaningful price uplift; premium goods should be sold immediately to avoid the crash.

6. **Task allocation wastes actions.** The farmer zigzags across the farm; hands are hired but not deployed efficiently. Plants should be batched around each worker's home zone to minimize walking.

7. **No replant-on-harvest optimization.** When a one-time crop hits cap, it should be harvested and replanted the same/following turn rather than waiting.

## GAM abbreviations for fields used in the code

Use these field names exactly as in `main.py`'s `GameState`:
- `state.tiles[y][x]` — 2D grid; `None`, `"LOCKED"`, or dict
- `t.get("kind")` — `"PLANT"`, `"WEED"`, `"COOP"`, `"PASTURE"`
- `t.get("crop")` — crop name for plants
- `t.get("yield_units")` — harvestable amount on tile
- `t.get("fertilized_until_day")` — last day fertilizer bonus applies; -1 if none
- `t.get("consecutive_unwatered")` / `t.get("consecutive_unfed")`
- `state.private["shed"]`, `state.private["seeds"]`, `state.private["inventories"]`
- `state.market["prices"]`, `state.market["inventory"]`
- `state.town["unlocked_shops"]`

## IMPLEMENT THE FOLLOWING CHANGES IN `main.py`

### 1. Market Order Budget Manager (CRITICAL)

Replace the current flat sell-list followed by buys with an ordered budget-aware builder:

```
def add_order(orders, order, max_total=10):
    if len(orders) < max_total:
        orders.append(order)
    # else drop the LOWEST-priority SELL if this is a buy; sells
    # are always the last priority.
```

Priority order for the market list:
1. `HIRE`
2. `BUY_ANIMAL`
3. `BUY_LAND`
4. `BUY_SEED` (by ROI, below)
5. `BUY_PRODUCT` (WHEAT feed only when below reserve; FERTILIZER up to cap)
6. `SELL` — but prepend in this products order: MELON, STRAWBERRY, WOOL, MILK, EGG, TOMATO, CARROT, WHEAT, FERTILIZER (price-sensitive first). Only add a SELL when the shed+inventory total for that product is > 0. Consolidate each product into ONE SELL order.

### 2. Land Purchase Policy (AGGRESSIVE)

Replace:
```python
if unlocked < 4 and empty_n <= 4 and day <= 24 and money >= land_cost + 500:
```
with a policy that fires early and often:

```python
if unlocked < 4 and day <= 22 and money >= land_cost:
    # Buy a quadrant when the farm is ~70% occupied, OR in the very early
    # days while money allows, OR when we have >2k banked and land is cheap.
    occupied_frac = 1 - (empty_n + weeds_n) / (25 * unlocked)
    if (day <= 3 and money >= land_cost + 200) or \
       (occupied_frac >= 0.7 and money >= land_cost) or \
       (money >= land_cost + 3000):
        orders.append(["BUY_LAND"])
        money -= land_cost
```

Rules:
- Buy NE first (closest shed side), then SW, then SE.
- Keep a cash floor of ~$250 for seeds/wheat/hands; never drain below while animals need feeding.
- Land is still worth buying through day 22; after that only the remaining fast wheat cycles pay back.

### 3. Crop Selection Overhaul — "CYCLE ROI" ORACLE

Replace the current `profit_per_day` comparison with a full-cycle-ROI function:

```python
def crop_roi(crop, day, market_price, fert_available):
    cd = CROPS[crop]
    if cd["ongoing"]:
        time_to_max = cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
    else:
        time_to_max = cd["max_yield_day"]
    cycle_days = time_to_max + 1   # +1 for the tile-occupancy turn after max yield
    if day + cycle_days > 29:
        return -1, 0, cycle_days   # won't finish

    repeats = max(1, (29 - day) // cycle_days)

    if cd["ongoing"]:
        base_yield = cd["max_yield"]          # tomato: 4, strawberry: 4
    else:
        # One-time crops: unfertilized wheat=4, carrot=3, melon=6.
        cap_no_fert = { "WHEAT": 4, "CARROT": 3, "MELON": 6 }
        base_yield = cd["max_yield"] if fert_available else cap_no_fert[crop]

    gross_per_cycle = base_yield * market_price
    net_per_cycle = gross_per_cycle - cd["seed"]
    roi_per_cycle = net_per_cycle / cycle_days
    total_profit = net_per_cycle * repeats
    return roi_per_cycle, repeats, cycle_days
```

- **Choose the crop with the highest `roi_per_cycle` that (1) can still fully ripen before day 29, and (2) you can afford to buy seeds for.**
- Use `state.market["prices"][crop]` (fallback to base).
- **Phase guide (a fallback when ROI is equal):**
  - Day 0–8: CARROT + WHEAT (both cycle fast; carrot has the better margin).
  - Day 8–19: Keep wheat cycling; add tomato if the town has a pizza shop / farmers market and you have the seed money.
  - Day 18–19: buy MELON seeds if any empty tiles and money ≥ $80; sell melons immediately on harvest — the price crashes fast.
  - Day 20–26: WHEAT only (carrot may still be ~equal; use actual prices).
  - Day ≥ 27: no more seed purchases.

- Keep the seed restock threshold (`need_seeds = max(0, empty_n + weeds_n − seed_count)`) but cap buy_qty to `min(need_seeds, money // seed_price, 2*workers+4)` — always buy **at most** room for 2 cycles of the best crop. ALSO consider keeping a small buffer for the current cycle when `day <= 10`.

### 4. Fertilizer Economic Engine

- **Buy FERTILIZER** whenever:
  - `shed["FERTILIZER"] < 3`
  - `money > 150`
  - There is ≥1 `fertilize_candidates()` (from `GameState.fertilize_candidates()`), OR you have WHEAT/CARROT plants inside their water-bonus window younger than `max_yield_day`.
  - Gate to `hour == 0` to avoid 24x restock; buy `min(5, max(0, 5 - shed_fert))` units.
- Add a helper:
  ```
  def fertilizable_now(tile, day):
      if tile.get("fertilized_until_day", -1) >= day: return False
      crop = tile.get("crop", "WHEAT")
      a = day - tile.get("planted_day", day)
      cd = CROPS[crop]
      if cd["ongoing"]:
          last_yield = cd["first_yield_day"] + cd["interval"]*(cd["max_yield"]-1)
          return cd["first_yield_day"] <= a+2 and a < last_yield and a >= cd["first_yield_day"]-2
      else:
          win_start = (cd["max_yield_day"] + 1) // 2
          return win_start <= a <= cd["max_yield_day"] and tile.get("yield_units",0) < cd["max_yield"]
  ```
- In the P5 fertilizer task loop, **skip plants already at max yield** (`yield_units >= cd["max_yield"]`) — fertilizing those wastes fertilizer.

### 5. Sell Strategy — Market-Aware

Implement `demand_bias(product) -> -1, 0, +1`:

```python
def demand_bias(state, product):
    shops = state.town.get("unlocked_shops", [])
    demand = 0
    for s in shops:
        demands = SHOP_DEMANDS[s]        # build a table: bakery→{EGG, WHEAT}, etc.
        if product in demands:
            demand += 2 if len(demands) == 1 else 1
    return demand
```

- **Premium** (MELON, STRAWBERRY, WOOL, MILK): **sell immediately** on harvest — never hold. The crash to $1 is unavoidable once quantity is on the market; a single 1-turn delay can halve the price.
- **Staples** (WHEAT, FERTILIZER): sell normally; keep the wheat reserve (1/animal/day) except on day ≥ 29.
- **Hinge-triggered** (CARROT, EGG, TOMATO): check `demand_bias`. If `>= 2` and the product's market inventory is low (`inv < I0` from `state.market["inventory"]`), the price may rise — hold up to 4–8 hours (1/3 of a day) before selling. If demand is low, sell immediately.
- **Batch splitting**: if you're about to sell MORE than ~15 units of a single staple product in one turn, split across **2 consecutive turns** (~8 units each) to reduce immediate price drag. Only do this if the current price is ≥ base; otherwise don't delay at all.

### 6. Animals & Pasture Economy

Continue geese through day 18, but extend:
- Buy up to **6 geese** (keep `MAX_ANIMALS = 7` total farm+shed cap).
- After day 14, stop buying geese; use the money for land/seeds unless wheat reserve would starve them.

Add livestock diversification **only in these conditions**:
- **Cow:** if `day ≥ 12` AND `money ≥ 500` AND `prices["MILK"] > 140` AND you have ≥2 free empty tiles to build pastures.
- **Sheep:** if `day ≥ 20` and money abundant (prices["WOOL"] > 180) and you have ≥2 empty tiles.
- Build pasture structures proactively when shed has cows/sheep waiting (mirror the existing coop-builder logic but for `kind == "PASTURE"`).

Do NOT buy cows/sheep if you don't have the pasture capacity or the money to also keep feeding wheat to the geese.

### 7. Task Scheduling Improvements (reduce walking)

1. **Zone ownership**: Assign the farmer to the NW quadrant, hand 0 to NE, hand 1 to SW, hand 2+ to SE. During the combined plant/weed/water pass, workers prefer tiles inside their zone (distance gets +2 penalty if outside). This cuts cross-quadrant walking dramatically.
2. **Replant-on-harvest**: when a one-time crop is harvested (`HARVEST`), if the tile is empty afterward and there is an available seed, schedule a `PLANT` for the next free worker (add a `(PLANT, pos, None, [crop])` p3-tier task explicitly in the same turn the harvest happens).
3. **Water batched**: in the P4 water pass, iterate plants in row-major order per zone, so the worker paths are one-way sweeps.
4. **Inventory drop**: keep the P8 `>= 15` trigger, but ONLY drop at the shed if there is shed capacity (`100 − total_shed_items > 0`). If capacity is gone, prefer to keep items in inventory and dump them into shed at the end of day.
5. **Hands always hired**: hire up to:
   - Day 0–8: 2 hands (cost ≤ $4/day)
   - Day ≥ 9: 3 hands if `money > 200` and there are ≥10 plant/weed/action tasks
   - Day ≥ 20: drop to 1 hand (fewer tasks; keep the cash).
   Each hand costs $1–8/day for 24 actions — they are enormously profitable.

### 8. Endgame Precision (Day 27–29)

- Day 27: **stop buying seeds** unless the seed is wheat and it can still ripen (plant + days-to-max ≤ 29). Do NOT fertilize new plants planted after day 26.
- Day 28: sell ALL excess wheat (no reserve) and ALL fertilizer. Begin forcing inventory DROPs (P8 already triggers at day 29 hour ≥ 23-dist; extend this to day 28 hour ≥ 20).
- Day 29 hour 0+:
  - Reserve = 0 for everything; `SELL` every product in shed + ALL worker inventories (sum them).
  - Force every worker to `DROP` at the shed by hour 10, then re-sell the new shed contents.
- Day 29 hour ≥ 23: force DROP if not at shed, move toward shed, DROP, and SELL everything.

### 9. Micro-Bug Fixes in Current Code

Fix these latent bugs in `main.py`:

1. **`GameState.fertilize_candidates()`**: add a cap check — skip a plant if `yield_units >= cd["max_yield"]` because fertilizing adds nothing.
2. **`harvestable_plants()`**: for one-time crops whose `age >= cd["max_yield_day"]`, they are already at-or-past peak; harvesting them returns the tile ~2 days early. Make sure they are harvestable even when `yield_units` has ticked to max (already works), but also for `age ≥ max_yield_day` you can harvest even if the unit count is low (usually the plant decays soon anyway).
3. **`build_market()` sell_candidates**: include inventory sums for `FERTILIZER` and ALL products at EVERY day, not just day ≥ 29 hour ≥ 23. Use `total_in_inv(product)` unconditionally.
4. **`GameState.total_shed_items()`**: enforce `shedCapacity = 100` (either from `state.obs.configuration` if present, or default 100) to avoid selling past capacity thinking more will arrive.
5. **Hard constraints**: rename constant `MAX_ANIMALS` to 7. Extend chicken window from day 18 to day 20.
6. **`assign_tier()`**: When a task `requires` an item but the worker is NOT at the shed, the code routes them to the shed PICKUP each turn — this can starve the actual tile task. Add a check: if the required item is NOT in shed and NOT in any inventory OR the worker is already carrying ≥5 of it, just run the task with what is held (if none, drop that task from queue).
7. **Time fairness**: define a helper `remaining_hours(day, hour) = (29 - day) * 24 + (23 - hour)` and use it anywhere we reason about planting/harvest deadlines.

### 10. Constants to redefine

```python
MAX_MARKET_ORDERS = 10
MAX_ANIMALS = 7
BUY_WHEAT_TOP = 20
FERT_PURCHASE_BUNDLE = 5
SEED_SAFETY = 25
FIRST_BUY_DAY = 1       # buy first land on day 1
EARLY_CASH_BUFFER = 250
```

---

## EXIT CRITERIA (verify before finishing)

1. Refactor the market-order builder so that in a dense mid-game (many sells + buys), the list **never exceeds 10** and always includes: `HIRE` (when affordable), `BUY_LAND` (when ruled), `SEED` and `BUY_PRODUCT` buys, and priority sells.
2. The agent **actually buys land** in early-mid game. Confirm via `python -c "from kaggle_environments import make;env=make('kaggriculture');env.run(['main.py','main.py']);o=env.state[0]['observation'];print(o['farms'][0]['unlocked_quadrants'])"` — the result should include at least 2 quadrants.
3. `python test_agent.py` passes (all existing tests still pass; adjust code not tests).
4. `python benchmark.py` 10-seed average differential vs starter improves ≥ 20% vs the current baseline (~$8,900 avg P0).
5. No exceptions are raised in any of the 10 seeds.

After implementing, iterate by running:
```
python test_agent.py
python verify_agent.py
python benchmark.py
```

Watch for NaN, exceptions, and declining averages between seeds. If performance degrades on a particular seed, inspect the replay (saved via `record_game_log.py`) to see where money leaks (e.g., animals starving, missed watering, unsold inventory).

---

## ADDITIONAL CONTEXT (current `main.py` structure)

- `GameState` (class) has helpers: `plants()`, `animals()`, `empty_structures()`, `weeds()`, `empty_tiles()`, `harvestable_plants()`, `harvestable_animals()`, `fertilize_candidates()`, `worker_inventory()`.
- `build_market(state)` builds `orders` in this order today: SELL list (all products), then BUY_PRODUCT wheat (ungated), then HIRE (hour 0), then BUY_LAND, then BUY_SEED, then BUY_ANIMAL.
- Tasks: `assign_tier(tier_tasks)` matches free workers to tasks by nearest-distance (and holder-preference for item-requiring tasks).
- The P1–P8 system queues: P1 critical water/feed, P2 harvest, P3 build/place, P4 bonus water/CARE/collect-fert, P5 fertilize, P6 (used for building), P8 inventory drop. Then a combined weed-plant pass assigns nearest-tile plant/dig.
- Known env behavior: the market orders list is processed in order, one unit at a time, and sells fund buys within the SAME turn (sells first). Use that to avoid ordering buys you cannot yet afford at the turn's money value.

GOAL: The highest sustainable coins-per-tile through the whole season. Prefer simple robust heuristics over complex fragile logic — the environment is stochastic (weeds, shop draws) and unforgiving (2 missed days = permanent loss).