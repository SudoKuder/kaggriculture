# Kaggriculture Agent — Heuristic Improvements

Implemented in `agent.py`, verified against the kaggle-environments game engine.

## Heuristics Added / Fixed

### 1. Early Harvest of Capped One-Time Crops
**File:** `agent.py` → `GameState.harvestable_plants()`

The yield cap can be reached **before** `max_yield_day` (e.g., fertilized melon peaks at age 8 while `max_yield_day` is 12). Previously the agent waited for `max_yield_day`, wasting 2-4 days of tile occupancy per melon. Now it harvests as soon as `yield_units >= max_yield` — freeing tiles for the next planting cycle.

**Rule:** `harvestable` = `age >= max_yield_day` OR `yield_units >= max_yield`.

### 2. Fertilizer for Ongoing Crops (Tomato / Strawberry)
**File:** `agent.py` → `GameState.fertilize_candidates()`

Previously only one-time crops were fertilizable candidates. But ongoing crops yield **2 instead of 1** on any production day where they're *both* fertilized *and* watered. Since fertilizer lasts 3 days (day..day+2), a tomato fertilized at age 6 covers production days 6 and 8 (interval=1), each producing 2 units instead of 1 — worth +2 units ≈ $120.

The candidate test now checks that at least one scheduled production day falls inside the 3-day fertilizer window:
- `last_yield_day = first_yield_day + interval * (max_yield - 1)`
- Candidate iff `a + 2 >= first_yield_day` and `a < last_yield_day`.

### 3. Watering Only Where It Adds Yield
**File:** `agent.py` → P4 task queue

Previously **every** unwatered plant got a water task. This wastes worker turns on:
- One-time crops already past their bonus window (watering adds zero yield; the plant is at max)
- Unfertilized ongoing crops (base production is 1 whether watered or not)

The P4 list now only schedules:
- One-time crops inside `[window_start, max_yield_day]` while not yet at cap
- Ongoing crops that currently have active fertilizer (watering then doubles the next production)

### 4. P1/P4 Task Deduplication
**File:** `agent.py` → task queue construction

P1 (critical survival) and P4 (daily upkeep) both emitted WATER/FEED tasks for the same tiles. Because the env only shows the *result* of your previous action, both would get scheduled in consecutive observations. The agent now tracks `water_queued` / `feed_queued` sets so each at-risk tile is queued exactly once.

### 5. Endgame Liquidation (Day 29)
**File:** `agent.py` → `build_market()`

On day 29 nothing in the shed can be converted to cash anymore except by selling. The agent now:
- Sets `wheat_reserve = 0` and `fert_reserve = 0` — sells **all** wheat and fertilizer
- Still feeds animals until the very end (feeding one wheat → one egg ~$50 net profit is still positive)

### 6. Real-Time Wheat Buy Price
**File:** `agent.py` → `build_market()`

Feed-wheat BUY_PRODUCT cost was hardcoded at `$25`. The wheat market price rises to $45+ under scarcity. The agent now reads `state.market["prices"]["WHEAT"]` for the live quote.

### 7. More Farm Hands
**File:** `agent.py` → `build_market()`

Hired hands are $1-3/day for up to 24 actions/day each — extremely cheap labor. The agent now hires:
- 2 hands day 1-2
- 3 hands day 3-11
- 4 hands day 12-24 (fib cost 1+1+2+3 = $7/day for 4 hands)

### 8. Weed-Inclusive Seed Calculation
**File:** `agent.py` → `build_market()`

`need_seeds` now counts weeds as near-future plantable tiles. Previously the farm would run out of seeds immediately after P3 (DIG) freed a batch of weed-covered tiles.

### 9. Later Land Buying Window
**File:** `agent.py` → `build_market()`

Land purchases extended from day ≤ 20 to day ≤ 24. Day-24 land buys let fast wheat mature by day 29 (first_yield_day=2).

### 10. Faster Melon Seed Gating Bugfix
**File:** `agent.py` → `build_market()`

Melon seed buy was **not** hour-gated, so the environment could buy 1 melon ($80) **every hour** of days 18-19 = up to $3,840 wasted. Now gated to `hour == 0`.

### 11. More Animals (4 → 6) + Faster Coop Building
**File:** `agent.py` → `build_market()` + P6

- Goose cap raised from 4 to 6. Each goose with CARE yields ~2 eggs/day ≈ $100/day.
- P6 now builds **up to 2 coops per turn** when there's a shed backlog, so geese don't sit idle in the shed for days.

### 12. Smarter Task Assignment (Holder Preference)
**File:** `agent.py` → `assign_task()`

For tasks requiring inventory items (FEED, FERTILIZE), prefer a free worker who already carries the item instead of making someone walk to the shed first. Saves round-trip worker turns.

### 13. Extended Planting Until Day 27
**File:** `agent.py` → P7

Wheat/carrot (first_yield=2) planted day 27 still harvest by day 29. Previously planting stopped day 26.

## Verification

| Test | Result |
| :---- | :---- |
| `python agent.py` (smoke test) | Pass |
| `python test_agent.py` (unit tests) | Pass |
| `python verify_agent.py` | Beats starter by 1.9x, beats pass by 3.2x |
| Multi-seed benchmark (seeds 1-10) | **10/10 wins** vs starter, avg $8,895 vs $3,529 |

## Files Added
- `benchmark.py` — multi-seed win-rate benchmark
- `test_agent.py` — targeted unit tests for the new heuristics