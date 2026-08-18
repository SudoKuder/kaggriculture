# Kaggriculture — Heuristic & Rule Reference

Implemented in `agent.py`, verified against the kaggle-environments game engine.

> This document lists **every rule/heuristic** encoded in `agent.py`.
> Sections 1–13 describe the original improvements; sections 14–24 complete the
> baseline rules (market-order gating, reserves, seed restock, task-routing,
> inventory caps, constants, and the P1–P8 priority queue).

---

## 1. Early Harvest of Capped One-Time Crops
**File:** `agent.py` → `GameState.harvestable_plants()`

The yield cap can be reached **before** `max_yield_day` (e.g., fertilized melon peaks at age 8 while `max_yield_day` is 12). The agent no longer waits the full `max_yield_day`, wasting 2–4 days of tile occupancy per melon — it harvests as soon as `yield_units >= max_yield`, freeing tiles for the next planting cycle.

**Rule:** `harvestable` = `age >= max_yield_day` **OR** `yield_units >= max_yield`.
**Ongoing crops:** harvestable whenever they have accumulated produce (`age >= first_yield_day` and `yield_units > 0`).

---

## 2. Fertilizer for Ongoing Crops (Tomato / Strawberry)
**File:** `agent.py` → `GameState.fertilize_candidates()`

Ongoing crops yield **2 instead of 1** on a production day when they are **both** fertilized *and* watered. Since fertilizer lasts 3 days (`day .. day+2`), a tomato fertilized at age 6 covers production days 6 and 8 (interval=1) — worth +2 units ≈ $120.

**Rule** (fertilize candidate iff):
- `last_yield_day = first_yield_day + interval * (max_yield - 1)`
- `a + 2 >= first_yield_day` **and** `a < last_yield_day` (at least one scheduled production falls in the 3-day window)
- extra guard `a >= max(0, first_yield_day - 2)`

A plant that is **already fertilized** (`fertilized_until_day >= day`) is skipped — re-fertilizing inside the active 3-day window wastes fertilizer.

One-time crops remain candidates inside their watering window: `window_start <= a <= max_yield_day`, where `window_start = (max_yield_day + 1) // 2`.

---

## 3. Watering Only Where It Adds Yield
**File:** `agent.py` → P4 task queue

No longer every unwatered plant gets a water task. P4 only schedules watering that **adds yield**:
- **One-time crops:** inside `[window_start, max_yield_day]` **and** `yield_units < max_yield` (past the window = at cap; watering adds zero).
- **Ongoing crops:** only while currently fertilized (`fertilized_until_day >= day`) — unfertilized watering is survival-only and left to P1.
- P4 also skips plants already `watered_today` or already queued by P1 (via `water_queued`).

Survival watering (plants one missed day from becoming weeds) is still P1 (see §24).

---

## 4. P1/P4 Task Deduplication
**File:** `agent.py` → task-queue construction

P1 (critical survival) and P4 (daily upkeep) both emitted WATER/FEED tasks for the same tiles; because the env only shows the **result** of your previous action, both would get scheduled repeatedly. `water_queued` / `feed_queued` sets ensure each at-risk tile is queued exactly once.

---

## 5. Wheat & Fertilizer Reserves, plus Endgame Liquidation (Day 29)
**File:** `agent.py` → `build_market()`

- **Wheat feed reserve:** `wheat_reserve = animals_n` (1 wheat/day per animal on farm). Only wheat *above* that reserve is sold: `sellable_wheat = max(0, shed_wheat - wheat_reserve)`.
- **Fertilizer reserve:** if any `fertilize_candidates()` exist, keep 3 fertilizer back; only the surplus is sold.
- **Endgame (day ≥ 29):** both reserves drop to 0 — sell **all** wheat and **all** fertilizer. Feeding animals still happens until the very end (1 wheat → ~1 egg ≈ $50 net is still profitable), and no buy-orders are placed (the day-gates in §15/§16 exclude day 29).

---

## 6. Real-Time Wheat Buy Price
**File:** `agent.py` → `build_market()` BUY_PRODUCT WHEAT

Feed-wheat BUY_PRODUCT cost was hardcoded at $25; the market price rises to $45+ under scarcity. The agent now quotes `state.market["prices"]["WHEAT"]` (fallback `max(1, 25)`), and only buys when it can afford the quantity.

The buy is **self-limiting**: it fires only when `shed_wheat < wheat_reserve`, caps at `BUY_WHEAT_TOP = 15` per turn, and is NOT hour-gated — feed must be replenished promptly or animals starve.

---

## 7. More Farm Hands
**File:** `agent.py` → `build_market()` HIRE

Hired hands cost $1–3/day (fib pricing `1,1,2,3,5,…`; a 5th+ hand costs 8) for up to 24 actions/day each — extremely cheap labor.

Rules:
- `hour == 0`, `1 <= day <= 24`, `money > 50`
- Target: 2 hands day 1–2, 3 hands day 3–11, 4 hands day 12–24
- Hire up to `target − have_hands`, each checked against `money >= fib[i]`

---

## 8. Weed-Inclusive Seed Calculation
**File:** `agent.py` → `build_market()` BUY_SEED

`need_seeds = max(0, empty_n + weeds_n − seed_count)` — weeds are counted because P3 (DIG) frees them the same day, and the crops at their overnight stage are a real fast-plant target. Previously the farm would run out of seeds right after clearing weeds.

---

## 9. Later Land Buying Window
**File:** `agent.py` → `build_market()` BUY_LAND

Land purchases extended from day ≤ 20 to day ≤ 24. Day-24 land buys let fast wheat mature by day 29 (`first_yield_day = 2`).

Rules:
- Only while `unlocked < 4`, `empty_n <= 4`, `day <= 24` and `money >= land_cost + 500`.
- `land_cost = {1: 1000, 2: 2000, 3: 4000}`.
- Deliberately **not** hour-gated — the money guard self-limits and earlier unlock = more tile-days.

---

## 10. Hour-Gated Melon Seed Buy (Bugfix)
**File:** `agent.py` → `build_market()` BUY_SEED MELON

Melon seed was previously **not** hour-gated, so the env could buy 1 melon ($80) **every hour** of days 18–19 = up to $3,840 wasted. Now it is gated to `hour == 0` (only inside the melon window `18 <= day <= 19`).

---

## 11. More Animals (4 → 6) + Faster Coop Building
**File:** `agent.py` → `build_market()` + P6

- Goose/MaxAnimals cap raised from 4 to 6 (`MAX_ANIMALS = 6`, counting farm **+** shed).
- Each goose with CARE yields ~2 eggs/day ≈ $100/day.
- Purchase rule: `hour == 0`, `8 <= day <= 18`, `on_farm + in_shed < 6`, `money >= 350`, buy 1 GOOSE (`300`).
- P6 builds **up to 2 coops per turn** when there's a shed backlog, so geese don't idle in the shed for days (details in §19).

---

## 12. Smarter Task Assignment (Holder Preference)
**File:** `agent.py` → `assign_task()`

For tasks that need an inventory item (FEED, FERTILIZE), prefer a free worker **already carrying** the item — avoids a shed round-trip. Otherwise pick the **closest free** worker.

The remainder of the scheduler's routing rules (missing items → shed PICKUP, nearest-shed hover) are in §22.

---

## 13. Extended Planting Until Day 27
**File:** `agent.py` → P7

Wheat/carrot (`first_yield_day = 2`) planted day 27 still harvest by day 29 (season ends day 29). Previously planting stopped day 26.

---

## 14. Sell ASAP — Economy-First, No Price-Waiting
**File:** `agent.py` → `build_market()` SELL block

All stored product sells are issued **immediately** on the turn they are seen — no order waits for a higher future price:
- Economy goods (FERTILIZER, CARROT, TOMATO, EGG, WHEAT, STRAWBERRY, MELON, MILK, WOOL) — this exact order matters: cheaper/stable produce sells first so the money funds buy-orders in the *same* turn.
- Premium goods (STRAWBERRY, MELON, MILK, WOOL) are sold last because they crash to the `$1` floor quickly once a glut forms — and our own bulk sell is the main glut source. Selling ASAP at the still-healthy price beats holding for a recovery our production prevents.
- Reserves applied before selling: wheat & fertilizer above the §5 reserves sell at full count; everything else sells at full shed count.

---

## 15. BUY-Side Hour-Gating Table
**File:** `agent.py` → `build_market()`

Market orders are processed **every hour** (turn). Recurring buys would fire 24×/day and drain cash instantly. Gates:

| Order | Gate | Why |
|---|---|---|
| `BUY_PRODUCT WHEAT` (feed) | **not** gated | Feed must replenish promptly; self-limits on `shed_wheat < wheat_reserve` and `BUY_WHEAT_TOP=15/turn` |
| `SELL` (all) | none | Sell immediately |
| `HIRE` | `hour == 0` | One hire batch per day; fib cost is already small |
| `BUY_LAND` | **not** gated | `money >= cost+500` self-limits; earlier unlock = more tile-days |
| `BUY_SEED` (WHEAT/CARROT/TOMATO) | **not** gated | 3 workers plant up to 72 seeds/day; hour-0-only restock starves & idles tiles. `need_seeds` bounds the order size |
| `BUY_SEED` MELON | `hour == 0` | Melon is $80/unit — buys every hour = $3,840 waste (see §10) |
| `BUY_ANIMAL` | `hour == 0` | At most 1 animal/day |

In total the market list is **capped at `MAX_MARKET_ORDERS = 10`**; extras are silently dropped by the env.

---

## 16. Seed Restock Threshold & Phase-Based Crop Selection
**File:** `agent.py` → `build_market()` BUY_SEED

Restock trigger:
- `seed_burn = hands_n + 1` (each worker plants 1 seed/turn; `hands_n` is today's hired hands)
- `seed_threshold = max(2 * seed_burn, 4)`
- Restock iff `seed_count < seed_threshold` and `need_seeds > 0` and `day <= 26`

Crop selection by day-phase (prices: WHEAT $10/seed, CARROT $20, TOMATO $50, MELON $80; per-turn quantity caps: 20 WHEAT, 10 CARROT, 10 TOMATO, 1 MELON):
- **day 0–7:** WHEAT + CARROT
- **day 8–17:** WHEAT + TOMATO (tomato needs 8 days to first yield; only bought while it can still produce)
- **day 18–19 (hour 0):** MELON only (melon needs 10 days — the last viable window)
- **day 20–26:** WHEAT only (fast crops that still mature)

P7 planting separately limits what is actually *planted* to only crops with enough remaining season (`SEASON − day >= first_yield_day`) — seeds bought are always plantable.

---

## 17. Animal Daily Upkeep Loop (P4) & Harvest
**File:** `agent.py` → P4 + `GameState.harvestable_animals()`

- **HARVEST** (P2): any animal with `yield_units > 0` is harvested — produce does not wait on the tile until `max_held`.
- **FEED** every animal not yet fed (outside the feed_queued dedupe) — animals must eat or escape.
- **CARE** every animal not yet cared today — banks +1 on the next production (cheap +50%/production).
- **COLLECT_FERTILIZER** on every `fertilizer_available` tile — free fertilizer income that otherwise does not accumulate.
- The P1 survival branch covers animals at immediate escape risk (`consecutive_unfed >= 1`).

---

## 18. Coop Building & Animal Placement (P6)
**File:** `agent.py` → P6

When there are animals in the shed (`waiting_animals > 0`):
- **No free coop** → build coops: sort empty tiles by distance to shed center (`(4,4)`), build `min(waiting_animals, 2)` coops per turn (up to 2/turn so a backlog never parks geese in the shed for days).
- **Free coop exists** → `PLACE` the goose, but **only as many as animals waiting** (`free_coops[:waiting_animals]`) — never emits more PLACE tasks than animals ready.

---

## 19. Plant-Tile Reservation for Building (P7)
**File:** `agent.py` → P7

Before planting, P7 removes tiles that are the target of a `BUILD_COOP` / `BUILD_PASTURE` task now in the queue — so construction is never delayed by a freshly planted seed on the same tile.

---

## 20. Inventory Overflow & Shed Drop (P8)
**File:** `agent.py` → P8

A worker whose inventory `total >= 15` units immediately drops it: walk to the nearest shed-adjacent tile and `DROP`. This prevents overflow loss (`shedCapacity = 100`) and keeps workers empty for the next pickup.

---

## 21. Worker Idle Hover Rule
**File:** `agent.py` → idle fallback

Any worker with no task left moves toward the **nearest shed access tile** and waits (`PASS`) when already there. Keeps idle hands near the PICKUP/DROP hub so the next task starts with a short trip.

---

## 22. Task Routers: Shed PICKUP & Item Requirements
**File:** `agent.py` → `assign_task()`

Tasks declare what inventory item they need (`requires`). Assignment logic:
1. If the task requires an item, prefer a **free worker that already carries it** (holder preference, §12).
2. Otherwise pick the closest free worker.
3. If the chosen worker is at the tile and has the item → execute the action.
4. If the worker lacks the item → re-route to the **shed**: when standing on a shed tile do `PICKUP <item> 5`, else move toward it. (The pickup persists in inventory for later turns.)
5. If no item needed → walk to the task tile.

---

## 23. Design Constants & Caps
| Constant | Value | Rule |
|---|---|---|
| `SEASON` | 30 days (0–29) | All day-windows in this doc are 0-indexed |
| `MAX_MARKET_ORDERS` | 10 | Sold or buys beyond this are silently dropped by the env |
| `BUY_WHEAT_TOP` | 15 | Max feed wheat bought per turn |
| `MAX_ANIMALS` | 6 | Geese total, farm + shed combined |
| `wheat_reserve` | #animals (1/day) | Beyond it, wheat is sold |
| `fert_reserve` | 3 (if candidates exist) | Beyond it, fertilizer is sold |
| P8 drop trigger | 15 (inline literal) | Inventory size that triggers a shed drop |
| Board | 10×10 | Shed at center; access tiles `(4,4)…(5,5)` |

---

## 24. Task Queue Priority Architecture (P1–P8)
**File:** `agent.py` → `agent()` task construction

Every turn the tasks are built in this order (each tile queued at most once — dedup via `water_queued`/`feed_queued`):

1. **P1 – Critical survival:** water any plant at risk for weeds (`consecutive_unwatered >= 1` & not watered), feed any animal at risk for escape (`consecutive_unfed >= 1` & not fed).
2. **P2 – Harvest:** `harvestable_plants()` + `harvestable_animals()` (§1 rules).
3. **P3 – TERRAIN:** DIG **all** weeds so the land is reusable.
4. **P4 – Daily upkeep for yield:** §3 (water where it adds value), §17 (feed remainder + CARE + COLLECT_FERTILIZER).
5. **P5 – FERTILIZE:** for every `fertilize_candidates()` when shed has fertilizer (§2).
6. **P6 – Animal logistics:** build coops & place shed animals (§18).
7. **P7 – PLANT:** plant seeds on empty tiles with enough season left (§13, §19); `PASS` stays idle otherwise.
8. **P8 – Inventory overflow:** shed-drop for anyone holding ≥ 15 units (§20).

Then each worker is assigned one of these per priority order; unassigned idle-hover (§21). The **market order list** is always built first (sell then buy, capped at 10) so sell cash funds the buys in the same turn.

---

## Verification

| Test | Result |
| :---- | :---- |
| `python agent.py` (smoke test) | Pass |
| `python test_agent.py` (unit tests) | Pass |
| `python verify_agent.py` | Beats starter by 1.9×, beats pass by 3.2× |
| Multi-seed benchmark (seeds 1–10) | **10/10 wins** vs starter, avg $8,895 vs $3,529 |

## Files Added
- `benchmark.py` — multi-seed win-rate benchmark
- `test_agent.py` — targeted unit tests for the new heuristics