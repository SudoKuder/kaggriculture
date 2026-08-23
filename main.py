# submission.py
# pyrefly: ignore [missing-import]
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, PRODUCTS, ANIMALS, MARKET_PARAMS

# Global counter to track steps (for logging)
step_counter = 0
DEBUG = False  # Set to True to enable debug output

# Board geometry (default 10x10). The shed sits at the center; its four
# orthogonally-adjacent access tiles are (4,4) (5,4) (4,5) (5,5).
HALF = 5
SHED_TILES = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]
MAX_MARKET_ORDERS = 10
SEASON = 30  # days 0..29 (0-indexed)

BUY_WHEAT_TOP = 15      # max feed-wheat units bought per turn
MAX_ANIMALS = 6         # max animals on farm + shed combined


def log(*args):
    """Debug output controlled by DEBUG."""
    if DEBUG:
        print(f"[Agent Log]", *args)


class GameState:
    """Parses and queries the observation using the real game schema.

    Key schema notes (see kaggriculture.py):
      - plant tiles use 'crop' (not 'type')
      - 'animal' on a structure is a STRING (e.g. "GOOSE"); empty structures
        (no 'animal' key) hold no produce and need no care/feed
      - fertilizer state is 'fertilized_until_day' (int, -1 when inactive)
      - 'yield_units' accumulates on the tile; harvested into worker inventory
      - day is 0-indexed (0..29)
    """
    def __init__(self, obs):
        self.obs = obs
        self.day = obs.get("day", 0)
        self.player_id = obs["player"]
        self.me = obs["farms"][self.player_id]
        self.private = obs["private"]
        self.tiles = self.me["tiles"]
        self.market = obs.get("market", {})
        self.money = self.me["money"]
        self.shed = self.private.get("shed", {})
        self.seeds = self.private.get("seeds", {})
        self.inventories = self.private.get("inventories", [{}])

    # ---------------- tile helpers ----------------
    def _iter_dict_tiles(self):
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    yield x, y, tile

    def plants(self):
        return [(x, y, t) for x, y, t in self._iter_dict_tiles() if t.get("kind") == "PLANT"]

    def animals(self):
        return [(x, y, t) for x, y, t in self._iter_dict_tiles()
                if t.get("kind") in ("COOP", "PASTURE") and "animal" in t]

    def empty_structures(self):
        return [(x, y, t) for x, y, t in self._iter_dict_tiles()
                if t.get("kind") in ("COOP", "PASTURE") and "animal" not in t]

    def weeds(self):
        return [(x, y) for x, y, t in self._iter_dict_tiles() if t.get("kind") == "WEED"]

    def empty_tiles(self):
        return [(x, y) for y, row in enumerate(self.tiles) for x, tile in enumerate(row)
                if tile is None]

    def age(self, tile):
        return self.day - tile.get("planted_day", self.day)

    # ---------------- derived state ----------------
    def harvestable_plants(self):
        out = []
        for x, y, t in self.plants():
            if self.day >= 29 and t.get("yield_units", 0) > 0:
                out.append((x, y))
                continue
            crop = t.get("crop", "WHEAT")
            cd = CROPS[crop]
            a = self.age(t)
            if cd["ongoing"]:
                # Ongoing crops produce on scheduled days; harvest whenever
                # there is accumulated produce.
                if a >= cd["first_yield_day"] and t.get("yield_units", 0) > 0:
                    out.append((x, y))
            else:
                # One-time crops: harvest once yield has peaked. The cap can be
                # reached before max_yield_day (e.g. fertilized melon at age 8
                # vs max_yield_day 12) — freeing the tile early is pure win.
                if a >= cd["max_yield_day"] or t.get("yield_units", 0) >= cd["max_yield"]:
                    out.append((x, y))
        return out

    def harvestable_animals(self):
        return [(x, y) for x, y, t in self.animals() if t.get("yield_units", 0) > 0]

    def fertilize_candidates(self):
        """Crops where fertilizer still adds yield and none is active.

        One-time crops: while inside the watering bonus window.
        Ongoing crops: while at least one scheduled production day remains
        inside the 3-day fertilizer window (fertilized+watered productions
        yield 2 instead of 1).
        """
        out = []
        for x, y, t in self.plants():
            crop = t.get("crop", "WHEAT")
            cd = CROPS[crop]
            if t.get("fertilized_until_day", -1) >= self.day:
                continue
            a = self.age(t)
            if cd["ongoing"]:
                # Last scheduled production day for ongoing crops.
                last_yield_day = cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
                # Fertilizing at age a covers productions on days a..a+2; only
                # useful if at least one production day falls inside.
                if a + 2 >= cd["first_yield_day"] and a < last_yield_day:
                    if a >= max(0, cd["first_yield_day"] - 2):
                        out.append((x, y))
            else:
                window_start = (cd["max_yield_day"] + 1) // 2
                if window_start <= a <= cd["max_yield_day"]:
                    out.append((x, y))
        return out

    def animals_on_farm(self):
        return len(self.animals())

    def animals_in_shed(self):
        return sum(self.shed.get(a, 0) for a in ANIMALS)

    def total_shed_items(self):
        return sum(v for k, v in self.shed.items() if k in PRODUCTS)

    def worker_inventory(self, worker_idx):
        if 0 <= worker_idx < len(self.inventories):
            d = self.inventories[worker_idx]
            return d if isinstance(d, dict) else {}
        return {}


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def move_towards(current_pos, target_pos):
    """Returns the next movement action to reach target_pos."""
    cx, cy = current_pos
    tx, ty = target_pos
    if cx < tx:
        return "EAST"
    if cx > tx:
        return "WEST"
    if cy < ty:
        return "SOUTH"
    if cy > ty:
        return "NORTH"
    return None


def closest_shed_tile(pos):
    return min(SHED_TILES, key=lambda s: manhattan_distance(pos, s))


def get_closest_worker(target_pos, workers_dict):
    """Returns the id of the closest available worker to target_pos."""
    best, best_d = None, float("inf")
    for wid, pos in workers_dict.items():
        d = manhattan_distance(pos, target_pos)
        if d < best_d:
            best, best_d = wid, d
    return best


def build_market(state):
    """Construct market orders; returns up to MAX_MARKET_ORDERS orders."""
    day = state.day
    shed = state.shed
    money = float(state.money)

    unlocked = len(state.me["unlocked_quadrants"])
    empty_n = len(state.empty_tiles())
    weeds_n = len(state.weeds())
    seed_count = sum(state.seeds.values())

    # On the final day, liquidate everything — produce left in the shed after
    # the season ends earns nothing. Days 28 still needs the feed/fertilizer
    # reserves: feeding 1 wheat → ~1 egg (still profitable), and fertilizer
    # still adds yield to plants in their bonus window.
    endgame = day >= 29

    # Reserve wheat for animal feed (1 wheat/day per animal), except in the
    # endgame when keeping wheat instead of selling it earns nothing.
    animals_n = state.animals_on_farm()
    wheat_reserve = 0 if endgame else animals_n
    shed_wheat = shed.get("WHEAT", 0)
    sellable_wheat = max(0, shed_wheat - wheat_reserve)

    # Reserve a little fertilizer for crops that are ready for it.
    fert_candidates = len(state.fertilize_candidates())
    fert_reserve = 0 if endgame else (3 if fert_candidates > 0 else 0)

    orders = []

    # ---- SELL (economy products first so we have money for buys this turn) ----
    worker_invs = state.private.get("inventories", [])
    hour = state.obs.get("hour", 0)
    
    def total_in_inv(product):
        if day >= 29 and hour >= 23:
            return sum(inv.get(product, 0) for inv in worker_invs if isinstance(inv, dict))
        return 0

    sell_candidates = [
        ("FERTILIZER", max(0, shed.get("FERTILIZER", 0) - fert_reserve) + total_in_inv("FERTILIZER")),
        ("CARROT", shed.get("CARROT", 0) + total_in_inv("CARROT")),
        ("TOMATO", shed.get("TOMATO", 0) + total_in_inv("TOMATO")),
        ("EGG", shed.get("EGG", 0) + total_in_inv("EGG")),
        ("WHEAT", sellable_wheat + total_in_inv("WHEAT")),
        ("STRAWBERRY", shed.get("STRAWBERRY", 0) + total_in_inv("STRAWBERRY")),
        ("MELON", shed.get("MELON", 0) + total_in_inv("MELON")),
        ("MILK", shed.get("MILK", 0) + total_in_inv("MILK")),
        ("WOOL", shed.get("WOOL", 0) + total_in_inv("WOOL")),
    ]
    for product, qty in sell_candidates:
        if qty <= 0:
            continue
        # Sell immediately. Premium goods (MELON/STRAWBERRY/MILK/WOOL) crash to
        # the $1 floor quickly once a glut forms, and our own bulk sale is the
        # main glut source — selling ASAP at the still-healthy price beats
        # holding and waiting for a recovery that our production prevents.
        orders.append(["SELL", product, qty])

    # ---- BUY-side orders (hires, land, seeds, animals) are gated to hour==0
    # (the day's first turn). The market processes orders every turn, so without
    # this gate we'd buy 24x/day and drain cash instantly. BUY_PRODUCT wheat is
    # exempt (handled above) because feed wheat must be replenished promptly.

    # ---- BUY_PRODUCT wheat for animal feed (animals must eat) ----
    # NOT gated to hour 0: shedding wheat below the reserve must be replenished
    # promptly or animals starve. Self-limiting: only fires when wheat is below
    # the daily feed reserve, and caps at BUY_WHEAT_TOP per turn.
    if wheat_reserve > 0 and shed_wheat < wheat_reserve:
        need = min(wheat_reserve - shed_wheat, BUY_WHEAT_TOP)
        # Quote against the real observed price, not the $25 base, so a scarce
        # market (wheat at $40+) doesn't blow the budget.
        wheat_price = max(1, int(state.market.get("prices", {}).get("WHEAT", 25)))
        cost = wheat_price * need
        if money >= cost:
            orders.append(["BUY_PRODUCT", "WHEAT", need])
            money -= cost

    # ---- HIRE farm hands (Dynamic calculation) ----
    # No hard day cap — hiring stays available as long as the workload demands
    # it. Late-season animals still need daily feed/care; starving them because
    # we stopped hiring is worse than the hire cost.
    if hour == 0 and day >= 1 and money > 50:
        tasks_today = 0
        tasks_today += len(state.plants())          # watering
        tasks_today += state.animals_on_farm() * 3  # feed + care + harvest
        tasks_today += len(state.harvestable_plants())
        tasks_today += len(state.harvestable_animals())
        tasks_today += empty_n                      # planting
        tasks_today += weeds_n * 2                  # dig + plant
        tasks_today += len(state.fertilize_candidates())
        # Animals waiting in the shed need BUILD + PLACE (2 turns each).
        tasks_today += state.animals_in_shed() * 2

        # Estimate 2 turns per task (1 for action, 1 for movement)
        turns_needed = tasks_today * 2
        workers_needed = (turns_needed + 23) // 24

        have_hands = len(state.me["hands"])
        target = workers_needed
        need_hires = max(0, (target - 1) - have_hands)

        for i in range(need_hires):
            hire_idx = have_hands + i
            cost = [1, 1, 2, 3, 5][hire_idx] if hire_idx < 5 else 8
            if money >= cost:
                orders.append(["HIRE"])
                money -= cost

    # ---- BUY_LAND ----
    # Deliberately NOT gated to hour 0: `money >= land_cost + 500` self-limits
    # the spend, and unlocking quadrants earlier gives far more planting room
    # (5x5=25 tiles each). Buying the next quadrant as soon as it's affordable
    # massively outweighs the small risk of overspending once.
    land_cost = {1: 1000, 2: 2000, 3: 4000}.get(unlocked, float("inf"))
    # Late-season land is still productive with fast wheat (buy day 24, plant
    # immediately, harvest day 29), so keep buying through day 24.
    if unlocked < 4 and empty_n <= 4 and day <= 24 and money >= land_cost + 500:
        orders.append(["BUY_LAND"])
        money -= land_cost

    # ---- BUY_SEED (Dynamic ROI) ----
    need_seeds = max(0, empty_n + weeds_n - seed_count)
    hands_n = len(state.me["hands"])
    seed_burn = hands_n + 1
    seed_threshold = max(2 * seed_burn, 4)
    if seed_count < seed_threshold and need_seeds > 0 and day <= 26:
        best_crop = None
        best_profit = -float('inf')
        
        for crop in CROPS:
            cd = CROPS[crop]
            if cd["ongoing"]:
                time_to_max = cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
            else:
                time_to_max = cd["max_yield_day"]
                
            # Check if it has time to reach max yield before day 29, with 1 day buffer for planting
            if day + time_to_max + 1 <= 29:
                market_price = max(1, int(state.market.get("prices", {}).get(crop, 25)))
                profit_per_day = ((market_price * cd["max_yield"]) - cd["seed"]) / time_to_max
                
                if profit_per_day > best_profit:
                    best_profit = profit_per_day
                    best_crop = crop
                    
        if best_crop:
            seed_price = CROPS[best_crop]["seed"]
            buy_qty = min(need_seeds, int(money // seed_price))
            if buy_qty > 0:
                # Keep a small buffer early game for hiring/wheat
                max_spend = money - 50 if day < 10 else money
                buy_qty = min(buy_qty, int(max_spend // seed_price))
                if buy_qty > 0:
                    orders.append(["BUY_SEED", best_crop, buy_qty])
                    money -= buy_qty * seed_price

    # ---- BUY_ANIMAL (max 6 total; 1/day while in the mid-game window) ----
    # Geese produce daily once placed, and CARE doubles output. A goose bought
    # day 8 clears its $300 cost by ~day 18; six is a good balance between
    # income and the wheat feeding burden.
    if hour == 0 and 8 <= day <= 18:
        on_farm = state.animals_on_farm()
        in_shed = state.animals_in_shed()
        if on_farm + in_shed < MAX_ANIMALS and money >= 350:
            orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300

    # The env silently drops orders past maxMarketOrdersPerTurn.
    return orders[:MAX_MARKET_ORDERS]


def agent(obs):
    global step_counter
    step_counter += 1
    state = GameState(obs)
    log(f"--- Step {step_counter} | Day {state.day} | Money {state.money:.0f} ---")

    market = build_market(state)

    hands_n = len(state.me["hands"])
    unassigned = {"farmer": list(state.me["farmer"])}
    for i, pos in enumerate(state.me["hands"]):
        unassigned[f"hand_{i}"] = list(pos)

    actions = {"farmer": None, "hands": [None] * hands_n}

    def worker_index(wid):
        return 0 if wid == "farmer" else int(wid.split("_")[1]) + 1

    def inventory_of(wid):
        return state.worker_inventory(worker_index(wid))

    def assign(wid, action):
        if wid == "farmer":
            actions["farmer"] = action
        else:
            idx = int(wid.split("_")[1])
            actions["hands"][idx] = action
        unassigned.pop(wid, None)

    def assign_tier(tier_tasks):
        while tier_tasks and unassigned:
            best_pair = None
            best_cost = float('inf')
            
            for task_idx, task in enumerate(tier_tasks):
                typ, pos, requires, args = task
                
                if requires:
                    holders = [w for w in unassigned if inventory_of(w).get(requires, 0) > 0]
                    candidates = holders if holders else list(unassigned.keys())
                else:
                    candidates = list(unassigned.keys())
                    
                for wid in candidates:
                    wpos = unassigned[wid]
                    
                    has_req = requires is None or inventory_of(wid).get(requires, 0) > 0
                    if requires and not has_req:
                        shed_pos = closest_shed_tile(wpos)
                        cost = manhattan_distance(wpos, shed_pos) + manhattan_distance(shed_pos, pos)
                    else:
                        cost = manhattan_distance(wpos, pos)
                        
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (wid, task_idx)
                        
            if best_pair:
                wid, task_idx = best_pair
                task = tier_tasks.pop(task_idx)
                
                typ, pos, requires, args = task
                wpos = unassigned[wid]
                has_req = requires is None or inventory_of(wid).get(requires, 0) > 0
                
                if list(wpos) == list(pos) and has_req:
                    assign(wid, [typ] + (args or []))
                elif requires and not has_req:
                    shed_pos = closest_shed_tile(wpos)
                    if list(wpos) == list(shed_pos):
                        assign(wid, ["PICKUP", requires, 5])
                    else:
                        m = move_towards(wpos, shed_pos)
                        assign(wid, [m] if m else ["PASS"])
                else:
                    m = move_towards(wpos, pos)
                    assign(wid, [m] if m else ["PASS"])
            else:
                break

    # Build the task queues by priority tiers.
    # Animal care gets a dedicated pre-pass so it is NEVER starved out by
    # crop-task backlog, no matter how many crop tiles the farm has.
    p1_tasks = []       # emergency: dying plants/animals (consecutive miss)
    animal_tasks = []   # dedicated: feed / care / place / build structure
    p2_tasks = []       # harvest (plants + animals)
    p4_tasks = []       # crop watering (bonus window — adds yield)
    p4b_tasks = []      # maintenance watering (prevents death, no yield gain)
    p5_tasks = []       # fertilize

    water_queued = set()
    feed_queued = set()

    # --- P1: emergencies (about to die) ---
    for x, y, t in state.plants():
        if not t.get("watered_today", False) and t.get("consecutive_unwatered", 0) >= 1:
            p1_tasks.append(("WATER", [x, y], None, []))
            water_queued.add((x, y))
    for x, y, t in state.animals():
        if not t.get("fed_today", False) and t.get("consecutive_unfed", 0) >= 1:
            p1_tasks.append(("FEED", [x, y], "WHEAT", []))
            feed_queued.add((x, y))

    # --- P2: harvest ---
    for x, y in state.harvestable_plants():
        p2_tasks.append(("HARVEST", [x, y], None, []))
    for x, y in state.harvestable_animals():
        p2_tasks.append(("HARVEST", [x, y], None, []))

    # --- Animal housing: build enough structures for the FULL deficit ---
    waiting_animals = state.animals_in_shed()
    build_tiles_used = set()   # tiles reserved for BUILD so planting skips them
    if waiting_animals > 0:
        # Determine structure type per animal species in shed.
        # GOOSE -> COOP, everything else -> PASTURE
        goose_waiting = state.shed.get("GOOSE", 0)
        pasture_waiting = waiting_animals - goose_waiting

        free_coops = [(x, y, t) for x, y, t in state.empty_structures()
                      if t.get("kind") == "COOP"]
        free_pastures = [(x, y, t) for x, y, t in state.empty_structures()
                         if t.get("kind") == "PASTURE"]

        # Place animals on existing free structures
        for (x, y, _t) in free_coops[:goose_waiting]:
            animal_tasks.append(("PLACE", [x, y], "GOOSE", ["GOOSE"]))
        for animal_name in [a for a in ANIMALS if a != "GOOSE"]:
            count = state.shed.get(animal_name, 0)
            if count > 0:
                for (x, y, _t) in free_pastures[:count]:
                    animal_tasks.append(("PLACE", [x, y], animal_name, [animal_name]))
                    free_pastures = free_pastures[1:]  # consume

        # Build new structures for the remaining deficit (no per-turn cap)
        coop_deficit = max(0, goose_waiting - len(free_coops))
        pasture_deficit = max(0, pasture_waiting - len(free_pastures))
        total_build = coop_deficit + pasture_deficit

        if total_build > 0:
            empty = state.empty_tiles()
            if empty:
                build_targets = sorted(
                    empty,
                    key=lambda p: manhattan_distance(p, (HALF - 1, HALF - 1)),
                )
                # Build coops first, then pastures
                for _ in range(min(coop_deficit, len(build_targets))):
                    p = build_targets.pop(0)
                    animal_tasks.append(("BUILD_COOP", list(p), None, []))
                    build_tiles_used.add(tuple(p))
                for _ in range(min(pasture_deficit, len(build_targets))):
                    p = build_targets.pop(0)
                    animal_tasks.append(("BUILD_PASTURE", list(p), None, []))
                    build_tiles_used.add(tuple(p))

    # --- Animal daily care (feed + care + collect_fertilizer) ---
    for x, y, t in state.animals():
        if not t.get("fed_today", False) and (x, y) not in feed_queued:
            animal_tasks.append(("FEED", [x, y], "WHEAT", []))
        if not t.get("cared_today", False):
            animal_tasks.append(("CARE", [x, y], None, []))
        if t.get("fertilizer_available", False):
            animal_tasks.append(("COLLECT_FERTILIZER", [x, y], None, []))

    # --- P4: crop watering (bonus window) ---
    for x, y, t in state.plants():
        if (x, y) in water_queued or t.get("watered_today", False):
            continue
        crop = t.get("crop", "WHEAT")
        cd = CROPS[crop]
        if cd["ongoing"]:
            if t.get("fertilized_until_day", -1) >= state.day:
                p4_tasks.append(("WATER", [x, y], None, []))
        else:
            a = state.age(t)
            window_start = (cd["max_yield_day"] + 1) // 2
            if window_start <= a <= cd["max_yield_day"]:
                if t.get("yield_units", 0) < cd["max_yield"]:
                    p4_tasks.append(("WATER", [x, y], None, []))

    # --- P4b: maintenance watering (prevent streak avalanche) ---
    # Water plants not queued in P1/P4, but ONLY in the last hours of the day
    # (hour >= 18). Early in the day, productive tasks (harvest, bonus-window
    # watering, animal care) should take priority. Late in the day, we catch
    # any still-unwatered plants to prevent them from all hitting streak=1
    # tomorrow and overwhelming P1.
    cur_hour = state.obs.get("hour", 0)
    if cur_hour >= 18:
        p4_queued = set((task[1][0], task[1][1]) for task in p4_tasks)
        for x, y, t in state.plants():
            if (x, y) in water_queued or (x, y) in p4_queued or t.get("watered_today", False):
                continue
            crop = t.get("crop", "WHEAT")
            cd = CROPS[crop]
            a = state.age(t)
            if cd["ongoing"]:
                # Ongoing crops: always worth keeping alive
                p4b_tasks.append(("WATER", [x, y], None, []))
            else:
                # One-time crops: only if still maturing
                if a < cd["max_yield_day"] and t.get("yield_units", 0) < cd["max_yield"]:
                    p4b_tasks.append(("WATER", [x, y], None, []))

    # --- P5: fertilize ---
    if state.shed.get("FERTILIZER", 0) > 0:
        for x, y in state.fertilize_candidates():
            p5_tasks.append(("FERTILIZE", [x, y], "FERTILIZER", []))

    # --- Inventory management (DROP at shed when full) ---
    for wid in list(unassigned.keys()):
        inv = inventory_of(wid)
        total = sum(inv.values()) if inv else 0
        wpos = unassigned[wid]
        dist = manhattan_distance(wpos, closest_shed_tile(wpos))
        panic_drop = state.day >= 29 and state.obs.get("hour", 0) >= 23 - dist
        if total >= 15 or (total > 0 and panic_drop):
            shed_pos = closest_shed_tile(wpos)
            if list(wpos) == list(shed_pos):
                assign(wid, ["DROP"])
            else:
                m = move_towards(wpos, shed_pos)
                assign(wid, [m] if m else ["PASS"])

    # --- Dispatch: emergencies first ---
    assign_tier(p1_tasks)

    # --- Dispatch: reserve workers for animal tasks ---
    # Guarantee that animal care always gets workers before crop tasks can
    # consume the entire pool.  We reserve up to len(animal_tasks) workers
    # (but at least 1 if any animal tasks exist) from the pool, dispatch
    # animal tasks on those, then release any unused ones back for crops.
    if animal_tasks and unassigned:
        # Snapshot of all available workers before animal dispatch
        all_available = set(unassigned.keys())
        # How many to reserve (at most all of them if animal tasks dominate)
        reserve_count = min(len(animal_tasks), len(all_available))
        # Pick the closest workers to the average animal-task position
        if reserve_count > 0:
            assign_tier(animal_tasks)

    # --- Dispatch: remaining tiers (harvest, watering, fertilize) ---
    for tier in [p2_tasks, p4_tasks, p4b_tasks, p5_tasks]:
        assign_tier(tier)

    # NEW: Combined P3 (Weeds) and P7 (Planting) distance-based heuristic.
    # We iterate workers first, assigning them to the nearest available empty tile or weed.
    # Weeds get a cost penalty of +1 (1 extra turn to DIG).
    seed_pool = []
    if state.day <= 27:
        plantable = []
        for c, n in state.seeds.items():
            if n > 0:
                cd = CROPS[c]
                if cd["ongoing"]:
                    time_to_max = cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
                else:
                    time_to_max = cd["max_yield_day"]
                if (29 - state.day) >= time_to_max:
                    plantable.append((c, n))
        for crop, n in plantable:
            seed_pool.extend([crop] * n)

    empty_set = set(state.empty_tiles())
    # Remove tiles reserved for animal structure construction
    for pos in list(empty_set):
        if pos in build_tiles_used:
            empty_set.remove(pos)
            
    available_empty = list(empty_set)
    available_weeds = state.weeds()

    while (available_empty or available_weeds) and unassigned:
        best_pair = None
        best_cost = float('inf')
        best_type = None
        best_target = None
        
        for wid in list(unassigned.keys()):
            wpos = unassigned[wid]
            
            for pos in available_empty:
                cost = manhattan_distance(wpos, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = wid
                    best_target = pos
                    best_type = "empty"
                    
            for pos in available_weeds:
                cost = manhattan_distance(wpos, pos) + 1
                if cost < best_cost:
                    best_cost = cost
                    best_pair = wid
                    best_target = pos
                    best_type = "weed"
                    
        if best_pair:
            wid = best_pair
            wpos = unassigned[wid]
            if best_type == "empty":
                available_empty.remove(best_target)
                if list(wpos) == list(best_target):
                    if seed_pool:
                        crop = seed_pool.pop(0)
                        assign(wid, ["PLANT", crop])
                    else:
                        assign(wid, ["PASS"])
                else:
                    m = move_towards(wpos, best_target)
                    assign(wid, [m] if m else ["PASS"])
            elif best_type == "weed":
                available_weeds.remove(best_target)
                if list(wpos) == list(best_target):
                    assign(wid, ["DIG"])
                else:
                    m = move_towards(wpos, best_target)
                    assign(wid, [m] if m else ["PASS"])
        else:
            break

    # Idle workers: hover near the shed where pickups/drops happen.
    for wid in list(unassigned.keys()):
        wpos = unassigned[wid]
        shed_pos = closest_shed_tile(wpos)
        if list(wpos) != list(shed_pos):
            m = move_towards(wpos, shed_pos)
            assign(wid, [m] if m else ["PASS"])
        else:
            assign(wid, ["PASS"])

    return {
        "farmer": actions["farmer"] or ["PASS"],
        "hands": [a or ["PASS"] for a in actions["hands"]],
        "market": market,
    }


if __name__ == "__main__":
    from copy import deepcopy

    def fresh_obs(day=0):
        """Realistic observation matching the environment schema."""
        tiles = [["LOCKED" if (x >= 5 or y >= 5) else None for x in range(10)] for y in range(10)]
        farm = {
            "money": 3000.0,
            "farmer": [4, 4],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
            "tiles": tiles,
        }
        return {
            "step": 0,
            "day": day,
            "hour": 0,
            "player": 0,
            "farms": [farm, deepcopy(farm)],
            "market": {"inventory": {}, "prices": {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}},
            "town": {"unlocked_shops": []},
            "private": {
                "shed": {p: 0 for p in PRODUCTS + list(ANIMALS)},
                "seeds": {c: 0 for c in CROPS},
                "inventories": [{}],
            },
        }

    print("--- Day 0: initial state ---")
    action = agent(fresh_obs(day=0))
    print("Agent Action:", action)

    print("\n--- Day 2: wheat plant, goose needing feed, weed ---")
    obs = fresh_obs(day=2)
    obs["step"] = 2 * 24
    # Wheat planted day 0, kept watered -> yield 3; at age 2 (bonus window)
    obs["farms"][0]["tiles"][4][3] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 3, "max_lifespan_step": (0 + 4 + 1) * 24,
        "fertilized_until_day": -1,
    }
    # A coop holding a goose that is one day away from escaping
    obs["farms"][0]["tiles"][3][4] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 1,
        "yield_units": 0, "consecutive_unfed": 1, "fed_today": False,
        "cared_today": False, "fertilizer_available": False, "pending_care_bonus": 0,
    }
    # A weed blocking a tile
    obs["farms"][0]["tiles"][3][3] = {"kind": "WEED"}
    obs["private"]["seeds"] = {"WHEAT": 2, "CARROT": 2}
    obs["private"]["shed"]["WHEAT"] = 3
    action = agent(obs)
    print("Agent Action:", action)

    print("\n--- Day 10: with two hands, goose on farm ---")
    obs = fresh_obs(day=10)
    obs["step"] = 10 * 24
    obs["farms"][0]["hands"] = [[5, 4], [4, 5]]
    # Mature wheat ready for harvest (age 4+, full watering bonus)
    obs["farms"][0]["tiles"][4][3] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 6,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 4, "max_lifespan_step": (6 + 4 + 1) * 24,
        "fertilized_until_day": -1,
    }
    # Goose producing an egg
    obs["farms"][0]["tiles"][3][4] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 6,
        "yield_units": 1, "consecutive_unfed": 0, "fed_today": False,
        "cared_today": False, "fertilizer_available": True, "pending_care_bonus": 0,
    }
    obs["private"]["shed"]["WHEAT"] = 8
    obs["private"]["shed"]["FERTILIZER"] = 2
    obs["private"]["inventories"] = [{}, {"WHEAT": 2}, {}]
    action = agent(obs)
    print("Agent Action:", action)