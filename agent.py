# submission.py
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
    sell_candidates = [
        ("FERTILIZER", max(0, shed.get("FERTILIZER", 0) - fert_reserve)),
        ("CARROT", shed.get("CARROT", 0)),
        ("TOMATO", shed.get("TOMATO", 0)),
        ("EGG", shed.get("EGG", 0)),
        ("WHEAT", sellable_wheat),
        ("STRAWBERRY", shed.get("STRAWBERRY", 0)),
        ("MELON", shed.get("MELON", 0)),
        ("MILK", shed.get("MILK", 0)),
        ("WOOL", shed.get("WOOL", 0)),
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
    hour = state.obs.get("hour", 0)

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

    # ---- HIRE farm hands (day-hire; very cheap, reset each day) ----
    # Cost = fib(n) for the n-th hire today (1,1,2,3,5,...). Each hand adds up
    # to 24 actions/day, so 3-4 hands are easily profitable once the farm grows.
    if hour == 0 and 1 <= day <= 24 and money > 50:
        have_hands = len(state.me["hands"])
        target = 2 if day < 3 else 3 if day < 12 else 4
        need = max(0, target - have_hands)
        for i in range(need):
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

    # ---- BUY_SEED (only plant what we have room for) ----
    # Self-limiting: buy when seeds are low relative to empty tiles. Not strictly
    # hour-gated because 3 workers planting 1 seed/turn burn up to 72 seeds/day;
    # hour-0-only restocking starves them and idles tiles. `need_seeds` bounds
    # the order to what could actually be planted.
    # Weeds get dug (P3) and become plantable the same day, so count them
    # toward the seed requirement too — otherwise we'd run out of seeds right
    # after clearing.
    need_seeds = max(0, empty_n + weeds_n - seed_count)
    hands_n = len(state.me["hands"])
    # Restock when we have less than 1 hour of work per worker (workers each
    # plant 1 seed/turn), plus a small buffer. This keeps tiles planted all day
    # without the degenerate 24x/day buying that hoards unusable seeds.
    seed_burn = hands_n + 1  # farmer + hired hands
    seed_threshold = max(2 * seed_burn, 4)
    if seed_count < seed_threshold and need_seeds > 0 and day <= 26:
        if day <= 7:  # Early: wheat + carrot
            wn = min(need_seeds, 20)
            if money >= wn * 10:
                orders.append(["BUY_SEED", "WHEAT", wn]); money -= wn * 10
            cn = min(max(0, need_seeds - wn), 10)
            if cn > 0 and money >= cn * 20:
                orders.append(["BUY_SEED", "CARROT", cn]); money -= cn * 20
        elif day <= 17:  # Mid: wheat + tomato (tomato needs 8 days to first yield)
            wn = min(need_seeds, 20)
            if money >= wn * 10:
                orders.append(["BUY_SEED", "WHEAT", wn]); money -= wn * 10
            tn = min(max(0, need_seeds - wn), 10)
            if tn > 0 and money >= tn * 50:
                orders.append(["BUY_SEED", "TOMATO", tn]); money -= tn * 50
        elif day <= 19 and hour == 0:  # Melon window (day 18-19): melon needs 10 days
            # Hour-gated: otherwise the env buys 1 melon EVERY hour of both
            # days (24x/day at $80 each) — a huge cash drain.
            if need_seeds > 0 and money >= 80:
                orders.append(["BUY_SEED", "MELON", 1]); money -= 80
        elif day <= 26:  # Late: fast crops only
            wn = min(need_seeds, 20)
            if money >= wn * 10:
                orders.append(["BUY_SEED", "WHEAT", wn]); money -= wn * 10

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

    def assign_task(task):
        """Assign one task (type, pos, requires, args) to the closest free worker.
        If the worker is missing a required inventory item, route them to the shed
        to PICKUP first; their inventory persists, so they complete the task on a
        later turn once they have the item."""
        if not unassigned:
            return
        typ, pos, requires, args = task
        if requires:
            # Prefer a worker already carrying the required item to avoid a
            # shed round-trip.
            holders = [wid for wid in unassigned if inventory_of(wid).get(requires, 0) > 0]
            wid = (min(holders, key=lambda w: manhattan_distance(unassigned[w], pos))
                   if holders else get_closest_worker(pos, unassigned))
        else:
            wid = get_closest_worker(pos, unassigned)
        if wid is None:
            return
        wpos = unassigned[wid]
        inv = inventory_of(wid)
        has_req = requires is None or inv.get(requires, 0) > 0

        if list(wpos) == list(pos) and has_req:
            assign(wid, [typ] + (args or []))
        elif requires and not has_req:
            # Worker needs the required item from the shed before acting.
            shed_pos = closest_shed_tile(wpos)
            if list(wpos) == list(shed_pos):
                assign(wid, ["PICKUP", requires, 5])
            else:
                m = move_towards(wpos, shed_pos)
                assign(wid, [m] if m else ["PASS"])
        else:
            m = move_towards(wpos, pos)
            assign(wid, [m] if m else ["PASS"])

    # Build the task queue by priority.
    # Each entry: (type, [x, y], requires_inventory_item, args)
    tasks = []

    # Track tiles already queued so P1 and P4 don't both schedule the same
    # water/feed (actions only become visible in the next observation).
    water_queued = set()
    feed_queued = set()

    # P1: Critical survival — plants one missed day from becoming weeds,
    #     animals one missed day from escaping.
    for x, y, t in state.plants():
        if not t.get("watered_today", False) and t.get("consecutive_unwatered", 0) >= 1:
            tasks.append(("WATER", [x, y], None, []))
            water_queued.add((x, y))
    for x, y, t in state.animals():
        if not t.get("fed_today", False) and t.get("consecutive_unfed", 0) >= 1:
            tasks.append(("FEED", [x, y], "WHEAT", []))
            feed_queued.add((x, y))

    # P2: Harvest ripe plants and animals with accumulated produce.
    for x, y in state.harvestable_plants():
        tasks.append(("HARVEST", [x, y], None, []))
    for x, y in state.harvestable_animals():
        tasks.append(("HARVEST", [x, y], None, []))

    # P3: Clear weeds so the land can be planted again.
    for x, y in state.weeds():
        tasks.append(("DIG", [x, y], None, []))

    # P4: Daily upkeep that adds yield value.
    #   - One-time crops: water only inside their bonus window and only if
    #     watering still adds yield (yield_units below the cap). Plants past
    #     the window are at max yield already — watering wastes a worker turn
    #     (their survival needs are covered by P1).
    #   - Ongoing crops (tomato/strawberry): watering only boosts yield while
    #     fertilized (fertilized+watered production days yield 2 instead of 1).
    #     Unfertilized ongoing watering is survival-only → P1.
    #   - Animals: feed the remainder, CARE every animal (banks +1 to the next
    #     production), and collect the free fertilizer.
    for x, y, t in state.plants():
        if (x, y) in water_queued or t.get("watered_today", False):
            continue
        crop = t.get("crop", "WHEAT")
        cd = CROPS[crop]
        if cd["ongoing"]:
            if t.get("fertilized_until_day", -1) >= state.day:
                tasks.append(("WATER", [x, y], None, []))
        else:
            a = state.age(t)
            window_start = (cd["max_yield_day"] + 1) // 2
            if window_start <= a <= cd["max_yield_day"]:
                if t.get("yield_units", 0) < cd["max_yield"]:
                    tasks.append(("WATER", [x, y], None, []))
    for x, y, t in state.animals():
        if not t.get("fed_today", False) and (x, y) not in feed_queued:
            tasks.append(("FEED", [x, y], "WHEAT", []))
    for x, y, t in state.animals():
        if not t.get("cared_today", False):
            tasks.append(("CARE", [x, y], None, []))
        if t.get("fertilizer_available", False):
            tasks.append(("COLLECT_FERTILIZER", [x, y], None, []))

    # P5: Crop FERTILIZE (if we have some).
    if state.shed.get("FERTILIZER", 0) > 0:
        for x, y in state.fertilize_candidates():
            tasks.append(("FERTILIZE", [x, y], "FERTILIZER", []))

    # P6: Animal logistics — build coops and place animals waiting in the shed.
    waiting_animals = state.animals_in_shed()
    if waiting_animals > 0:
        free_coops = [(x, y, t) for x, y, t in state.empty_structures() if t.get("kind") == "COOP"]
        if not free_coops:
            empty = state.empty_tiles()
            if empty:
                # Build one coop per waiting goose (up to 2/turn) so a backlog
                # doesn't keep animals in the shed for days.
                build_targets = sorted(empty, key=lambda p: manhattan_distance(p, (HALF - 1, HALF - 1)))
                for _ in range(min(waiting_animals, 2, len(build_targets))):
                    p = build_targets.pop(0)
                    tasks.append(("BUILD_COOP", list(p), None, []))
        else:
            # Only emit as many PLACE tasks as there are animals waiting.
            for (x, y, t) in free_coops[:waiting_animals]:
                tasks.append(("PLACE", [x, y], "GOOSE", ["GOOSE"]))

    # P7: Plant seeds (only crops that still have time to mature by day 29).
    # Day 27 wheat/carrot still produce a harvest by day 29.
    if state.day <= 27:
        plantable = [(c, n) for c, n in state.seeds.items() if n > 0
                     and SEASON - state.day >= CROPS[c]["first_yield_day"]]
        seed_pool = []
        for crop, n in plantable:
            seed_pool.extend([crop] * n)

        empty = state.empty_tiles()
        # Don't plant on tiles reserved for higher-priority building tasks.
        for (typ, pos, _, _) in tasks:
            if typ in ("BUILD_COOP", "BUILD_PASTURE") and tuple(pos) in empty:
                empty.remove(tuple(pos))
        for pos in empty:
            if not seed_pool:
                break
            crop = seed_pool.pop(0)
            tasks.append(("PLANT", list(pos), None, [crop]))

    # P8: Inventory management — large inventories need DROPping at the shed
    #     so harvested goods can be sold from the shed.
    for wid in list(unassigned.keys()):
        inv = inventory_of(wid)
        total = sum(inv.values()) if inv else 0
        if total >= 15:
            shed_pos = closest_shed_tile(unassigned[wid])
            if list(unassigned[wid]) == list(shed_pos):
                assign(wid, ["DROP"])
            else:
                m = move_towards(unassigned[wid], shed_pos)
                assign(wid, [m] if m else ["PASS"])

    # Assign all tasks by priority.
    for task in tasks:
        if not unassigned:
            break
        assign_task(task)

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