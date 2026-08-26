"""Candidate plan generator for the strategic decision layer.

Produces a small set of concrete market-order override plans that the
strategic layer evaluates.  Each plan specifies *overrides* to the
default heuristic's ``build_market()`` behaviour.

Plans are rule-based (not learned) — we enumerate the genuinely uncertain
strategic choices and let the value network pick the best one.
"""

# pyrefly: ignore [missing-import]
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, MARKET_PARAMS


def _can_afford(money, cost, buffer=100):
    return money >= cost + buffer


def generate_candidates(obs):
    """Return a list of candidate plan dicts for the current observation.

    Every plan dict has the following keys (all optional — missing keys
    mean "use the heuristic default"):

        sell_hold : dict[str, str]
            Product → "sell" or "hold".  Missing products default to "sell".
        buy_land : bool | None
            True = force buy, False = force skip, None = heuristic decides.
        buy_seed : dict | None
            {"crop": str, "qty": int} override, or None for heuristic.
        buy_animal : dict | None
            {"type": str, "qty": int} override, or None for heuristic.
        hire_target : int | None
            Target number of *total* workers (farmer + hands), or None.
        label : str
            Human-readable name for logging.
    """
    player_id = obs["player"]
    me = obs["farms"][player_id]
    day = obs.get("day", 0)
    money = me.get("money", 0)
    unlocked = len(me.get("unlocked_quadrants", ["NW"]))
    shed = obs.get("private", {}).get("shed", {})
    seeds = obs.get("private", {}).get("seeds", {})

    # Count empty tiles
    empty_n = sum(1 for row in me["tiles"] for t in row if t is None)

    # Land costs
    land_cost = {1: 1000, 2: 2000, 3: 4000}.get(unlocked, float("inf"))

    # Animals on farm + shed
    animals_farm = 0
    for row in me["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and "animal" in t:
                animals_farm += 1
    animals_shed = sum(shed.get(a, 0) for a in ["GOOSE", "COW", "SHEEP"])
    total_animals = animals_farm + animals_shed

    candidates = []

    # --- Plan 0: BASELINE (heuristic as-is) ---
    candidates.append({
        "label": "baseline",
    })

    # --- Plan 1: AGGRESSIVE LAND ---
    if unlocked < 4 and _can_afford(money, land_cost, 200) and day <= 24:
        candidates.append({
            "buy_land": True,
            "label": "aggressive_land",
        })

    # --- Plan 2: HOLD PREMIUM GOODS ---
    # Only makes sense if we have premium goods in shed and it's not endgame
    premium = ["MELON", "STRAWBERRY", "MILK", "WOOL"]
    has_premium = any(shed.get(p, 0) > 0 for p in premium)
    if has_premium and day < 27:
        candidates.append({
            "sell_hold": {p: "hold" for p in premium},
            "label": "hold_premium",
        })

    # --- Plan 3: SELL EVERYTHING NOW ---
    if has_premium and day < 28:
        candidates.append({
            "sell_hold": {p: "sell" for p in premium},
            "label": "sell_all_now",
        })

    # --- Plan 4: ANIMAL RUSH (buy geese aggressively) ---
    if total_animals < 6 and day <= 18 and _can_afford(money, 300, 200):
        n_buy = min(2, 6 - total_animals)
        if _can_afford(money, 300 * n_buy, 200):
            candidates.append({
                "buy_animal": {"type": "GOOSE", "qty": n_buy},
                "label": f"animal_rush_{n_buy}",
            })

    # --- Plan 5: MELON PIVOT ---
    # In the melon window, consider switching all seeds to melon
    if 14 <= day <= 19 and _can_afford(money, 80, 200):
        melon_qty = min(empty_n, int((money - 200) // 80))
        if melon_qty > 0:
            candidates.append({
                "buy_seed": {"crop": "MELON", "qty": melon_qty},
                "label": f"melon_pivot_{melon_qty}",
            })

    # --- Plan 6: CONSERVATIVE (skip all buys) ---
    if day >= 5:  # only meaningful after opening
        candidates.append({
            "buy_land": False,
            "buy_seed": None,
            "buy_animal": None,
            "hire_target": 1,  # farmer only
            "label": "conservative",
        })

    # --- Plan 7: HIRE HEAVY ---
    if day <= 20 and _can_afford(money, 15, 100):
        candidates.append({
            "hire_target": 5,
            "label": "hire_heavy",
        })

    # --- Plan 8: COW INVESTMENT (if day allows ROI) ---
    if day <= 14 and total_animals < 6 and _can_afford(money, 400, 300):
        candidates.append({
            "buy_animal": {"type": "COW", "qty": 1},
            "label": "cow_invest",
        })

    # --- Plan 9: WHEAT ECONOMY (fast cycling) ---
    seed_count = sum(seeds.values())
    if day <= 25 and empty_n > 5 and seed_count < 5:
        wheat_qty = min(empty_n, int((money - 100) // 10))
        if wheat_qty > 0:
            candidates.append({
                "buy_seed": {"crop": "WHEAT", "qty": wheat_qty},
                "label": f"wheat_economy_{wheat_qty}",
            })

    # Always have at least 2 candidates (baseline + one alternative)
    if len(candidates) < 2:
        candidates.append({
            "sell_hold": {},
            "label": "noop_alternative",
        })

    return candidates
