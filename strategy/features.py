"""State feature extraction for the strategic decision layer.

Extracts a fixed-size numeric vector from the observation, capturing own state,
opponent visible state, and market conditions.  This vector feeds the value
network and the candidate-plan scorer.
"""

import math
import numpy as np

# pyrefly: ignore [missing-import]
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, PRODUCTS, ANIMALS, MARKET_PARAMS,
)

# Canonical orderings so feature indices are deterministic.
CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                  "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
SHOP_TYPES = [
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
]

FEATURE_DIM = 78  # 64 state features + 14 plan features

def encode_plan(features, plan):
    """Augment the 64-dim state feature vector with a 14-dim plan encoding.
    
    Plan Encoding (14 dims):
    [0]: buy_land (1.0 for True, -1.0 for False, 0.0 for None)
    [1]: hire_target (target / 6.0, or 0.0 if None)
    [2-5]: sell_hold premium goods (1.0 for hold, 0.0 for sell/None) 
           Order: MELON, STRAWBERRY, MILK, WOOL
    [6-10]: buy_seed crop qty / 100.0 (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON)
    [11-13]: buy_animal qty / 10.0 (GOOSE, COW, SHEEP)
    """
    plan_feats = [0.0] * 14
    
    if plan is None:
        plan = {}
        
    # 0: buy_land
    if "buy_land" in plan and plan["buy_land"] is not None:
        plan_feats[0] = 1.0 if plan["buy_land"] else -1.0
        
    # 1: hire_target
    if "hire_target" in plan and plan["hire_target"] is not None:
        plan_feats[1] = plan["hire_target"] / 6.0
        
    # 2-5: sell_hold premium
    sell_hold = plan.get("sell_hold", {})
    premium = ["MELON", "STRAWBERRY", "MILK", "WOOL"]
    for i, p in enumerate(premium):
        if sell_hold.get(p) == "hold":
            plan_feats[2 + i] = 1.0
            
    # 6-10: buy_seed
    buy_seed = plan.get("buy_seed")
    if buy_seed is not None:
        try:
            idx = CROP_ORDER.index(buy_seed["crop"])
            plan_feats[6 + idx] = buy_seed["qty"] / 100.0
        except ValueError:
            pass
            
    # 11-13: buy_animal
    buy_animal = plan.get("buy_animal")
    if buy_animal is not None:
        try:
            idx = ANIMAL_ORDER.index(buy_animal["type"])
            plan_feats[11 + idx] = buy_animal["qty"] / 10.0
        except ValueError:
            pass
            
    # Concatenate features and plan_feats
    return np.concatenate([features, np.array(plan_feats, dtype=np.float32)])


def _log_money(m):
    """Log-scaled money.  Handles negatives gracefully."""
    return math.copysign(math.log1p(abs(m)), m) / 10.0


def _count_tiles(tiles):
    """Count tile types on a 10x10 board.

    Returns dict with keys: empty, locked, weed, and per-crop / per-structure
    counts, plus aggregate yield_units per crop/animal.
    """
    counts = {
        "empty": 0, "locked": 0, "weed": 0,
    }
    for crop in CROP_ORDER:
        counts[f"plant_{crop}"] = 0
        counts[f"yield_{crop}"] = 0
    counts["coop_occupied"] = 0
    counts["coop_empty"] = 0
    counts["pasture_occupied"] = 0
    counts["pasture_empty"] = 0
    counts["animal_yield"] = 0
    counts["total_animals"] = 0

    for row in tiles:
        for tile in row:
            if tile is None:
                counts["empty"] += 1
            elif tile == "LOCKED":
                counts["locked"] += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind", "")
                if kind == "WEED":
                    counts["weed"] += 1
                elif kind == "PLANT":
                    crop = tile.get("crop", "WHEAT")
                    counts[f"plant_{crop}"] = counts.get(f"plant_{crop}", 0) + 1
                    counts[f"yield_{crop}"] = counts.get(f"yield_{crop}", 0) + tile.get("yield_units", 0)
                elif kind == "COOP":
                    if "animal" in tile:
                        counts["coop_occupied"] += 1
                        counts["total_animals"] += 1
                        counts["animal_yield"] += tile.get("yield_units", 0)
                    else:
                        counts["coop_empty"] += 1
                elif kind == "PASTURE":
                    if "animal" in tile:
                        counts["pasture_occupied"] += 1
                        counts["total_animals"] += 1
                        counts["animal_yield"] += tile.get("yield_units", 0)
                    else:
                        counts["pasture_empty"] += 1
    return counts


def extract_features(obs):
    """Convert a raw observation dict into a numpy float32 feature vector.

    The vector is always FEATURE_DIM long, regardless of game state.
    """
    player_id = obs["player"]
    opp_id = 1 - player_id
    me = obs["farms"][player_id]
    opp = obs["farms"][opp_id]
    private = obs.get("private", {})
    market = obs.get("market", {})
    town = obs.get("town", {})

    day = obs.get("day", 0)
    hour = obs.get("hour", 0)

    feats = []

    # --- Temporal features (3) ---
    feats.append(day / 29.0)                    # season progress [0, 1]
    feats.append(hour / 23.0)                   # hour within day [0, 1]
    feats.append(max(0, 29 - day) / 29.0)       # remaining fraction

    # --- Own money (1) ---
    feats.append(_log_money(me.get("money", 0)))

    # --- Own tile counts (12) ---
    my_tc = _count_tiles(me.get("tiles", []))
    feats.append(my_tc["empty"] / 25.0)
    feats.append(my_tc["weed"] / 25.0)
    for crop in CROP_ORDER:
        feats.append(my_tc.get(f"plant_{crop}", 0) / 25.0)
    feats.append(my_tc["coop_occupied"] / 6.0)
    feats.append(my_tc["pasture_occupied"] / 6.0)
    feats.append(my_tc["coop_empty"] / 6.0)
    feats.append(my_tc["pasture_empty"] / 6.0)

    # --- Own crop yields (5) ---
    for crop in CROP_ORDER:
        max_y = CROPS[crop]["max_yield"]
        feats.append(my_tc.get(f"yield_{crop}", 0) / max(1, max_y * 25))

    # --- Own animal yield + count (2) ---
    feats.append(my_tc["total_animals"] / 6.0)
    feats.append(my_tc["animal_yield"] / 24.0)

    # --- Own shed inventory (9) ---
    shed = private.get("shed", {})
    for prod in PRODUCT_ORDER:
        feats.append(min(shed.get(prod, 0), 100) / 100.0)

    # --- Own seeds (5) ---
    seeds = private.get("seeds", {})
    for crop in CROP_ORDER:
        feats.append(min(seeds.get(crop, 0), 50) / 50.0)

    # --- Own infrastructure (2) ---
    feats.append(len(me.get("unlocked_quadrants", ["NW"])) / 4.0)
    feats.append((1 + len(me.get("hands", []))) / 6.0)   # total workers

    # --- Opponent visible state (5) ---
    feats.append(_log_money(opp.get("money", 0)))
    opp_tc = _count_tiles(opp.get("tiles", []))
    feats.append(opp_tc["total_animals"] / 6.0)
    total_opp_plants = sum(opp_tc.get(f"plant_{c}", 0) for c in CROP_ORDER)
    feats.append(total_opp_plants / 100.0)
    feats.append(len(opp.get("unlocked_quadrants", ["NW"])) / 4.0)
    feats.append((1 + len(opp.get("hands", []))) / 6.0)

    # --- Market prices (9) ---
    prices = market.get("prices", {})
    for prod in PRODUCT_ORDER:
        base = MARKET_PARAMS.get(prod, {}).get("base", 50)
        feats.append(prices.get(prod, base) / max(1, base * 2))

    # --- Shop unlocks (8) ---
    shops = town.get("unlocked_shops", [])
    for shop_type in SHOP_TYPES:
        feats.append(shops.count(shop_type) / 3.0)

    # --- Pad to 64 (original state dimension) ---
    while len(feats) < 64:
        feats.append(0.0)

    return np.array(feats[:64], dtype=np.float32)
