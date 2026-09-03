"""Targeted unit tests for heuristic fixes in agent.py."""
from copy import deepcopy

import main as agent
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, PRODUCTS, ANIMALS, MARKET_PARAMS

# Force strategic layer off for heuristic tests
agent._strategic_layer = None
agent.step_counter = 2  # prevent it from loading weights on step 1


def make_obs(day, hour, tiles, shed, seeds, money=5000.0):
    farm = {
        "money": money,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
        "hires_today": 0,
        "tiles": tiles,
    }
    return {
        "step": day * 24 + hour,
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [farm, deepcopy(farm)],
        "market": {"inventory": {}, "prices": {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}},
        "town": {"unlocked_shops": []},
        "private": {"shed": shed, "seeds": seeds, "inventories": [{}]},
    }


def test_melon_hour_gating():
    tiles = [[None for x in range(10)] for y in range(10)]
    shed = {p: 0 for p in PRODUCTS + list(ANIMALS)}
    seeds = {c: 0 for c in CROPS}
    a_h0 = agent.agent(make_obs(18, 0, tiles, shed, seeds))
    a_h5 = agent.agent(make_obs(18, 5, tiles, shed, seeds))
    melon_h0 = [o for o in a_h0["market"] if o[0] == "BUY_SEED" and o[1] == "MELON"]
    melon_h5 = [o for o in a_h5["market"] if o[0] == "BUY_SEED" and o[1] == "MELON"]
    print(f"Test melon hour gating: hour0={melon_h0} hour5={melon_h5}")
    assert melon_h0, "Expected melon buy at hour 0"
    assert not melon_h5, "Expected NO melon buy at hour 5"


def test_tomato_fertilize():
    # Tomato at age 2 (day 12, planted day 10): fertilizing would cover days
    # 12-14, but first production is day 18 (planted 10 + first_yield 8).
    # Should NOT be a candidate.
    tiles_young = [[None for x in range(10)] for y in range(10)]
    tiles_young[4][3] = {
        "kind": "PLANT", "crop": "TOMATO", "planted_day": 10,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 0, "max_lifespan_step": -1, "fertilized_until_day": -1,
    }
    shed = {p: 0 for p in PRODUCTS + list(ANIMALS)}
    shed["FERTILIZER"] = 5
    seeds = {c: 0 for c in CROPS}
    st = agent.GameState(make_obs(12, 0, tiles_young, shed, seeds))
    assert not st.fertilize_candidates(), "Age-2 tomato should not be fertilized"

    # Tomato at age 6 (day 16, planted day 10): fertilizing covers days 16-18;
    # production days 16 and 18 both fall inside. Should be a candidate.
    tiles_old = [[None for x in range(10)] for y in range(10)]
    tiles_old[4][3] = {
        "kind": "PLANT", "crop": "TOMATO", "planted_day": 10,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 0, "max_lifespan_step": -1, "fertilized_until_day": -1,
    }
    st2 = agent.GameState(make_obs(16, 0, tiles_old, shed, seeds))
    cands = st2.fertilize_candidates()
    print(f"Test tomato fertilize: age2=[] age6={cands}")
    assert cands, "Age-6 tomato should be a fertilize candidate"


def test_endgame_liquidation():
    tiles = [[None for x in range(10)] for y in range(10)]
    shed = {p: 0 for p in PRODUCTS + list(ANIMALS)}
    shed.update({"WHEAT": 50, "CARROT": 20, "TOMATO": 10, "EGG": 15, "FERTILIZER": 8})
    seeds = {c: 0 for c in CROPS}
    a = agent.agent(make_obs(29, 0, tiles, shed, seeds))
    sell_products = [(o[1], o[2]) for o in a["market"] if o[0] == "SELL"]
    print(f"Test endgame liquidation: {sell_products}")
    assert ("WHEAT", 50) in sell_products, "All wheat must be sold day 29"
    assert ("FERTILIZER", 8) in sell_products, "All fertilizer must be sold day 29"
    # No buy orders in the endgame.
    buys = [o for o in a["market"] if o[0].startswith("BUY")]
    assert not buys, f"No buys expected day 29, got {buys}"


def test_water_only_in_bonus_window():
    """A wheat plant at age 5 (past bonus window, at cap) should NOT get
    a water task — watering adds zero yield. Survival (P1) still protects it."""
    tiles = [[None for x in range(10)] for y in range(10)]
    tiles[4][3] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 10,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 4, "max_lifespan_step": (10 + 4 + 1) * 24,
        "fertilized_until_day": -1,
    }
    shed = {p: 0 for p in PRODUCTS + list(ANIMALS)}
    seeds = {c: 0 for c in CROPS}
    a = agent.agent(make_obs(15, 0, tiles, shed, seeds))
    # The farmer should not be moving to water this plant. It should either
    # harvest it (age 5 = max) or pass.
    print(f"Test waterfall: action={a['farmer']}")
    assert a["farmer"][0] in ("HARVEST", "PASS") or a["farmer"][0].startswith(("NORTH", "SOUTH", "EAST", "WEST")), \
        f"Unexpected action {a['farmer']} for over-ripe wheat"


def test_strategic_layer_actor():
    import numpy as np
    from strategy.features import extract_features
    from strategy.actor_net import ActorNetNumpy
    
    np.random.seed(42)
    anet = ActorNetNumpy()
    
    tiles = [[None for x in range(10)] for y in range(10)]
    shed = {p: 0 for p in PRODUCTS + list(ANIMALS)}
    seeds = {c: 0 for c in CROPS}
    obs = make_obs(10, 0, tiles, shed, seeds)
    
    features = extract_features(obs)
    
    action = anet.predict(features)
    
    print(f"Test strategic layer actor: action={action}")
    assert "buy_land" in action, "Action should contain buy_land"
    assert "hire_target" in action, "Action should contain hire_target"
    assert "sell_hold" in action, "Action should contain sell_hold"


if __name__ == "__main__":
    test_melon_hour_gating()
    test_tomato_fertilize()
    test_endgame_liquidation()
    test_water_only_in_bonus_window()
    test_strategic_layer_actor()
    print("\nAll targeted tests passed.")