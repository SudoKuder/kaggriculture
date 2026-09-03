import re

with open('main.py', 'r') as f:
    content = f.read()

# 1. Update _ActorNetNumpy predict in main.py
old_predict = """
        return {
            "buy_land": self._sigmoid(logits[0]),
            "hire_target": int(_np.argmax(logits[1:7])) + 1,
            "sell_hold": self._sigmoid(logits[7:16]),
            "buy_seed_crop": int(_np.argmax(logits[16:21])),
            "buy_seed_frac": self._sigmoid(logits[21]),
            "buy_animal_type": int(_np.argmax(logits[22:25])),
            "buy_animal_frac": self._sigmoid(logits[25]),
        }
"""
new_predict = """
        return {
            "buy_land": self._sigmoid(logits[0]),
            "hire_target": int(_np.argmax(logits[1:7])) + 1,
            "sell_hold": self._sigmoid(logits[7:16]),
            "buy_seed_crop": int(_np.argmax(logits[16:21])),
            "buy_seed_frac": self._sigmoid(logits[21]),
            "buy_animal_type": int(_np.argmax(logits[22:25])),
            "buy_animal_frac": self._sigmoid(logits[25]),
            "weed_penalty": self._sigmoid(logits[26]) * 10.0,
            "maint_water_hour": self._sigmoid(logits[27]) * 23.0,
            "panic_drop_hour": self._sigmoid(logits[28]) * 23.0,
            "seed_threshold_mult": self._sigmoid(logits[29]) * 5.0,
            "land_unlock_buffer": self._sigmoid(logits[30]) * 2000.0,
        }
"""
content = content.replace(old_predict.strip(), new_predict.strip())

# 2. Update _StrategicLayer.decide
old_decide = """
        plan["label"] = "learned_action"
        return plan
"""
new_decide = """
        plan["weed_penalty"] = float(action_dict["weed_penalty"])
        plan["maint_water_hour"] = float(action_dict["maint_water_hour"])
        plan["panic_drop_hour"] = float(action_dict["panic_drop_hour"])
        plan["seed_threshold_mult"] = float(action_dict["seed_threshold_mult"])
        plan["land_unlock_buffer"] = float(action_dict["land_unlock_buffer"])

        plan["label"] = "learned_action"
        return plan
"""
content = content.replace(old_decide.strip(), new_decide.strip())

# 3. Update build_market land_unlock_buffer (approx lines 505 and 511)
old_land = """
        if should_unlock and money >= land_cost + 500:
            orders.append(["BUY_LAND"])
            money -= land_cost
        else:
            # Late-season land is still productive with fast wheat (buy day 24, plant
            # immediately, harvest day 29), so keep buying through day 24.
            if should_unlock and unlocked < 4 and empty_n <= 4 and day <= 24 and money >= land_cost + 500:
                orders.append(["BUY_LAND"])
                money -= land_cost
"""
new_land = """
        land_buffer = plan.get("land_unlock_buffer", 500.0) if plan else 500.0
        if should_unlock and money >= land_cost + land_buffer:
            orders.append(["BUY_LAND"])
            money -= land_cost
        else:
            # Late-season land is still productive with fast wheat (buy day 24, plant
            # immediately, harvest day 29), so keep buying through day 24.
            if should_unlock and unlocked < 4 and empty_n <= 4 and day <= 24 and money >= land_cost + land_buffer:
                orders.append(["BUY_LAND"])
                money -= land_cost
"""
content = content.replace(old_land.strip(), new_land.strip())

# 4. Update build_market seed threshold and wage buffer (approx line 530 and 559)
old_seed = """
        need_seeds = max(0, empty_n + weeds_n - seed_count)
        hands_n = len(state.me["hands"])
        seed_burn = hands_n + 1
        seed_threshold = max(2 * seed_burn, 4)
        if seed_count < seed_threshold and need_seeds > 0 and day <= 26:
"""
new_seed = """
        need_seeds = max(0, empty_n + weeds_n - seed_count)
        hands_n = len(state.me["hands"])
        seed_burn = hands_n + 1
        seed_mult = plan.get("seed_threshold_mult", 2.0) if plan else 2.0
        seed_threshold = max(seed_mult * seed_burn, 4.0)
        if seed_count < seed_threshold and need_seeds > 0 and day <= 26:
"""
content = content.replace(old_seed.strip(), new_seed.strip())

# We might not be able to easily replace wage_buffer with exact string match due to formatting, so we can use regex
content = re.sub(
    r"wage_buffer = max\(100, wage_bill \* 3\)",
    r'wage_buffer = max(100, wage_bill * 3)', # Wait! The prompt didn't say wage buffer multiplier, wait... I forgot wage_buffer in the network! 
    # Ah! I didn't add wage_buffer to the network, I only added 5 parameters! Let's check my task list.
    # 1. weed_penalty, 2. maint_water_hour, 3. panic_drop_hour, 4. seed_threshold_mult, 5. land_unlock_buffer.
    # Ah I see, I didn't add wage buffer! I'll leave wage_buffer alone since it wasn't requested.
    content
)

# 5. panic_drop_hour in inventory management
old_inventory = """
    # --- Inventory management (DROP at shed when full) ---
    for wid in list(unassigned.keys()):
        inv = inventory_of(wid)
        total = sum(inv.values()) if inv else 0
        wpos = unassigned[wid]
        dist = manhattan_distance(wpos, closest_shed_tile(wpos))
        panic_drop = state.day >= 29 and state.obs.get("hour", 0) >= 23 - dist
"""
new_inventory = """
    # --- Inventory management (DROP at shed when full) ---
    for wid in list(unassigned.keys()):
        inv = inventory_of(wid)
        total = sum(inv.values()) if inv else 0
        wpos = unassigned[wid]
        dist = manhattan_distance(wpos, closest_shed_tile(wpos))
        panic_hour = _current_plan.get(state.player_id, {}).get("panic_drop_hour", 23.0)
        panic_drop = state.day >= 29 and state.obs.get("hour", 0) >= panic_hour - dist
"""
content = content.replace(old_inventory.strip(), new_inventory.strip())

# 6. weed_penalty in planting/weeding
old_weed = """
            for pos in available_weeds:
                cost = manhattan_distance(wpos, pos) + 1
                if cost < best_cost:
"""
new_weed = """
            for pos in available_weeds:
                weed_penalty = _current_plan.get(state.player_id, {}).get("weed_penalty", 1.0)
                cost = manhattan_distance(wpos, pos) + weed_penalty
                if cost < best_cost:
"""
content = content.replace(old_weed.strip(), new_weed.strip())

# 7. maint_water_hour
old_maint = """
    # --- P4b: maintenance watering (prevent streak avalanche) ---
    # Water plants not queued in P1/P4, but ONLY in the last hours of the day
    # (hour >= 18). Early in the day, productive tasks (harvest, bonus-window
    # watering, animal care) should take priority. Late in the day, we catch
    # any still-unwatered plants to prevent them from all hitting streak=1
    # tomorrow and overwhelming P1.
    cur_hour = state.obs.get("hour", 0)
    if cur_hour >= 18:
"""
new_maint = """
    # --- P4b: maintenance watering (prevent streak avalanche) ---
    # Water plants not queued in P1/P4, but ONLY in the last hours of the day
    # (hour >= 18). Early in the day, productive tasks (harvest, bonus-window
    # watering, animal care) should take priority. Late in the day, we catch
    # any still-unwatered plants to prevent them from all hitting streak=1
    # tomorrow and overwhelming P1.
    cur_hour = state.obs.get("hour", 0)
    maint_hour = _current_plan.get(state.player_id, {}).get("maint_water_hour", 18.0)
    if cur_hour >= maint_hour:
"""
content = content.replace(old_maint.strip(), new_maint.strip())

with open('main.py', 'w') as f:
    f.write(content)
