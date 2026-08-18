# submission.py
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, PRODUCTS

# Global counter to track steps (for logging)
step_counter = 0
DEBUG = True  # Set to False to disable debug output

def log(*args):
    """Helper to print debug statements easily, controlled by the DEBUG flag."""
    if DEBUG:
        print(f"[Agent Log]", *args)


class GameState:
    """Helper class to parse and query the observation dictionary."""
    def __init__(self, obs):
        self.obs = obs
        self.day = obs.get("day", 1)
        self.player_id = obs["player"]
        self.me = obs["farms"][self.player_id]
        self.private = obs["private"]
        self.tiles = self.me["tiles"]

    def get_thirsty_plants(self):
        """Returns a list of (x, y, tile) for plants that need watering today."""
        thirsty = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    if not tile.get("watered_today", False):
                        thirsty.append((x, y, tile))
        return thirsty

    def get_harvestable_tiles(self):
        """Returns lists of high-value harvests and normal harvests: (high_value, normal)"""
        high_value = []
        normal = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    if tile.get("kind") == "PLANT":
                        crop_age = self.day - tile.get("planted_day", self.day)
                        plant_type = tile.get("type", "WHEAT")
                        
                        first_yield = CROPS.get(plant_type, {}).get("first_yield_day", 2)
                        max_yield = CROPS.get(plant_type, {}).get("max_yield_day", 4)
                        
                        if crop_age >= max_yield:
                            high_value.append((x, y, tile, "HARVEST"))
                        elif crop_age >= first_yield:
                            normal.append((x, y, tile, "HARVEST"))
                    elif tile.get("kind") in ["COOP", "PASTURE"]:
                        yield_units = tile.get("yield_units", 0)
                        if "animal" in tile and isinstance(tile["animal"], dict):
                            yield_units = tile["animal"].get("yield_units", 0)
                        if yield_units > 0:
                            high_value.append((x, y, tile, "HARVEST"))
                            
        return high_value, normal

    def get_critical_care_tiles(self):
        """Priority 1: Survival. Plants unfed=1, animals unfed=1"""
        critical = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    if tile.get("kind") == "PLANT":
                        if tile.get("consecutive_unwatered", 0) == 1 and not tile.get("watered_today", False):
                            critical.append((x, y, tile, "WATER"))
                    elif tile.get("kind") in ["COOP", "PASTURE"]:
                        unfed = tile.get("consecutive_unfed", 0)
                        fed_today = tile.get("fed_today", False)
                        if "animal" in tile and isinstance(tile["animal"], dict):
                            unfed = tile["animal"].get("consecutive_unfed", 0)
                            fed_today = fed_today or tile["animal"].get("fed_today", False)
                            
                        if unfed == 1 and not fed_today:
                            critical.append((x, y, tile, "FEED"))
        return critical

    def get_maintenance_tiles(self):
        """Priority 3: Maintenance. Rest of unwatered plants and unfed animals."""
        maintenance = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    if tile.get("kind") == "PLANT":
                        if not tile.get("watered_today", False):
                            maintenance.append((x, y, tile, "WATER"))
                    elif tile.get("kind") in ["COOP", "PASTURE"]:
                        fed_today = tile.get("fed_today", False)
                        if "animal" in tile and isinstance(tile["animal"], dict):
                            fed_today = fed_today or tile["animal"].get("fed_today", False)
                        if not fed_today:
                            maintenance.append((x, y, tile, "FEED"))
        return maintenance

    def get_care_and_fertilize_tiles(self):
        """Priority 4: Animal CARE and crop FERTILIZE."""
        care = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    if tile.get("kind") in ["COOP", "PASTURE"]:
                        cared = tile.get("cared_today", False)
                        if "animal" in tile and isinstance(tile["animal"], dict):
                            cared = cared or tile["animal"].get("cared_today", False)
                        if not cared:
                            care.append((x, y, tile, "CARE"))
                    elif tile.get("kind") == "PLANT":
                        # Fertilize high value crops if we have fertilizer?
                        # Simplification: Assume we just FERTILIZE if we have the item, we'll queue the task.
                        # Actually, agent needs to know if we have fertilizer. We'll queue it anyway.
                        if not tile.get("fertilized_today", False):
                            # Only high value crops: MELON, STRAWBERRY
                            if tile.get("type") in ["MELON", "STRAWBERRY"]:
                                care.append((x, y, tile, "FERTILIZE"))
        return care

    def get_empty_tiles(self):
        """Priority 5: Empty tiles."""
        empty = []
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile is None:
                    empty.append((x, y))
        return empty


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_closest_worker(target_pos, workers_dict):
    """
    Finds the closest worker to a target position.
    workers_dict: dict of {worker_id: (x, y)}
    Returns: closest_worker_id
    """
    closest_worker = None
    min_dist = float('inf')
    
    for worker_id, pos in workers_dict.items():
        dist = manhattan_distance(pos, target_pos)
        if dist < min_dist:
            min_dist = dist
            closest_worker = worker_id
            
    return closest_worker

def move_towards(current_pos, target_pos):
    """Returns the next movement action to reach target_pos."""
    cx, cy = current_pos
    tx, ty = target_pos
    if cx < tx: return "EAST"
    if cx > tx: return "WEST"
    if cy < ty: return "SOUTH"
    if cy > ty: return "NORTH"
    return None

def agent(obs):
    global step_counter
    step_counter += 1
    log(f"--- Step {step_counter} --- Day {obs.get('day', '?')} ---")
    
    state = GameState(obs)
    
    # Market & Phase Logic
    market = []
    day = state.day
    money = state.me["money"]
    shed = state.private.get("shed", {})
    seeds = state.private.get("seeds", {})
    empty_tiles_count = len(state.get_empty_tiles())
    unlocked_quads = len(state.me["unlocked_quadrants"])
    
    # Current seeds count (excluding fertilizer)
    current_seeds = sum(v for k, v in seeds.items() if k != "FERTILIZER")
    seeds_to_buy = max(0, empty_tiles_count + 5 - current_seeds)
    
    # Sell harvested products
    for item, qty in shed.items():
        if item in ["GOOSE", "COW", "SHEEP", "FERTILIZER"]:
            if day >= 26:  # Liquidate in end game
                market.append(["SELL", item, qty])
        else:
            if qty > 0:
                market.append(["SELL", item, qty])

    if day <= 8:
        # Early Game: Fast Cash
        land_cost = 1000 * (2 ** (unlocked_quads - 1)) if unlocked_quads < 4 else float('inf')
        # Target seed cost ~15. Buffer = 25 * 15 = 375
        if empty_tiles_count <= 2 and money >= land_cost + 375:
            market.append(["BUY_LAND"])
            money -= land_cost
            
        wheat_to_buy = seeds_to_buy // 2
        carrot_to_buy = seeds_to_buy - wheat_to_buy
        if wheat_to_buy > 0 and money >= wheat_to_buy * 10:
            market.append(["BUY_SEED", "WHEAT", wheat_to_buy])
            money -= wheat_to_buy * 10
        if carrot_to_buy > 0 and money >= carrot_to_buy * 20:
            market.append(["BUY_SEED", "CARROT", carrot_to_buy])
            money -= carrot_to_buy * 20

    elif day <= 20:
        # Mid Game: High Yield & Animals
        land_cost = 1000 * (2 ** (unlocked_quads - 1)) if unlocked_quads < 4 else float('inf')
        if empty_tiles_count <= 2 and money >= land_cost + 500: # Slightly larger buffer for Mid Game
            market.append(["BUY_LAND"])
            money -= land_cost

        # Animal capacity cap (40% of land)
        total_land = unlocked_quads * 25
        animal_cap = int(0.4 * total_land)
        
        existing_animals = sum(
            1 for row in state.tiles for tile in row 
            if isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]
        )
        animals_in_shed = sum(v for k,v in shed.items() if k in ["GOOSE", "COW", "SHEEP"])
        total_animals = existing_animals + animals_in_shed
        
        if total_animals < animal_cap and money >= 500:
            market.append(["BUY_ANIMAL", "GOOSE", 1]) # Max 1 animal per day
            money -= 500
            
        tomato_to_buy = seeds_to_buy // 2
        strawberry_to_buy = seeds_to_buy - tomato_to_buy
        if tomato_to_buy > 0 and money >= tomato_to_buy * 20:
            market.append(["BUY_SEED", "TOMATO", tomato_to_buy])
            money -= tomato_to_buy * 20
        if strawberry_to_buy > 0 and money >= strawberry_to_buy * 30:
            market.append(["BUY_SEED", "STRAWBERRY", strawberry_to_buy])
            money -= strawberry_to_buy * 30

    elif day <= 25:
        # Late Game: Quick Turnarounds
        melon_to_buy = seeds_to_buy // 2
        wheat_to_buy = seeds_to_buy - melon_to_buy
        if melon_to_buy > 0 and money >= melon_to_buy * 50:
            market.append(["BUY_SEED", "MELON", melon_to_buy])
            money -= melon_to_buy * 50
        if wheat_to_buy > 0 and money >= wheat_to_buy * 10:
            market.append(["BUY_SEED", "WHEAT", wheat_to_buy])
            money -= wheat_to_buy * 10

    else:
        # End Game: Liquidation
        pass # Only selling, handled above

    # Gather available workers
    unassigned_workers = {"farmer": state.me["farmer"]}
    for i, hand in enumerate(state.me["hands"]):
        unassigned_workers[f"hand_{i}"] = hand
        
    worker_actions = {"farmer": ["PASS"], "hands": [["PASS"] for _ in state.me["hands"]]}
    
    def assign_worker(worker_id, action):
        if worker_id == "farmer":
            worker_actions["farmer"] = action
        else:
            idx = int(worker_id.split("_")[1])
            worker_actions["hands"][idx] = action
        del unassigned_workers[worker_id]

    tasks = []
    queued_positions = set()
    
    def enqueue_tasks(tile_list, default_args=None):
        for x, y, tile, action_type in tile_list:
            if (x, y) not in queued_positions:
                tasks.append({"type": action_type, "pos": [x, y], "args": default_args or []})
                queued_positions.add((x, y))

    high_value_harvests, normal_harvests = state.get_harvestable_tiles()

    # Priority 1: Survival (Emergency)
    enqueue_tasks(state.get_critical_care_tiles())
    
    # Priority 2: High-Value Harvests
    enqueue_tasks(high_value_harvests)
    
    # Priority 3: Maintenance
    enqueue_tasks(state.get_maintenance_tiles())
    
    # Priority 2.5: Normal Harvests (Insert here before Care)
    enqueue_tasks(normal_harvests)
    
    # Priority 4: Animal Care & Fertilization
    if state.private.get("shed", {}).get("FERTILIZER", 0) > 0 or state.private.get("seeds", {}).get("FERTILIZER", 0) > 0:
        # Assuming FERTILIZER might be in shed or inventory. Just queuing it.
        pass
    enqueue_tasks(state.get_care_and_fertilize_tiles())
    
    # Priority 5: Planting/Building
    if day <= 25 and unassigned_workers:
        available_seeds = []
        for seed_type, qty in seeds.items():
            if seed_type != "FERTILIZER":
                available_seeds.extend([seed_type] * qty)
                
        empty_tiles = [
            (x, y) for x, y in state.get_empty_tiles() 
            if (x, y) not in queued_positions
        ]

        for worker_id in list(unassigned_workers.keys()):
            if not available_seeds or not empty_tiles:
                break
                
            worker_pos = unassigned_workers[worker_id]
            # Find the unassigned empty tile nearest to this specific worker
            closest_tile = min(empty_tiles, key=lambda pos: manhattan_distance(worker_pos, pos))
            
            seed_to_plant = available_seeds.pop(0)
            empty_tiles.remove(closest_tile)
            queued_positions.add(closest_tile)
            
            if list(closest_tile) == worker_pos:
                assign_worker(worker_id, ["PLANT", seed_to_plant])
            else:
                move = move_towards(worker_pos, closest_tile)
                if move:
                    assign_worker(worker_id, [move])
                else:
                    assign_worker(worker_id, ["PASS"])

    # Priority 6: Logistics (Shed)
    # If unassigned workers exist, send them to the shed to wait/drop off.
    # Shed is at center. Assuming boardSize=10, tiles (4,4), (5,4), (4,5), (5,5) are adjacent.
    shed_positions = [(4,4), (5,4), (4,5), (5,5)]
    # We will just append a generic DROP task at (4,4) for any leftover worker.
    # A robust agent would check inventory capacity.
    tasks.append({"type": "DROP", "pos": [4, 4], "args": []})

    # Assign tasks to the closest unassigned workers
    for task in tasks:
        # For Priority 6, we might want all remaining workers to do it, so we don't break.
        if not unassigned_workers:
            break
            
        closest_id = get_closest_worker(task["pos"], unassigned_workers)
        worker_pos = unassigned_workers[closest_id]
        
        if worker_pos == task["pos"]:
            assign_worker(closest_id, [task["type"]] + task["args"])
        else:
            move = move_towards(worker_pos, task["pos"])
            if move:
                assign_worker(closest_id, [move])
            else:
                assign_worker(closest_id, ["PASS"])
                
    # If any workers are still unassigned after all tasks, make them move to shed (4,4) or PASS
    for worker_id in list(unassigned_workers.keys()):
        worker_pos = unassigned_workers[worker_id]
        if worker_pos != [4, 4]:
            move = move_towards(worker_pos, [4, 4])
            assign_worker(worker_id, [move] if move else ["PASS"])
        else:
            assign_worker(worker_id, ["DROP"])

    return {
        "farmer": worker_actions["farmer"],
        "hands": worker_actions["hands"],
        "market": market
    }

if __name__ == "__main__":
    # Realistic initial observation matching the actual game state
    test_obs = {
        "step": 1,
        "day": 1,
        "hour": 6,
        "player": 0,
        "farms": [
            {
                "money": 3000,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": [0],
                "tiles": [["LOCKED" if (x >= 5 or y >= 5) else None for x in range(10)] for y in range(10)]
            },
            {
                "money": 3000,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": [0],
                "tiles": [["LOCKED" if (x >= 5 or y >= 5) else None for x in range(10)] for y in range(10)]
            }
        ],
        "market": {
            "inventory": {}
        },
        "town": {
            "unlocked_shops": []
        },
        "private": {
            "seeds": {"WHEAT": 5},
            "shed": {}
        }
    }
    
    print("Testing agent with initial state...")
    action = agent(test_obs)
    print("Agent Action:", action)
    
    print("\nTesting agent on mature plant...")
    # Modify the dummy observation to simulate standing on a mature wheat plant
    test_obs["farms"][0]["tiles"][4][4] = {
        "kind": "PLANT",
        "planted_day": -1, # day 1 - (-1) = age 2
        "watered_today": False
    }
    action2 = agent(test_obs)
    print("Agent Action:", action2)
