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
    
    # Market logic
    market = []
    wheat_seeds = state.private.get("seeds", {}).get("WHEAT", 0)
    if wheat_seeds == 0 and state.me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
    wheat_in_shed = state.private.get("shed", {}).get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

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
    seeds_to_plant = wheat_seeds
    for x, y in state.get_empty_tiles():
        if seeds_to_plant <= 0:
            break
        if (x, y) not in queued_positions:
            tasks.append({"type": "PLANT", "pos": [x, y], "args": ["WHEAT"]})
            queued_positions.add((x, y))
            seeds_to_plant -= 1

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
