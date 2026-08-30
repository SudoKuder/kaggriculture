import json
import sys

def analyze(filepath):
    print(f"Loading {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    
    for player_id in [0, 1]:
        print(f"\n--- Player {player_id} ---")
        
        weed_counts = []
        wheat_counts = []
        hand_counts = []
        
        unique_tiles = set()
        
        prev_hands = 0
        hands_left_events = []
        
        for step_num, step in enumerate(steps):
            obs = step[0]['observation']
            
            if obs['step'] == 0:
                continue
                
            farm = obs['farms'][player_id]
            tiles = farm['tiles']
            
            flat_tiles = []
            for row in tiles:
                for tile in row:
                    flat_tiles.append(tile)
                    if isinstance(tile, dict):
                        # Convert dict to a tuple of items so it can be added to the set
                        unique_tiles.add(tuple(sorted(tile.items())))
                    elif tile is not None:
                        unique_tiles.add(tile)
            
            weed_count = sum(1 for t in flat_tiles if isinstance(t, dict) and t.get('type') == 'WEED' or t == 'WEED')
            wheat_count = sum(1 for t in flat_tiles if isinstance(t, dict) and 'WHEAT' in t.get('type', '') or (isinstance(t, str) and 'WHEAT' in t))
            
            weed_counts.append(weed_count)
            wheat_counts.append(wheat_count)
            
            num_hands = len(farm['hands'])
            hand_counts.append(num_hands)
            
            if step_num > 1 and num_hands < prev_hands:
                hands_left_events.append((obs['step'], obs['day'], obs['hour'], prev_hands - num_hands))
            prev_hands = num_hands
                
        print(f"Unique tile types observed:")
        for ut in unique_tiles:
            print("  ", ut)
        
        if weed_counts:
            print(f"Max weeds: {max(weed_counts)} at step {weed_counts.index(max(weed_counts)) + 1}")
        if wheat_counts:
            print(f"Max wheat: {max(wheat_counts)} at step {wheat_counts.index(max(wheat_counts)) + 1}")
            
        if hands_left_events:
            print(f"Hands left events (step, day, hour, count):")
            for e in hands_left_events:
                print(f"  {e}")
        else:
            print("No hands left events.")
        
if __name__ == "__main__":
    analyze("102681618.json")
