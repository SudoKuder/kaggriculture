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
        idle_farmer_hands = 0
        total_actions = 0
        hands_left_events = []
        
        prev_hands = 0
        
        for step_num, step in enumerate(steps):
            obs = step[0]['observation']
            
            if obs['step'] == 0:
                continue
                
            farm = obs['farms'][player_id]
            tiles = farm['tiles']
            
            weed_count = sum(1 for t in tiles if t['type'] == 'WEED')
            wheat_count = sum(1 for t in tiles if t['type'] == 'WHEAT')
            
            weed_counts.append(weed_count)
            wheat_counts.append(wheat_count)
            
            num_hands = len(farm['hands'])
            hand_counts.append(num_hands)
            
            # Check for hands leaving (hands decreased)
            if step_num > 0 and num_hands < prev_hands:
                hands_left_events.append((obs['day'], obs['hour'], prev_hands - num_hands))
            prev_hands = num_hands
            
            # Check actions (actions are in step[player_id]['action'])
            action = step[player_id].get('action', "")
            if isinstance(action, str):
                # Kaggriculture actions are strings for each unit? Or one string per agent?
                pass
            
            # actually let's see how actions are formatted
            if step_num == 1 and player_id == 0:
                print(f"Example action format: {step[player_id].get('action')}")
                
        print(f"Max weeds: {max(weed_counts)} at step {weed_counts.index(max(weed_counts))}")
        print(f"Max wheat: {max(wheat_counts)} at step {wheat_counts.index(max(wheat_counts))}")
        print(f"Hands left events: {hands_left_events}")
        
if __name__ == "__main__":
    analyze("102681618.json")
