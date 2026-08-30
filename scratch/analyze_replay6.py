import json

def analyze(filepath):
    print(f"Loading {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    
    player_id = 0
    print(f"\n--- Player {player_id} ---")
    
    for step_num, step in enumerate(steps):
        obs = step[0]['observation']
        if obs['step'] == 0: continue
            
        farm = obs['farms'][player_id]
        tiles = farm['tiles']
        
        flat_tiles = []
        for row in tiles:
            for tile in row:
                if tile is not None and not isinstance(tile, str):
                    flat_tiles.append(tile)
        
        weed_count = sum(1 for t in flat_tiles if t.get('kind') == 'WEED' or t.get('crop') == 'WEED' or 'WEED' in str(t))
        wheat_count = sum(1 for t in flat_tiles if t.get('crop') == 'WHEAT')
        
        num_hands = len(farm['hands'])
        money = farm['money']
        
        if step_num % 24 == 0 or weed_count > 0 or num_hands < 5:
            # print daily summary or interesting events
            print(f"Step {step_num} (Day {obs['day']} Hr {obs['hour']}): Money: {money}, Hands: {num_hands}, Wheat: {wheat_count}, Weeds: {weed_count}")

        # Check if actions are idle
        actions = step[player_id].get('action', [])
        # Kaggriculture actions are usually strings per agent or list of strings?
        # Let's inspect actions on day 28/29
        if obs['day'] >= 28 and obs['hour'] < 3:
            print(f"  Actions at day {obs['day']} hr {obs['hour']}: {actions}")

if __name__ == "__main__":
    analyze("102681618.json")
