import json
import sys

def analyze(filepath):
    print(f"\n======================================")
    print(f"Loading {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    last_obs = steps[-1][0]['observation']
    
    money_0 = last_obs['farms'][0]['money']
    money_1 = last_obs['farms'][1]['money']
    print(f"Final Score -> Player 0: {money_0} | Player 1: {money_1}")
    if money_0 > money_1:
        print("Player 0 won.")
    else:
        print("Player 1 won.")
        
    for player_id in [0, 1]:
        print(f"\n--- Player {player_id} ---")
        
        for step_num, step in enumerate(steps):
            obs = step[0]['observation']
            if obs['step'] == 0: continue
            
            # Print daily summary at end of day (hour 23)
            if obs['hour'] == 23:
                farm = obs['farms'][player_id]
                money = farm['money']
                num_hands = len(farm['hands'])
                
                # Analyze tiles
                flat_tiles = []
                for row in farm['tiles']:
                    for tile in row:
                        if tile is not None and not isinstance(tile, str):
                            flat_tiles.append(tile)
                            
                weed_count = sum(1 for t in flat_tiles if t.get('kind') == 'WEED')
                plant_count = sum(1 for t in flat_tiles if t.get('kind') == 'PLANT')
                coop_count = sum(1 for t in flat_tiles if t.get('kind') == 'COOP')
                pasture_count = sum(1 for t in flat_tiles if t.get('kind') == 'PASTURE')
                
                # Check for animals in coops/pastures
                animals = sum(1 for t in flat_tiles if t.get('kind') in ['COOP', 'PASTURE'] and t.get('animal'))
                
                print(f"Day {obs['day']:2d} end: Money: {money:6.1f} | Hands: {num_hands:2d} | Plants: {plant_count:2d} | Weeds: {weed_count:2d} | Animals: {animals:2d} (Coops: {coop_count}, Pastures: {pasture_count})")
                
                # Let's also track what crops they have planted
                crop_types = {}
                for t in flat_tiles:
                    if t.get('kind') == 'PLANT':
                        c = t.get('crop')
                        crop_types[c] = crop_types.get(c, 0) + 1
                if crop_types:
                    print(f"    Crops: {crop_types}")

if __name__ == "__main__":
    analyze("102683526.json")
    analyze("102685747.json")
