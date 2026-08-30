import json
import sys

def analyze(filepath):
    print(f"Loading {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    
    obs = steps[1][0]['observation']
    farm = obs['farms'][0]
    tiles = farm['tiles']
    
    print("Type of tiles:", type(tiles))
    if len(tiles) > 0:
        print("Type of tile[0]:", type(tiles[0]))
        print("Tile[0]:", tiles[0])
        
if __name__ == "__main__":
    analyze("102681618.json")
