import json
import sys

def analyze(filepath):
    print(f"Loading {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("Keys in JSON:", list(data.keys()))
    if 'steps' in data:
        steps = data['steps']
        print(f"Number of steps: {len(steps)}")
        
        # Look at the first step
        first_step = steps[0]
        print("First step length (number of agents):", len(first_step))
        print("First step structure:", type(first_step[0]))
        if isinstance(first_step[0], dict):
            print("Keys in step 0 player 0:", list(first_step[0].keys()))
            if 'observation' in first_step[0]:
                obs = first_step[0]['observation']
                print("Observation keys:", list(obs.keys()))
                if 'farms' in obs:
                    print("Farm keys:", list(obs['farms'][0].keys()))
                
                # Check for farmers / units
                print("Farmers/Units representation:")
                for k in obs.keys():
                    if k not in ['farms', 'step', 'market', 'town', 'grid']:
                        print(f"  {k}: {type(obs[k])}")
                
if __name__ == "__main__":
    analyze("102681618.json")
