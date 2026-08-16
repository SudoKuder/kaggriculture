"""
Analyze game replays in detail to understand what happened
"""
import json
from pathlib import Path

def load_replay(filename="replay.json"):
    """Load replay file"""
    with open(filename, "r") as f:
        return json.load(f)

def print_step_details(state, step_num):
    """Print detailed info for a specific step"""
    obs = state[0]['observation']
    
    if obs['step'] != step_num:
        return False
    
    print(f"\n{'='*70}")
    print(f"STEP {step_num} | Day {obs['day']} | Hour {obs['hour']}")
    print(f"{'='*70}")
    
    # Market info
    print("\n📊 MARKET:")
    print(f"  Inventory: {obs['market']['inventory']}")
    print(f"  Prices: {obs['market']['prices']}")
    
    # Town info
    print(f"\n🏘️  TOWN:")
    print(f"  Shops: {obs['town']['unlocked_shops']}")
    
    # Farm info
    for player_idx in [0, 1]:
        farm = obs['farms'][player_idx]
        private = obs['private'] if obs['player'] == player_idx else None
        
        print(f"\n👨‍🌾 PLAYER {player_idx}:")
        print(f"  Money: ${farm['money']:.0f}")
        print(f"  Farmer: {farm['farmer']} | Hands: {farm['hands']}")
        print(f"  Unlocked quadrants: {farm['unlocked_quadrants']}")
        
        if private:
            print(f"  Shed: {private['shed']}")
            print(f"  Seeds: {private['seeds']}")
    
    return True

def print_summary(state):
    """Print game summary"""
    obs = state[0]['observation']
    
    print(f"\n{'='*70}")
    print("GAME SUMMARY")
    print(f"{'='*70}")
    print(f"Total Steps: {obs['step']}")
    print(f"Total Days: {obs['day']}")
    print(f"Season End: {obs['day'] == 29 and obs['hour'] == 23}")
    
    for player_idx in [0, 1]:
        farm = obs['farms'][player_idx]
        print(f"\nPlayer {player_idx}: ${farm['money']:.0f}")
    
    # Winner
    p0_money = obs['farms'][0]['money']
    p1_money = obs['farms'][1]['money']
    
    print(f"\n🏆 RESULT:")
    if p0_money > p1_money:
        print(f"Player 0 WINS by ${p0_money - p1_money:.0f}")
    elif p1_money > p0_money:
        print(f"Player 1 WINS by ${p1_money - p0_money:.0f}")
    else:
        print("TIE!")

def list_steps_with_activity(state, player=None):
    """List steps where something interesting happened"""
    print(f"\n{'='*70}")
    print("INTERESTING STEPS")
    print(f"{'='*70}")
    
    obs = state[0]['observation']
    
    # Track previous state
    prev_money = [3000, 3000]
    prev_market = {}
    
    for step in range(obs['step'] + 1):
        current_obs = state[0]['observation']
        
        # Check for money changes
        for p in [0, 1]:
            if current_obs['farms'][p]['money'] != prev_money[p]:
                print(f"Step {step}: Player {p} money changed: "
                      f"${prev_money[p]:.0f} → ${current_obs['farms'][p]['money']:.0f}")
                prev_money[p] = current_obs['farms'][p]['money']
    
    print("\n(Note: Full replay analysis requires detailed state tracking)")

if __name__ == "__main__":
    print("Replay Analysis Tool")
    print("Usage:")
    print("  from analyze_replay import load_replay, print_step_details, print_summary")
    print("  state = load_replay('replay.json')")
    print("  print_summary(state)")
    print("  print_step_details(state, 50)  # Step 50")
