"""
Real-time game logger - Records every agent decision as the game runs
"""
import json
import os
import glob
from kaggle_environments import make

# Global action log
action_log = []

def logged_agent(obs, agent_func):
    """Wrapper to log agent actions"""
    action = agent_func(obs)
    step = obs['step']
    day = obs['day']
    hour = obs['hour']
    player = obs['player']
    
    # Log the action with context
    log_entry = {
        'step': step,
        'day': day,
        'hour': hour,
        'player': player,
        'farmer_pos': obs['farms'][player]['farmer'],
        'money_before': obs['farms'][player]['money'],
        'farmer_action': action.get('farmer', ['PASS']),
        'market_orders': action.get('market', []),
        'tiles': obs['farms'][player]['tiles'],
    }
    
    action_log.append(log_entry)
    
    return action

def record_game_with_logging(seed=42):
    """Run game and log all actions"""
    global action_log
    action_log = []
    
    from agent import agent as player_agent
    
    # Create wrappers
    def agent_0(obs):
        return logged_agent(obs, player_agent)
    
    def agent_1(obs):
        return logged_agent(obs, player_agent)
    
    print("Running game with detailed action logging...")
    env = make("kaggriculture", configuration={"seed": seed})
    env.run([agent_0, agent_1])
    
    # Get final state
    obs = env.state[0]['observation']
    
    # Save log to JSON
    os.makedirs("game_logs", exist_ok=True)
    existing_logs = glob.glob("game_logs/game_log*.json")
    max_num = 0
    for log_file in existing_logs:
        try:
            basename = os.path.basename(log_file)
            num_str = basename.replace("game_log", "").replace(".json", "")
            if num_str.isdigit():
                max_num = max(max_num, int(num_str))
        except Exception:
            pass
    
    next_num = max_num + 1
    log_filename = f"game_logs/game_log{next_num}.json"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump({
            'action_log': action_log,
            'final_state': {
                'p0_money': obs['farms'][0]['money'],
                'p1_money': obs['farms'][1]['money'],
                'total_turns': obs['step']
            }
        }, f, indent=2, default=str)
    
    print(f"[OK] Recorded {len(action_log)} actions to {log_filename}")
    return env, action_log, obs

if __name__ == "__main__":
    env, log, obs = record_game_with_logging()
    
    print("\n[SUMMARY] Action Summary:")
    p0_actions = [a for a in log if a['player'] == 0]
    p1_actions = [a for a in log if a['player'] == 1]
    
    print(f"Player 0: {len(p0_actions)} actions | Final: ${obs['farms'][0]['money']:.0f}")
    print(f"Player 1: {len(p1_actions)} actions | Final: ${obs['farms'][1]['money']:.0f}")
    
    # Show some sample actions
    print("\nFirst 10 actions:")
    for action in log[:10]:
        print(f"  Step {action['step']:3d} | P{action['player']} | "
              f"Farmer: {action['farmer_action']} | Market: {action['market_orders']}")
