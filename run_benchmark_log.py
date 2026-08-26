"""
Run the trained strategic agent against the heuristic-only agent and log it.
"""
import json
import os
import glob
from kaggle_environments import make
import main

# Global action log
action_log = []

def logged_agent(obs, agent_func, name):
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
        'agent_name': name,
        'farmer_pos': obs['farms'][player]['farmer'],
        'money_before': obs['farms'][player]['money'],
        'farmer_action': action.get('farmer', ['PASS']),
        'market_orders': action.get('market', []),
        'tiles': obs['farms'][player]['tiles'],
    }
    
    action_log.append(log_entry)
    return action

def trained_agent(obs):
    return main.agent(obs)

def heuristic_agent(obs):
    """Run the agent but force the strategic layer off for this player."""
    original_layer = main._strategic_layer
    main._strategic_layer = None
    
    try:
        action = main.agent(obs)
    finally:
        main._strategic_layer = original_layer
        
    return action

def run_benchmark(seed=42):
    """Run game and log all actions"""
    global action_log
    action_log = []
    
    # Create wrappers
    def agent_0(obs):
        return logged_agent(obs, trained_agent, "TrainedStrategic")
    
    def agent_1(obs):
        return logged_agent(obs, heuristic_agent, "HeuristicOnly")
    
    print(f"Running benchmark (Trained vs Heuristic) with seed {seed}...")
    env = make("kaggriculture", configuration={"seed": seed})
    env.run([agent_0, agent_1])
    
    # Get final state
    obs = env.state[0]['observation']
    
    # Save log to JSON
    os.makedirs("game_logs", exist_ok=True)
    existing_logs = glob.glob("game_logs/benchmark_log*.json")
    max_num = 0
    for log_file in existing_logs:
        try:
            basename = os.path.basename(log_file)
            num_str = basename.replace("benchmark_log", "").replace(".json", "")
            if num_str.isdigit():
                max_num = max(max_num, int(num_str))
        except Exception:
            pass
    
    next_num = max_num + 1
    log_filename = f"game_logs/benchmark_log{next_num}.json"
    
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Game seed")
    args = parser.parse_args()
    
    env, log, obs = run_benchmark(seed=args.seed)
    
    print("\n[SUMMARY] Match Results:")
    p0_actions = [a for a in log if a['player'] == 0]
    p1_actions = [a for a in log if a['player'] == 1]
    
    m0 = obs['farms'][0]['money']
    m1 = obs['farms'][1]['money']
    
    print(f"Player 0 (Trained): {len(p0_actions)} actions | Final: ${m0:.0f}")
    print(f"Player 1 (Heuristic): {len(p1_actions)} actions | Final: ${m1:.0f}")
    
    if m0 > m1:
        print(f"\n=> Trained Agent WINS by ${m0 - m1:.0f}")
    elif m1 > m0:
        print(f"\n=> Heuristic Agent WINS by ${m1 - m0:.0f}")
    else:
        print("\n=> TIE")
