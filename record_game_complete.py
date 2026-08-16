"""
Records complete game state (board + actions) at each step for visual replay
"""
import json
from kaggle_environments import make

game_states = []

def record_full_game_state(seed=42):
    """Run game and record board state + actions at each step"""
    
    from agent import agent as player_agent
    
    global game_states
    game_states = []
    
    action_log_full = []
    
    def recording_agent_wrapper(obs, agent_func):
        """Wrapper to capture both actions and game state"""
        action = agent_func(obs)
        step = obs['step']
        
        # Store complete game state
        state_snapshot = {
            'step': step,
            'day': obs['day'],
            'hour': obs['hour'],
            'player': obs['player'],
            'p0_money': obs['farms'][0]['money'],
            'p1_money': obs['farms'][1]['money'],
            'p0_board': obs['farms'][0]['tiles'],
            'p1_board': obs['farms'][1]['tiles'],
            'p0_farmer': obs['farms'][0]['farmer'],
            'p1_farmer': obs['farms'][1]['farmer'],
            'p0_hands': obs['farms'][0]['hands'],
            'p1_hands': obs['farms'][1]['hands'],
            'market_inventory': obs['market']['inventory'],
            'market_prices': obs['market']['prices'],
            'shops': obs['town']['unlocked_shops'],
        }
        
        # Add private info if this is the player
        if obs['player'] == 0:
            state_snapshot['p0_shed'] = obs['private']['shed']
            state_snapshot['p0_seeds'] = obs['private']['seeds']
        else:
            state_snapshot['p1_shed'] = obs['private']['shed']
            state_snapshot['p1_seeds'] = obs['private']['seeds']
        
        # Record action
        action_snapshot = {
            'step': step,
            'day': day,
            'hour': hour,
            'player': obs['player'],
            'farmer_action': action.get('farmer', ['PASS']),
            'market_orders': action.get('market', []),
        }
        
        action_log_full.append(action_snapshot)
        game_states.append((step, obs['player'], state_snapshot, action_snapshot))
        
        return action
    
    def agent_0(obs):
        return recording_agent_wrapper(obs, player_agent)
    
    def agent_1(obs):
        return recording_agent_wrapper(obs, player_agent)
    
    print("Recording complete game with board states...")
    env = make("kaggriculture", configuration={"seed": seed})
    env.run([agent_0, agent_1])
    
    # Save to JSON
    with open("game_complete.json", "w", encoding="utf-8") as f:
        json.dump({
            'game_states': [
                {
                    'step': s[0],
                    'player': s[1],
                    'state': s[2],
                    'action': s[3]
                }
                for s in game_states
            ]
        }, f, indent=2, default=str)
    
    print(f"Recorded {len(game_states)} state snapshots to game_complete.json")
    return env, game_states

if __name__ == "__main__":
    env, states = record_full_game_state()
    print(f"\nTotal states: {len(states)}")
    print("Run: python view_game_animated.py")
