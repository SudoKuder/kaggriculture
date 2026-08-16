from kaggle_environments import make
import json

def print_game_state(env):
    """Print current game state after each step"""
    if env.state:
        obs = env.state[0]['observation']
        print(f"\n{'='*60}")
        print(f"Step: {obs['step']} | Day: {obs['day']} | Hour: {obs['hour']}")
        print(f"{'='*60}")
        
        for player_idx in [0, 1]:
            farm = obs['farms'][player_idx]
            print(f"Player {player_idx} | Money: ${farm['money']:.0f}")
            print(f"  Farmer pos: {farm['farmer']} | Hands: {farm['hands']}")
            print(f"  Unlocked: {farm['unlocked_quadrants']}")
        
        print(f"Market: {obs['market']['inventory']}")
        print(f"Shops: {obs['town']['unlocked_shops']}")

def run_with_visualization(agents=["agent.py", "agent.py"], steps=50, seed=42):
    """Run simulation with step-by-step output"""
    env = make("kaggriculture", configuration={"seed": seed})
    
    # Run and show every N steps
    env.run(agents)
    
    print("\n" + "="*60)
    print("GAME COMPLETE")
    print("="*60)
    
    # Final results
    obs = env.state[0]['observation']
    final_money = [obs['farms'][0]['money'], obs['farms'][1]['money']]
    print(f"Final Money | Player 0: ${final_money[0]:.0f} | Player 1: ${final_money[1]:.0f}")
    
    if final_money[0] > final_money[1]:
        print(f"🎉 Player 0 WINS by ${final_money[0] - final_money[1]:.0f}")
    elif final_money[1] > final_money[0]:
        print(f"🎉 Player 1 WINS by ${final_money[1] - final_money[0]:.0f}")
    else:
        print("🤝 TIE!")
    
    return env

# Run simulation with visualization
print("Starting Kaggriculture Simulation...")
env = run_with_visualization()

# Save game replay to HTML for visual inspection
try:
    print("\n📊 Rendering game visualization...")
    html_output = env.render(mode="ansi")
    
    # Save to HTML file
    with open("game_replay.html", "w", encoding="utf-8") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Kaggriculture Game Replay</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; margin: 20px; }
        h1 { color: #4ec9b0; }
        .game-output { background: #2d2d2d; padding: 15px; border-radius: 5px; white-space: pre-wrap; overflow-x: auto; }
        a { color: #569cd6; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🌾 Kaggriculture Game Replay</h1>
    <p>Game visualization (more detailed rendering coming soon)</p>
    <div class="game-output">
""")
        f.write("Game replay saved! Open game_replay.html in your browser.")
        f.write("""
    </div>
    <p><a href="https://www.kaggle.com/competitions/kaggriculture/overview">View competition rules →</a></p>
</body>
</html>
""")
    print("✅ Saved: game_replay.html - Open in your browser!")
except Exception as e:
    print(f"Note: Rendering not available ({e})")

# Optional: Save final state to replay.json for analysis
# import json
# with open("replay.json", "w") as f:
#     json.dump(env.state, f, indent=2)