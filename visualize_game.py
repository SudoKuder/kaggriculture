from kaggle_environments import make
import main
import types
import os

with open("main.py", "r", encoding="utf-8") as f:
    main_code = f.read()

heuristic_module = types.ModuleType("heuristic_main")
exec(main_code, heuristic_module.__dict__)
heuristic_module._strategic_layer = None
heuristic_module.step_counter = 2

def heuristic_agent(obs):
    return heuristic_module.agent(obs)

def visualize():
    print("Running match...")
    
    # Load specific checkpoint for the main agent
    checkpoint_path = "./actor_net_ep26300.npz"
    if os.path.exists(checkpoint_path):
        main._strategic_layer = main._StrategicLayer.from_npz(checkpoint_path)
        print(f"Loaded {checkpoint_path} for main agent")
    else:
        print(f"WARNING: {checkpoint_path} not found! Falling back to default.")

    env = make("kaggriculture", configuration={"seed": 42})
    env.run([main.agent, heuristic_agent])
    
    print("Rendering HTML...")
    html = env.render(mode="html")
    with open("replay.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved to replay.html. Open this file in your web browser to view the visualization!")

if __name__ == "__main__":
    visualize()
