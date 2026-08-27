"""Multi-seed benchmark: strategic agent vs heuristic agent."""
from kaggle_environments import make
import main

import types
import sys

with open("main.py", "r", encoding="utf-8") as f:
    main_code = f.read()

heuristic_module = types.ModuleType("heuristic_main")
exec(main_code, heuristic_module.__dict__)
heuristic_module._strategic_layer = None
# Force step_counter past 1 so it never tries to initialize the strategic layer
heuristic_module.step_counter = 2

def heuristic_agent(obs):
    return heuristic_module.agent(obs)

def main_benchmark():
    wins = 0
    total = 0
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        env = make("kaggriculture", configuration={"seed": seed})
        # P0: Strategic Layer, P1: Heuristic
        env.run([main.agent, heuristic_agent])
        obs = env.state[0]["observation"]
        m0 = obs["farms"][0]["money"]
        m1 = obs["farms"][1]["money"]
        total += 1
        if m0 > m1:
            wins += 1
        result = "WIN" if m0 > m1 else ("LOSS" if m1 > m0 else "TIE")
        print(f"seed={seed}: P0 (Strategic)=${m0:.0f} P1 (Heuristic)=${m1:.0f} {result}")
    print(f"\nWins: {wins}/{total}")

if __name__ == "__main__":
    main_benchmark()