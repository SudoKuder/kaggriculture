"""Full-game verification: run the fixed agent against the starter agent and self-play."""
from kaggle_environments import make
import sys


def run_single(agent_a, agent_b, seed=42, label=""):
    env = make("kaggriculture", configuration={"seed": seed})
    try:
        env.run([agent_a, agent_b])
    except Exception as e:
        print(f"[{label}] EXCEPTION: {type(e).__name__}: {e}")
        return None
    obs = env.state[0]["observation"]
    m0 = obs["farms"][0]["money"]
    m1 = obs["farms"][1]["money"]
    print(f"[{label}] Final money: P0=${m0:.0f}  P1=${m1:.0f}  winner={'P0' if m0 > m1 else 'P1' if m1 > m0 else 'TIE'}")
    return env


def main():
    # Fixed agent vs starter agent
    run_single("agent.py", "starter", seed=42, label="agent-vs-starter")

    # Fixed agent vs pass agent
    run_single("agent.py", "pass", seed=7, label="agent-vs-pass")

    # Self-play (fixed vs fixed)
    run_single("agent.py", "agent.py", seed=99, label="self-play")

    print("\nAll simulations completed.")


if __name__ == "__main__":
    main()