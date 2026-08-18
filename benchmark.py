"""Multi-seed benchmark: improved agent vs starter agent."""
from kaggle_environments import make


def main():
    wins = 0
    total = 0
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        env = make("kaggriculture", configuration={"seed": seed})
        env.run(["agent.py", "starter"])
        obs = env.state[0]["observation"]
        m0 = obs["farms"][0]["money"]
        m1 = obs["farms"][1]["money"]
        total += 1
        if m0 > m1:
            wins += 1
        result = "WIN" if m0 > m1 else ("LOSS" if m1 > m0 else "TIE")
        print(f"seed={seed}: P0=${m0:.0f} P1=${m1:.0f} {result}")
    print(f"\nWins: {wins}/{total}")


if __name__ == "__main__":
    main()