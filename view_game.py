from kaggle_environments import make

env = make("kaggriculture", configuration={"seed": 42})
env.run(["agent.py", "agent.py"])

obs = env.state[0]['observation']
step = obs['step']

# Inspect specific step
print(f"Step {step} - Final State:")
print(f"Player 0 Money: ${obs['farms'][0]['money']}")
print(f"Player 1 Money: ${obs['farms'][1]['money']}")
print(f"Market: {obs['market']}")
print(f"Town Shops: {obs['town']['unlocked_shops']}")