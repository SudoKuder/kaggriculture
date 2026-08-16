from kaggle_environments import make

# 1. Create the Kaggriculture environment
env = make("kaggriculture", configuration={"seed": 42}, debug=True)

# 2. Reset the environment to inspect initial state
obs = env.reset()

# Print observation structure (farms, market prices, etc.)
print("Initial observation keys:", obs[0]['observation'].keys())
print("Player 0 Farm Info:", obs[0]['observation']['farms'][0])

