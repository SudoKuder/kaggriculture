# Kaggriculture Agent Architecture

The agent is built using a **hybrid architecture** that combines a hard-coded, rule-based tactical layer with a neural network-driven strategic layer. 

Instead of having a neural network control every single micro-action (which is hard to train and prone to errors), the network makes high-level, once-a-day strategic decisions. The rule-based system then executes those decisions flawlessly.

Here is an exhaustive breakdown of every component, rule, and network in the agent.

---

## 1. The Strategic Layer (Neural Network)

The strategic layer acts as the "manager" of the farm. It is a small **Neural Network (Multi-Layer Perceptron)** that fires exactly **once per day** (at hour 0). 

### What is the Network?
- **Architecture:** It is a 4-layer feed-forward network (64 -> 64 -> 32 -> 1).
- **Activation Functions:** ReLU (Rectified Linear Unit) for hidden layers, and linear for the final output.
- **Input:** A 64-dimensional feature vector representing the current state of the game (e.g., current money, day, number of animals, planted crops, unlocked land) *combined* with a specific proposed action plan.
- **Output:** A single scalar value representing the **predicted end-of-game money advantage** (your money minus the opponent's money) if the given plan is executed.
- **Inference:** While trained in PyTorch, the weights are exported to raw NumPy arrays (`strategy_weights_inline.py`). This allows the agent to run natively in the submission without cold-start penalties or heavy PyTorch dependencies.

### How it Works (Candidate Evaluation)
Instead of generating raw actions, the network evaluates a set of human-defined **Candidate Plans**.
1. At the start of a day, the rules engine generates a list of valid strategic plans.
2. The network scores every single plan by predicting the final game outcome.
3. The agent selects the plan with the highest predicted score.

---

## 2. The Candidate Plans (Strategic Directives)

The candidate plans are the bridge between the neural network and the rules. They are defined in `strategy/candidates.py`. These plans act as **overrides** to the default heuristic rules. The network evaluates up to 10 distinct options depending on the game state.

Here is an **in-depth breakdown of every Candidate Plan rule**:

1. **Baseline (`baseline`)**
   - **Trigger:** Always available.
   - **Effect:** Makes no changes to the heuristic. Allows the default tactical market rules to govern hiring, buying land, and buying seeds based on standard ROI calculations.

2. **Aggressive Land (`aggressive_land`)**
   - **Trigger:** Available if you have fewer than 4 quadrants unlocked, it is on or before day 24, and you have enough money to buy the land plus a 200 buffer.
   - **Effect:** Forces the tactical layer to instantly buy a new quadrant.

3. **Hold Premium Goods (`hold_premium`)**
   - **Trigger:** Available if there are premium goods (Melon, Strawberry, Milk, Wool) in the shed, and the day is strictly before day 27.
   - **Effect:** Instructs the tactical market rule to *exclude* premium goods from the sell list. They stay in the shed to wait for better market prices.

4. **Sell All Now (`sell_all_now`)**
   - **Trigger:** Available if there are premium goods in the shed and it is strictly before day 28.
   - **Effect:** Forces the tactical market rule to sell all premium goods immediately. Overrides any wait-and-see behavior, which is useful when the network spots a price peak.

5. **Animal Rush (`animal_rush_X`)**
   - **Trigger:** Available if you have fewer than 6 animals in total, it is on or before day 18, and you can afford at least 1 goose (300) plus a 200 buffer.
   - **Effect:** Forces the purchase of up to 2 geese at once (up to the max cap of 6 total animals). Geese provide steady, daily passive income.

6. **Melon Pivot (`melon_pivot_X`)**
   - **Trigger:** Available strictly in the middle of the season (days 14 through 19) if you can afford at least one melon seed (80) plus a 200 buffer.
   - **Effect:** Overrides standard dynamic seed buying. Instantly attempts to purchase enough Melon seeds to fill every empty tile on the farm (budget permitting).

7. **Conservative (`conservative`)**
   - **Trigger:** Always available from day 5 onwards.
   - **Effect:** Defensive mode. Explicitly skips buying land, skips buying any seeds, skips buying animals, and restricts the hiring target to exactly 1 (meaning it only uses the main farmer, and will fire any extra hands by refusing to re-hire them).

8. **Hire Heavy (`hire_heavy`)**
   - **Trigger:** Available on or before day 20, if you can afford the basic worker hire cost (15) plus a 100 buffer.
   - **Effect:** Overrides the dynamic worker calculation and forces the agent to try and maintain 5 workers (farmer + 4 hands) on the farm.

9. **Cow Invest (`cow_invest`)**
   - **Trigger:** Available early in the game (on or before day 14) so the cow has time to pay off its cost, provided you have fewer than 6 animals and can afford the cow (400) plus a 300 buffer.
   - **Effect:** Forces the purchase of exactly 1 cow.

10. **Wheat Economy (`wheat_economy_X`)**
    - **Trigger:** Available on or before day 25, if there are more than 5 empty tiles on the farm and you have fewer than 5 seeds in the shed.
    - **Effect:** Fills the empty tiles with fast-cycling Wheat. Overrides seed purchases to buy wheat at 10 apiece (maintaining a 100 buffer).

*(Note: If fewer than 2 candidates are generated based on triggers, a dummy `noop_alternative` is added to ensure the value network has something to compare against).*

---

## 3. The Tactical Layer (Rules & Heuristics)

The tactical layer (`main.py`) acts as the "workers" on the farm. It is **100% rule-based**. It takes the high-level plan selected by the neural network and executes it.

### Market Rules (`build_market`)
Every turn, the agent checks the market and issues up to 10 orders. 
- **Selling Rules:** 
  - Iterates through a prioritized list: Fertilizer, Carrot, Tomato, Egg, Wheat, Strawberry, Melon, Milk, Wool.
  - Keeps a strict reserve of 3 Fertilizer (to boost crop yields) and exactly enough Wheat to feed all living animals that day. 
  - On the very last day (Day 29, Hour 23), it performs a panic liquidation, dumping everything in the workers' inventories and the shed, as leftover items score zero.
- **Buying Feed:** If wheat drops below the daily feed reserve, it buys exactly what it needs from the market (up to 15 per turn). This is an emergency rule and can happen at any hour to prevent animal starvation.
- **Buying Land:** Buys land if money exceeds the land cost plus a 500 buffer. It will keep buying land even late into the season (up to day 24) since fast crops like wheat can still yield a return. (Overridden if the plan says otherwise).
- **Hiring Workers:** *Gated to hour 0.* If no plan dictates the target, it dynamically calculates needed workers. It sums up all tasks (plants needing water, animals needing care/feed, harvestable items, empty tiles to plant, weeds to dig) multiplied by 2 (1 turn to move, 1 to act). It divides this by 24 hours to estimate how many hands are needed, and hires them.
- **Buying Seed:** *Gated to hour 0 (Except Melons).* If no plan overrides this, it evaluates the dynamic ROI of all crops. It calculates `(market_price * max_yield - seed_cost) / time_to_max`. It only buys seeds if there is enough time left in the 30-day season for the crop to fully mature. *Important Rule:* Melons can strictly only be bought at `hour == 0` to prevent sudden market-drain if melon prices spike mid-day.
- **Buying Animals:** *Gated to hour 0.* If not overridden, it passively attempts to buy 1 Goose between days 8 and 18 if money permits.

### Worker Assignment Rules (`assign_tier`)
Workers are assigned tasks based on Manhattan distance to the target. If they need an item (like a seed or fertilizer) and don't have it, the distance calculation routes them to the central Shed first.

Tasks are grouped into strict Priority Tiers to ensure critical tasks aren't starved by backlog:
1. **Tier P1 (Emergencies):** Plants or animals that have consecutive unwatered/unfed streaks of >= 1. These will die soon, so they are addressed immediately.
2. **Dedicated Animal Tier:** To prevent large crop fields from starving animals of attention, workers are reserved specifically for animal tasks. This handles:
   - Placing animals from the shed into coops/pastures.
   - Building coops (for geese) and pastures (for cows/sheep) if there's a deficit.
   - Feeding (if unfed today).
   - Caring (if uncared today, doubles yield).
   - Collecting fertilizer.
3. **Tier P2 (Harvesting):** Harvesting fully mature one-time crops, ongoing crops with accumulated yield, and harvestable animals.
4. **Tier P4 (Bonus Watering):** Watering crops that are in their peak yield window (or ongoing crops). This increases the yield.
5. **Tier P4b (Maintenance Watering):** *Only executed after hour 18.* Waters crops outside their bonus window just to keep them alive. Delayed to late in the day so it doesn't block harvesting.
6. **Tier P5 (Fertilizing):** Spreads fertilizer on crops entering their bonus window.
7. **Inventory Drop:** If a worker's inventory exceeds 15 items, they head to the shed to drop them off. On Day 29, they drop items rapidly to ensure they can be sold before time runs out.
8. **Tier P3 & P7 (Weeds & Planting):** Lowest priority. Workers move to the nearest empty tile (or weed, which carries a +1 turn distance penalty to dig) and plant available seeds.

---

## 4. How the Agent is Trained

The training process (`strategy/train.py`) uses **Self-Play Monte Carlo Value Network Training**.

### 1. Opponent Pool
The agent trains by playing thousands of simulated games against a pool of diverse opponents. This pool includes:
- The completely random starter agent.
- The pure heuristic (rule-based) agent without a neural network.
- **Past Snapshots:** Older versions of the neural network itself, forcing it to continuously adapt to better strategies.

### 2. Experience Collection
During a training game, at the start of every day (hour 0), the agent looks at the generated candidate plans.
- **Epsilon-Greedy Exploration:** With a probability of `epsilon` (which decays over time from 50% to 5%), the agent picks a *random* candidate plan to try out new strategies. Otherwise, it picks the plan the network currently thinks is best.
- It records its current state and the plan it chose.

### 3. Reward and Loss Calculation
The neural network is entirely agnostic to intermediate rewards. It only cares about **winning the game**.
- The game runs to completion.
- The agent looks at the final money score: `Money Delta = (My Final Money) - (Opponent's Final Money)`.
- It goes back to every single state-plan combo recorded during that game and assigns them all this final `Money Delta` as the target label.
- A mini-batch of these experiences is sampled, and the neural network weights are updated using **Mean Squared Error (MSE)** loss to make its predictions closer to the actual outcome.

### 4. Continuous Iteration
Every 100 episodes, the training script saves the current weights and adds this new version of the agent into the opponent pool, creating an automated curriculum where the agent must constantly learn to beat stronger versions of itself.
