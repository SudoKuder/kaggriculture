# Lifecycle & Timeline Rules (Macro-Strategy) Implementation

This plan outlines how the agent's logic will be updated to adapt to the 4 game phases based on the current day (1-30).

## User Review Required

> [!WARNING]
> This represents a significant shift in market and planting behavior. Please review the triggers for buying land and liquidating assets to ensure they align with your intended strategy.

## Proposed Changes

### [Component Name: Phase Detection & Market Logic]

#### [MODIFY] [agent.py](file:///c:/codes/kaggriculture/agent.py)
We will introduce phase detection based on `state.day`:
- **Early Game (Days 1-8)**: 
  - **Market**: Buy `WHEAT` and `CARROT` seeds. Buy `BUY_LAND` if `len(get_empty_tiles()) == 0` and we have enough money (costs scale: 1k, 2k, 4k).
- **Mid Game (Days 9-20)**:
  - **Market**: Buy `GOOSE` / `COW` animals. Buy `TOMATO` and `STRAWBERRY` seeds.
- **Late Game (Days 21-25)**:
  - **Market**: Stop animals/strawberries. Buy `MELON` and fast crops (`WHEAT`, `CARROT`).
- **End Game (Days 26-30)**:
  - **Market**: Sell all items (products, seeds, fertilizer) from the shed. Do not buy anything.

### [Component Name: Task Assignment Logic]

#### [MODIFY] [agent.py](file:///c:/codes/kaggriculture/agent.py)
- **Planting Priority (Priority 5)**: Will be updated to dynamically plant seeds that we currently hold in inventory, rather than hardcoding `WHEAT`. In the End Game, this priority will be completely disabled.
- **Animal Placement**: We will add `BUILD_COOP` / `BUILD_PASTURE` and `PLACE` animal logic for the mid-game when we buy animals.
- **Liquidation**: In the End Game, the agent will prioritize `SELL` orders in the market phase for all non-seed inventory items.

## Open Questions

> [!IMPORTANT]
> 1. For land expansion in the Early Game, should we strictly buy land when `empty_tiles == 0`, or should we reserve a specific cash buffer?
> 2. How many animals/seeds should we buy per turn in the Mid Game? Should we buy up to a certain capacity or just spend all available money?

## Verification Plan

### Automated Tests
- Run `python simulate.py` to ensure the agent correctly shifts its purchasing behavior as the days progress, without crashing.
- Verify through logs that `SELL` actions are triggered aggressively during days 26-30.
