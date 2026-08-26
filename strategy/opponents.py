"""Opponent pool management for training diversity.

Provides a set of opponent agents with different play styles so the
value function doesn't overfit to a narrow self-play equilibrium.
"""

import random
import sys
import os

# Ensure the repo root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_starter_agent():
    """Return the built-in starter agent string for kaggle_environments."""
    return "starter"


def _make_heuristic_agent():
    """Return our own heuristic agent function (imported from main.py)."""
    from main import agent as heuristic_agent
    return heuristic_agent


def _make_noisy_agent(base_agent_fn, noise_rate=0.2):
    """Wrap an agent function with random market-order noise.

    With probability ``noise_rate`` each market order is replaced by a
    random valid alternative (PASS, extra sell, skip buy, etc.).  This
    prevents the value network from memorising the exact opponent
    behaviour.
    """
    def noisy_agent(obs):
        action = base_agent_fn(obs)
        market = action.get("market", [])
        noised = []
        for order in market:
            if random.random() < noise_rate:
                # Replace with a no-op (skip this order)
                continue
            noised.append(order)
        action["market"] = noised
        return action
    return noisy_agent


def _make_passive_agent():
    """An agent that only does basic watering / feeding via the heuristic
    but never buys seeds, land, or animals.  Represents a very
    conservative opponent.
    """
    from main import agent as base

    def passive(obs):
        action = base(obs)
        market = action.get("market", [])
        # Strip out all buy orders
        filtered = [o for o in market if o[0] == "SELL"]
        action["market"] = filtered
        return action
    return passive


class OpponentPool:
    """Manages a pool of opponent agents for training."""

    def __init__(self, include_starter=True, include_heuristic=True,
                 include_passive=True, noise_variants=2):
        self._agents = []
        self._labels = []

        if include_starter:
            self._agents.append(_make_starter_agent())
            self._labels.append("starter")

        if include_heuristic:
            heur = _make_heuristic_agent()
            self._agents.append(heur)
            self._labels.append("heuristic")

            # Add noisy variants
            for i in range(noise_variants):
                rate = 0.1 + 0.1 * i  # 0.1, 0.2, ...
                self._agents.append(_make_noisy_agent(heur, noise_rate=rate))
                self._labels.append(f"noisy_{rate:.1f}")

        if include_passive:
            self._agents.append(_make_passive_agent())
            self._labels.append("passive")

        # Past agent snapshots are added via add_snapshot()
        self._snapshots = []

    def add_snapshot(self, agent_fn, label="snapshot"):
        """Add a past agent version as a training opponent."""
        self._snapshots.append(agent_fn)
        self._labels.append(label)
        self._agents.append(agent_fn)

    def sample(self):
        """Sample a random opponent from the pool.

        Returns:
            (agent, label): An agent (function or string) and its label.
        """
        idx = random.randrange(len(self._agents))
        return self._agents[idx], self._labels[idx]

    def all_agents(self):
        """Return all (agent, label) pairs."""
        return list(zip(self._agents, self._labels))

    def __len__(self):
        return len(self._agents)
