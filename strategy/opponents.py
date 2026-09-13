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
    """Manages a pool of opponent agents for training.

    Self-play snapshots are capped at ``max_snapshots`` to prevent an
    ever-growing pool of stale opponents from diluting training signal.
    When the cap is exceeded, the oldest snapshot is evicted.  Sampling
    is weighted so that more recent snapshots are selected more often.

    ``fixed_opponent_fraction`` controls what share of ``sample()`` calls
    draw from the base (non-self-play) agents.  Setting this to 0.7 means
    70 % of games are played against heuristic / noisy / passive opponents,
    preventing the reward signal from collapsing to zero when the self-play
    pool grows large.
    """

    def __init__(self, include_starter=True, include_heuristic=True,
                 include_passive=True, noise_variants=2,
                 max_snapshots=5, fixed_opponent_fraction=0.7):
        self._base_agents = []
        self._base_labels = []
        self._snapshots = []       # list of (agent, label)
        self._max_snapshots = max_snapshots
        self._fixed_frac = fixed_opponent_fraction

        if include_starter:
            self._base_agents.append(_make_starter_agent())
            self._base_labels.append("starter")

        if include_heuristic:
            heur = _make_heuristic_agent()
            self._base_agents.append(heur)
            self._base_labels.append("heuristic")

            # Add noisy variants
            for i in range(noise_variants):
                rate = 0.1 + 0.1 * i  # 0.1, 0.2, ...
                self._base_agents.append(_make_noisy_agent(heur, noise_rate=rate))
                self._base_labels.append(f"noisy_{rate:.1f}")

        if include_passive:
            self._base_agents.append(_make_passive_agent())
            self._base_labels.append("passive")

    def add_snapshot(self, agent_fn, label="snapshot"):
        """Add a past agent version as a training opponent.

        If adding this snapshot would exceed ``max_snapshots``, the oldest
        snapshot is evicted first.
        """
        self._snapshots.append((agent_fn, label))
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def sample(self):
        """Sample a random opponent from the pool.

        Fixed (base) agents are sampled with probability
        ``fixed_opponent_fraction`` (default 0.7).  When snapshots exist and
        the roll falls into the self-play bucket, more recent snapshots are
        preferred (linearly increasing weights).

        Returns:
            (agent, label): An agent (function or string) and its label.
        """
        has_snapshots = len(self._snapshots) > 0
        # Always use fixed opponents when no snapshots yet.
        # Otherwise respect the configured fraction.
        use_fixed = (not has_snapshots) or (random.random() < self._fixed_frac)
        if use_fixed:
            idx = random.randrange(len(self._base_agents))
            return self._base_agents[idx], self._base_labels[idx]
        else:
            # Weight toward recent snapshots: weight[i] = i + 1
            n = len(self._snapshots)
            weights = list(range(1, n + 1))
            total = sum(weights)
            r = random.random() * total
            cumulative = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    return self._snapshots[idx]
            return self._snapshots[-1]  # fallback

    def all_agents(self):
        """Return all (agent, label) pairs."""
        pairs = list(zip(self._base_agents, self._base_labels))
        pairs.extend(self._snapshots)
        return pairs

    def __len__(self):
        return len(self._base_agents) + len(self._snapshots)

