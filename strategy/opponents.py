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


def _make_top_player_scripted():
    """Opponent modeled on top-player replay analysis (Majkel1337).

    Fixed macro-strategy:
    - Day 7: buy NE quadrant
    - Day 10: buy SW quadrant
    - Never buy SE (4th quadrant)
    - Diversify all 5 crop types
    - Target ~16 animals
    - Hire aggressively (target 5 workers)
    - Sell everything every turn (diversified)

    Uses the heuristic agent as the tactical base but forces specific
    strategic decisions via the plan dict.
    """
    import main as _main

    _CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

    def top_player_scripted(obs):
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        player_id = obs["player"]
        me = obs["farms"][player_id]
        money = me["money"]
        n_quads = len(me.get("unlocked_quadrants", ["NW"]))

        plan = {}

        # Land expansion: day 7 → NE, day 10 → SW, never 4th
        if n_quads == 1 and day >= 7 and money >= 1000:
            plan["buy_land"] = True
        elif n_quads == 2 and day >= 10 and money >= 2000:
            plan["buy_land"] = True
        else:
            plan["buy_land"] = False

        # Hire aggressively: target 5 workers
        plan["hire_target"] = 5

        # Sell everything (no holding)
        plan["sell_hold"] = {p: "sell" for p in
                            ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "EGG", "MILK", "WOOL", "FERTILIZER"]}

        # Diversified crop buying: cycle through all 5 types
        from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
        crop = _CROP_ORDER[day % 5]
        seed_price = CROPS[crop]["seed"]
        # Spend ~30% of money on seeds
        qty = int((money * 0.3) / seed_price) if seed_price > 0 else 0
        if qty > 0 and day < 20:  # stop buying seeds late
            plan["buy_seed"] = {"crop": crop, "qty": qty}
        else:
            plan["buy_seed"] = None

        # Animal buying: target ~16 total, mix types
        from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ANIM_DATA
        total_animals = sum(1 for row in me.get("tiles", []) for t in row
                          if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")
                          and "animal" in t)
        if total_animals < 16 and day >= 3 and money >= 300:
            anim_types = ["GOOSE", "COW", "SHEEP"]
            anim = anim_types[day % 3]
            anim_price = ANIM_DATA[anim]["cost"]
            qty_anim = min(2, int(money * 0.15 / anim_price)) if anim_price > 0 else 0
            if qty_anim > 0:
                plan["buy_animal"] = {"type": anim, "qty": qty_anim}
            else:
                plan["buy_animal"] = None
        else:
            plan["buy_animal"] = None

        plan["label"] = "top_player_script"

        # Inject the plan and run the heuristic
        _main.IS_TRAINING = True
        _main._current_plan[player_id] = plan
        return _main.agent(obs)

    return top_player_scripted


class OpponentPool:
    """Manages a pool of opponent agents for training.

    Self-play snapshots are capped at ``max_snapshots`` to prevent an
    ever-growing pool of stale opponents from diluting training signal.
    When the cap is exceeded, the oldest *non-milestone* snapshot is
    evicted.  Sampling weights snapshots by how close their win rate is
    to 50% -- i.e., whoever currently challenges the policy most, not
    just whoever is newest.

    ``fixed_opponent_fraction`` controls what share of ``sample()`` calls
    draw from the base (non-self-play) agents.  Setting this to 0.7 means
    70 % of games are played against heuristic / noisy / passive opponents,
    preventing the reward signal from collapsing to zero when the self-play
    pool grows large.
    """

    class _SnapshotEntry:
        """Tracks an opponent snapshot with rolling win-rate statistics."""
        __slots__ = ("agent", "label", "wins", "games", "is_milestone")

        def __init__(self, agent, label, is_milestone=False):
            self.agent = agent
            self.label = label
            self.wins = 0       # wins by the *current policy* against this snapshot
            self.games = 0
            self.is_milestone = is_milestone

        @property
        def win_rate(self):
            if self.games == 0:
                return 0.5  # assume 50% until proven otherwise
            return self.wins / self.games

        @property
        def challenge_score(self):
            """How close this opponent's win rate is to 50%.
            Lower = more challenging = higher sampling priority.
            Snapshots with < 2 games get score 0 (max priority) to
            force at least a couple of trials before deprioritizing.
            """
            if self.games < 2:
                return 0.0
            return abs(self.win_rate - 0.5)

    def __init__(self, include_starter=True, include_heuristic=True,
                 include_passive=True, noise_variants=2,
                 max_snapshots=5, fixed_opponent_fraction=0.7):
        self._base_agents = []
        self._base_labels = []
        self._snapshots = []       # list of _SnapshotEntry
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

        # Strategically distinct opponent modeled on top-player replays
        self._base_agents.append(_make_top_player_scripted())
        self._base_labels.append("top_player_script")

    def add_snapshot(self, agent_fn, label="snapshot"):
        """Add a past agent version as a training opponent.

        The first snapshot is always marked as a milestone (never evicted).
        When the pool is at half capacity, the middle snapshot is also
        promoted to milestone.  Non-milestone snapshots are evicted
        oldest-first when the cap is exceeded.
        """
        is_milestone = (len(self._snapshots) == 0)  # first snapshot = milestone

        entry = self._SnapshotEntry(agent_fn, label, is_milestone=is_milestone)
        self._snapshots.append(entry)

        # Promote middle snapshot to milestone when we hit half capacity
        if len(self._snapshots) == self._max_snapshots // 2 + 1:
            mid = len(self._snapshots) // 2
            if not self._snapshots[mid].is_milestone:
                self._snapshots[mid].is_milestone = True

        # Evict oldest non-milestone if over capacity
        while len(self._snapshots) > self._max_snapshots:
            evicted = False
            for i, s in enumerate(self._snapshots):
                if not s.is_milestone:
                    self._snapshots.pop(i)
                    evicted = True
                    break
            if not evicted:
                # All are milestones (shouldn't happen) -- evict oldest
                self._snapshots.pop(0)
                break

    def report_result(self, label, won):
        """Update win-rate counters after a game against the given opponent.

        Args:
            label: The opponent label returned by sample().
            won: True if the current policy won.
        """
        for s in self._snapshots:
            if s.label == label:
                s.games += 1
                if won:
                    s.wins += 1
                return
        # Fixed opponents don't need tracking

    def sample(self):
        """Sample a random opponent from the pool.

        Fixed (base) agents are sampled with probability
        ``fixed_opponent_fraction`` (default 0.7).  When snapshots exist and
        the roll falls into the self-play bucket, snapshots are weighted by
        challenge score (closeness to 50% win rate).

        Returns:
            (agent, label): An agent (function or string) and its label.
        """
        has_snapshots = len(self._snapshots) > 0
        use_fixed = (not has_snapshots) or (random.random() < self._fixed_frac)
        if use_fixed:
            idx = random.randrange(len(self._base_agents))
            return self._base_agents[idx], self._base_labels[idx]
        else:
            # Weight by challenge: lower challenge_score = more weight.
            eps = 0.05
            weights = [1.0 / (s.challenge_score + eps) for s in self._snapshots]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    s = self._snapshots[idx]
                    return s.agent, s.label
            s = self._snapshots[-1]
            return s.agent, s.label

    def all_agents(self):
        """Return all (agent, label) pairs."""
        pairs = list(zip(self._base_agents, self._base_labels))
        pairs.extend((s.agent, s.label) for s in self._snapshots)
        return pairs

    def __len__(self):
        return len(self._base_agents) + len(self._snapshots)


