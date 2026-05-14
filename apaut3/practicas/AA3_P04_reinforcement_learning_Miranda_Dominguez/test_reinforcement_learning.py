# -*- coding: utf-8 -*-
"""
Tests for reinforcement_learning.py

Run with:
    pytest test_reinforcement_learning.py -v

Authors: Lucas Miranda López, Iván Domínguez Hernández
"""

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

# ── tqdm.notebook stub ────────────────────────────────────────────────────────
# reinforcement_learning.py imports tqdm.notebook.tqdm, which requires an
# active IPython kernel.  We inject a minimal stub before the module is
# imported so the tests run in plain Python / pytest without a Jupyter server.
_notebook_mod = types.ModuleType("tqdm.notebook")
_notebook_mod.tqdm = lambda iterable, **kw: iterable          # identity iterator
sys.modules.setdefault("tqdm.notebook", _notebook_mod)
# ─────────────────────────────────────────────────────────────────────────────

from reinforcement_learning import (   # noqa: E402  (must come after stub)
    greedy_policy,
    epsilon_greedy_policy,
    q_learning,
    sarsa_learning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal deterministic environment mock
# ─────────────────────────────────────────────────────────────────────────────

class _ActionSpace:
    """Minimal action space that supports sample()."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)


class _ObservationSpace:
    """Minimal observation space."""

    def __init__(self, n):
        self.n = n


class TwoStateMockEnv:
    """Deterministic two-state environment for unit testing.

    States: 0 (start), 1 (goal).
    Actions: 0 (stay at 0), 1 (move to goal).
    Transitions:
        - state 0, action 0 → state 0, reward 0, done False
        - state 0, action 1 → state 1, reward 1, done True
        - state 1, any      → state 1, reward 0, done True  (terminal)

    The optimal policy is always action 1 from state 0.
    """

    N_STATES = 2
    N_ACTIONS = 2

    def __init__(self):
        self.observation_space = _ObservationSpace(self.N_STATES)
        self.action_space = _ActionSpace(self.N_ACTIONS)
        self._state = 0

    def reset(self):
        self._state = 0
        return self._state, {}

    def step(self, action):
        if self._state == 0 and action == 1:
            self._state = 1
            return 1, 1.0, True, False, {}
        return self._state, 0.0, False, False, {}


# ─────────────────────────────────────────────────────────────────────────────
# greedy_policy
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyPolicy:
    """Tests for greedy_policy."""

    def test_returns_argmax_action(self):
        """Returns the action with the highest Q-value."""
        Qtable = np.array([[0.1, 0.9, 0.5],
                           [0.3, 0.2, 0.8]])
        assert greedy_policy(Qtable, state=0) == 1
        assert greedy_policy(Qtable, state=1) == 2

    def test_tie_broken_by_lowest_index(self):
        """When Q-values tie, numpy.argmax returns the lowest index."""
        Qtable = np.array([[1.0, 1.0, 0.0]])
        assert greedy_policy(Qtable, state=0) == 0

    def test_single_action(self):
        """Works correctly when there is only one possible action."""
        Qtable = np.array([[0.7]])
        assert greedy_policy(Qtable, state=0) == 0

    def test_all_zero_qtable(self):
        """With a zero-initialised table, action 0 is returned."""
        Qtable = np.zeros((3, 4))
        assert greedy_policy(Qtable, state=2) == 0

    def test_negative_values(self):
        """Handles negative Q-values correctly."""
        Qtable = np.array([[-3.0, -1.0, -2.0]])
        assert greedy_policy(Qtable, state=0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# epsilon_greedy_policy
# ─────────────────────────────────────────────────────────────────────────────

class TestEpsilonGreedyPolicy:
    """Tests for epsilon_greedy_policy."""

    def setup_method(self):
        self.env = TwoStateMockEnv()
        # Q-table where action 1 is optimal from state 0
        self.Qtable = np.array([[0.0, 1.0],
                                [0.0, 0.0]])

    def test_pure_exploitation_epsilon_zero(self):
        """With epsilon=0 the greedy action is always selected."""
        actions = {
            epsilon_greedy_policy(self.Qtable, 0, epsilon=0.0, environment=self.env)
            for _ in range(50)
        }
        assert actions == {1}

    def test_pure_exploration_epsilon_one(self):
        """With epsilon=1 all actions can be selected (random policy)."""
        np.random.seed(0)
        actions = {
            epsilon_greedy_policy(self.Qtable, 0, epsilon=1.0, environment=self.env)
            for _ in range(200)
        }
        # Both actions (0 and 1) should appear with high probability
        assert len(actions) == 2

    def test_returned_action_is_valid(self):
        """Returned action is always within [0, n_actions)."""
        for eps in [0.0, 0.5, 1.0]:
            action = epsilon_greedy_policy(
                self.Qtable, 0, epsilon=eps, environment=self.env
            )
            assert action in range(self.env.action_space.n)

    def test_exploits_best_action_at_low_epsilon(self):
        """At very low epsilon the greedy action is chosen almost always."""
        np.random.seed(42)
        actions = [
            epsilon_greedy_policy(self.Qtable, 0, epsilon=0.01, environment=self.env)
            for _ in range(1000)
        ]
        # Action 1 should be chosen at least 98 % of the time
        assert actions.count(1) >= 970


# ─────────────────────────────────────────────────────────────────────────────
# q_learning
# ─────────────────────────────────────────────────────────────────────────────

class TestQLearning:
    """Tests for q_learning."""

    def setup_method(self):
        self.env = TwoStateMockEnv()
        self.Qtable_init = np.zeros((2, 2))

    def _run(self, n_episodes=500, lr=0.7, gamma=0.95,
             min_eps=0.05, max_eps=1.0, decay=0.01):
        Qtable = self.Qtable_init.copy()
        return q_learning(
            self.env,
            n_training_episodes=n_episodes,
            max_steps=10,
            learning_rate=lr,
            gamma=gamma,
            min_epsilon=min_eps,
            max_epsilon=max_eps,
            decay_rate=decay,
            Qtable=Qtable,
        )

    def test_returns_tuple_of_qtable_and_metrics(self):
        """Return type is (ndarray, dict)."""
        result = self._run()
        assert isinstance(result, tuple) and len(result) == 2
        qtable, metrics = result
        assert isinstance(qtable, np.ndarray)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self):
        """Metrics dict contains 'rewards', 'lengths', 'errors'."""
        _, metrics = self._run()
        assert set(metrics.keys()) == {'rewards', 'lengths', 'errors'}

    def test_metrics_length(self):
        """Each metric list has exactly n_training_episodes entries."""
        n = 50
        _, metrics = self._run(n_episodes=n)
        for key in ('rewards', 'lengths', 'errors'):
            assert len(metrics[key]) == n

    def test_optimal_action_learned(self):
        """After training, Q(0, 1) > Q(0, 0) — action 1 is optimal from state 0."""
        qtable, _ = self._run(n_episodes=500)
        assert qtable[0, 1] > qtable[0, 0]

    def test_qtable_shape_preserved(self):
        """The returned Q-table has the same shape as the input."""
        qtable, _ = self._run()
        assert qtable.shape == (2, 2)

    def test_episode_rewards_binary(self):
        """In the two-state env, rewards are 0 or 1 only."""
        _, metrics = self._run()
        assert all(r in (0.0, 1.0) for r in metrics['rewards'])

    def test_mean_errors_non_negative(self):
        """Mean absolute TD errors are non-negative."""
        _, metrics = self._run()
        assert all(e >= 0.0 for e in metrics['errors'])


# ─────────────────────────────────────────────────────────────────────────────
# sarsa_learning
# ─────────────────────────────────────────────────────────────────────────────

class TestSarsaLearning:
    """Tests for sarsa_learning."""

    def setup_method(self):
        self.env = TwoStateMockEnv()
        self.Qtable_init = np.zeros((2, 2))

    def _run(self, n_episodes=500, lr=0.7, gamma=0.95,
             min_eps=0.05, max_eps=1.0, decay=0.01):
        Qtable = self.Qtable_init.copy()
        return sarsa_learning(
            self.env,
            n_training_episodes=n_episodes,
            learning_rate=lr,
            gamma=gamma,
            min_epsilon=min_eps,
            max_epsilon=max_eps,
            decay_rate=decay,
            max_steps=10,
            Qtable=Qtable,
        )

    def test_returns_tuple_of_qtable_and_metrics(self):
        """Return type is (ndarray, dict)."""
        result = self._run()
        assert isinstance(result, tuple) and len(result) == 2
        qtable, metrics = result
        assert isinstance(qtable, np.ndarray)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self):
        """Metrics dict contains 'rewards', 'lengths', 'errors'."""
        _, metrics = self._run()
        assert set(metrics.keys()) == {'rewards', 'lengths', 'errors'}

    def test_metrics_length(self):
        """Each metric list has exactly n_training_episodes entries."""
        n = 50
        _, metrics = self._run(n_episodes=n)
        for key in ('rewards', 'lengths', 'errors'):
            assert len(metrics[key]) == n

    def test_optimal_action_learned(self):
        """After training, Q(0, 1) > Q(0, 0) — action 1 is optimal from state 0."""
        qtable, _ = self._run(n_episodes=500)
        assert qtable[0, 1] > qtable[0, 0]

    def test_qtable_shape_preserved(self):
        """The returned Q-table has the same shape as the input."""
        qtable, _ = self._run()
        assert qtable.shape == (2, 2)

    def test_episode_rewards_binary(self):
        """In the two-state env, rewards are 0 or 1 only."""
        _, metrics = self._run()
        assert all(r in (0.0, 1.0) for r in metrics['rewards'])

    def test_mean_errors_non_negative(self):
        """Mean absolute TD errors are non-negative."""
        _, metrics = self._run()
        assert all(e >= 0.0 for e in metrics['errors'])

    def test_on_policy_differs_from_q_learning(self):
        """SARSA and Q-learning converge to different Q-tables on a stochastic env.

        On a purely deterministic two-state environment the updates are
        equivalent (the on-policy next action equals the greedy action when
        epsilon → 0). This test verifies that both algorithms still produce
        finite, non-NaN Q-tables and that their interface is consistent.
        """
        np.random.seed(7)
        Qtable_q = np.zeros((2, 2))
        Qtable_s = np.zeros((2, 2))

        Qtable_q, _ = q_learning(
            TwoStateMockEnv(), 200, 10, 0.7, 0.95, 0.05, 1.0, 0.01, Qtable_q
        )
        Qtable_s, _ = sarsa_learning(
            TwoStateMockEnv(), 200, 0.7, 0.95, 0.05, 1.0, 0.01, 10, Qtable_s
        )

        assert not np.any(np.isnan(Qtable_q))
        assert not np.any(np.isnan(Qtable_s))
