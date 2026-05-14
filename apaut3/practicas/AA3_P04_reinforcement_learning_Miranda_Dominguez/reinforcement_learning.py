# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:06:34 2025
@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...

"""
from tqdm.notebook import tqdm
import numpy as np


def greedy_policy(Qtable, state):
    """Select the action with the highest Q-value for the given state.

    Implements a purely exploitative (greedy) policy: no randomness is
    involved. Ties are broken by ``numpy.argmax``, which returns the
    lowest-index maximiser.

    Args:
        Qtable (numpy.ndarray): Q-value table of shape
            ``(n_states, n_actions)``.
        state (int): Index of the current state.

    Returns:
        int: Index of the greedy action for ``state``.
    """
    action = np.argmax(Qtable[state])
    return action


def epsilon_greedy_policy(Qtable, state, epsilon, environment):
    """Select an action using the epsilon-greedy exploration strategy.

    With probability ``epsilon`` a random action is sampled from the
    environment's action space (exploration); with probability
    ``1 - epsilon`` the greedy action is chosen (exploitation).

    Args:
        Qtable (numpy.ndarray): Q-value table of shape
            ``(n_states, n_actions)``.
        state (int): Index of the current state.
        epsilon (float): Exploration probability in ``[0, 1]``.
        environment: A Gymnasium environment exposing
            ``action_space.sample()``.

    Returns:
        int: Index of the selected action.
    """
    if np.random.random() < epsilon:
        action = environment.action_space.sample()
    else:
        action = np.argmax(Qtable[state])
    return action


def q_learning(
    environment,
    n_training_episodes,
    max_steps,
    learning_rate,
    gamma,
    min_epsilon,
    max_epsilon,
    decay_rate,
    Qtable,
):
    """Train a Q-learning agent using an off-policy TD update rule.

    At each step the Q-table is updated with the Bellman residual:

        Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]

    The behaviour policy is epsilon-greedy with exponential decay:

        ε_t = ε_min + (ε_max − ε_min) · exp(−λ · t)

    Args:
        environment: A Gymnasium environment with a discrete observation
            and action space.
        n_training_episodes (int): Number of episodes to train for.
        max_steps (int): Maximum number of steps allowed per episode.
        learning_rate (float): Step size α ∈ (0, 1].
        gamma (float): Discount factor γ ∈ [0, 1).
        min_epsilon (float): Minimum exploration probability ε_min.
        max_epsilon (float): Initial exploration probability ε_max.
        decay_rate (float): Exponential decay rate λ > 0 for epsilon.
        Qtable (numpy.ndarray): Initial Q-value table of shape
            ``(n_states, n_actions)``. Modified in-place.

    Returns:
        tuple[numpy.ndarray, dict]: A pair ``(Qtable, metrics)`` where
        ``Qtable`` is the trained Q-value table and ``metrics`` is a
        dict with keys:

        - ``'rewards'`` (list[float]): Total undiscounted reward per episode.
        - ``'lengths'`` (list[int]): Number of steps per episode.
        - ``'errors'`` (list[float]): Mean absolute TD error per episode.
    """
    episode_rewards = []
    episode_lengths = []
    episode_errors = []

    for episode in tqdm(range(n_training_episodes)):
        epsilon = (
            min_epsilon
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        )
        state, info = environment.reset()
        episode_over = False
        n_steps = 0
        total_reward = 0.0
        step_errors = []

        while not episode_over and n_steps < max_steps:
            n_steps += 1
            action = epsilon_greedy_policy(Qtable, state, epsilon, environment)
            new_state, reward, terminated, truncated, info = environment.step(action)
            episode_over = terminated or truncated

            # Off-policy update: bootstrap with max over next state
            td_error = reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action]
            step_errors.append(abs(td_error))
            Qtable[state, action] += learning_rate * td_error

            total_reward += reward
            state = new_state

        episode_rewards.append(total_reward)
        episode_lengths.append(n_steps)
        episode_errors.append(np.mean(step_errors) if step_errors else 0.0)

    metrics = {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'errors': episode_errors,
    }
    return Qtable, metrics


def sarsa_learning(
    environment,
    n_training_episodes,
    learning_rate,
    gamma,
    min_epsilon,
    max_epsilon,
    decay_rate,
    max_steps,
    Qtable,
):
    """Train a SARSA agent using an on-policy TD update rule.

    At each step the Q-table is updated with the on-policy Bellman residual:

        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') − Q(s, a)]

    where ``a'`` is sampled from the same epsilon-greedy policy used for
    acting. The first action of each episode is selected before entering
    the step loop so that the ``(s, a, r, s', a')`` quintet is complete
    at update time.

    The behaviour policy uses exponential epsilon decay:

        ε_t = ε_min + (ε_max − ε_min) · exp(−λ · t)

    Args:
        environment: A Gymnasium environment with a discrete observation
            and action space.
        n_training_episodes (int): Number of episodes to train for.
        learning_rate (float): Step size α ∈ (0, 1].
        gamma (float): Discount factor γ ∈ [0, 1).
        min_epsilon (float): Minimum exploration probability ε_min.
        max_epsilon (float): Initial exploration probability ε_max.
        decay_rate (float): Exponential decay rate λ > 0 for epsilon.
        max_steps (int): Maximum number of steps allowed per episode.
        Qtable (numpy.ndarray): Initial Q-value table of shape
            ``(n_states, n_actions)``. Modified in-place.

    Returns:
        tuple[numpy.ndarray, dict]: A pair ``(Qtable, metrics)`` where
        ``Qtable`` is the trained Q-value table and ``metrics`` is a
        dict with keys:

        - ``'rewards'`` (list[float]): Total undiscounted reward per episode.
        - ``'lengths'`` (list[int]): Number of steps per episode.
        - ``'errors'`` (list[float]): Mean absolute TD error per episode.
    """
    episode_rewards = []
    episode_lengths = []
    episode_errors = []

    for episode in tqdm(range(n_training_episodes)):
        epsilon = (
            min_epsilon
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        )
        state, info = environment.reset()
        # On-policy: choose first action before the loop
        action = epsilon_greedy_policy(Qtable, state, epsilon, environment)
        episode_over = False
        n_steps = 0
        total_reward = 0.0
        step_errors = []

        while not episode_over and n_steps < max_steps:
            n_steps += 1
            new_state, reward, terminated, truncated, info = environment.step(action)
            episode_over = terminated or truncated

            # On-policy update: bootstrap with the actual next action
            next_action = epsilon_greedy_policy(Qtable, new_state, epsilon, environment)
            td_error = reward + gamma * Qtable[new_state, next_action] - Qtable[state, action]
            step_errors.append(abs(td_error))
            Qtable[state, action] += learning_rate * td_error

            total_reward += reward
            state = new_state
            action = next_action

        episode_rewards.append(total_reward)
        episode_lengths.append(n_steps)
        episode_errors.append(np.mean(step_errors) if step_errors else 0.0)

    metrics = {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'errors': episode_errors,
    }
    return Qtable, metrics


# ── Deep Q-Network (DQN) ──────────────────────────────────────────────────────
# Requires PyTorch: pip install torch
try:
    import torch
    import torch.nn as nn
    from collections import deque
    import random as _random

    class QNetwork(nn.Module):
        """Two-hidden-layer feedforward network approximating Q(s, a; theta)."""

        def __init__(self, n_states, n_actions, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, x):
            return self.net(x)


    class ReplayBuffer:
        """Fixed-size experience replay buffer with uniform sampling."""

        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            return _random.sample(self.buffer, batch_size)

        def __len__(self):
            return len(self.buffer)


    def dqn_learning(
        environment,
        n_training_episodes,
        max_steps,
        learning_rate,
        gamma,
        min_epsilon,
        max_epsilon,
        decay_rate,
        batch_size=64,
        buffer_capacity=10000,
        target_update_freq=200,
        hidden_dim=128,
        step_penalty=0.0,
    ):
        n_states = environment.observation_space.n
        n_actions = environment.action_space.n
        grid_size = int(n_states ** 0.5)  # side length of the grid (8 for 8x8)

        # Use (row, col) normalised coordinates instead of one-hot.
        # States close together in the grid have similar inputs, so the network
        # can generalise across spatially neighbouring states.
        policy_net = QNetwork(2, n_actions, hidden_dim)
        target_net = QNetwork(2, n_actions, hidden_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        buffer = ReplayBuffer(buffer_capacity)

        episode_rewards = []
        episode_lengths = []
        episode_errors = []

        def state_to_tensor(s):
            """Encode state as normalised (row, col) coordinates in [0, 1]^2."""
            row = s // grid_size
            col = s % grid_size
            return torch.tensor(
                [row / (grid_size - 1), col / (grid_size - 1)],
                dtype=torch.float32,
            )

        for episode in tqdm(range(n_training_episodes)):
            epsilon = (
                min_epsilon
                + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            )
            state, info = environment.reset()
            episode_over = False
            n_steps = 0
            total_reward = 0.0
            step_errors = []

            while not episode_over and n_steps < max_steps:
                n_steps += 1

                # Epsilon-greedy action selection using the policy network
                if np.random.random() < epsilon:
                    action = environment.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(state_to_tensor(state)).argmax().item()

                new_state, reward, terminated, truncated, info = environment.step(action)
                episode_over = terminated or truncated
                # Apply step penalty to every transition to create a dense reward
                # signal: the agent is penalised for each step that is not the goal,
                # encouraging shorter paths and providing learning signal even when
                # the goal has not been reached.
                shaped_reward = reward + step_penalty
                buffer.push(state, action, shaped_reward, new_state, episode_over)
                total_reward += reward  # track original reward for fair comparison with tabular methods
                state = new_state

                # Gradient step once the buffer has enough samples
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    s_b, a_b, r_b, ns_b, d_b = zip(*batch)

                    s_t  = torch.stack([state_to_tensor(s) for s in s_b])
                    ns_t = torch.stack([state_to_tensor(s) for s in ns_b])
                    a_t  = torch.tensor(a_b, dtype=torch.long)
                    r_t  = torch.tensor(r_b, dtype=torch.float32)
                    d_t  = torch.tensor(d_b, dtype=torch.float32)

                    # Current Q-values for the taken actions
                    current_q = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

                    # Bellman target using the frozen target network
                    with torch.no_grad():
                        max_next_q = target_net(ns_t).max(dim=1).values
                        target_q = r_t + gamma * max_next_q * (1.0 - d_t)

                    loss = nn.functional.mse_loss(current_q, target_q)
                    step_errors.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping prevents exploding gradients with sparse rewards
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()

            # Periodically sync the target network weights
            if episode % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            episode_rewards.append(total_reward)
            episode_lengths.append(n_steps)
            episode_errors.append(np.mean(step_errors) if step_errors else 0.0)

        metrics = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'errors': episode_errors,
        }
        return policy_net, metrics


    def dqn_greedy_episode(environment, policy_net, n_states):
        """Run one greedy episode with the DQN policy and return rendered frames."""
        frames = []
        state, info = environment.reset()
        done = False
        grid_size = int(n_states ** 0.5)

        while not done:
            frames.append(environment.render())
            row = state // grid_size
            col = state % grid_size
            coords = torch.tensor(
                [row / (grid_size - 1), col / (grid_size - 1)],
                dtype=torch.float32,
            )
            with torch.no_grad():
                action = policy_net(coords).argmax().item()
            state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            if done:
                frames.append(environment.render())

        environment.close()
        return frames

except ImportError:
    pass
