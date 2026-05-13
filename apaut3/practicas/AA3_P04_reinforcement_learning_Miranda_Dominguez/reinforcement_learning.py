# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:06:34 2025
@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...

"""
from tqdm.notebook import tqdm
import numpy as np


def greedy_policy(Qtable, state):
    # Exploitation only: take the action with the highest (state, action) value
    action = np.argmax(Qtable[state])
    return action


def epsilon_greedy_policy(Qtable, state, epsilon, environment):
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
    ):
        n_states = environment.observation_space.n
        n_actions = environment.action_space.n

        policy_net = QNetwork(n_states, n_actions, hidden_dim)
        target_net = QNetwork(n_states, n_actions, hidden_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        buffer = ReplayBuffer(buffer_capacity)

        episode_rewards = []
        episode_lengths = []
        episode_errors = []

        def state_to_tensor(s):
            """One-hot encode a discrete state index."""
            one_hot = torch.zeros(n_states)
            one_hot[s] = 1.0
            return one_hot

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
                buffer.push(state, action, reward, new_state, episode_over)
                total_reward += reward
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

        while not done:
            frames.append(environment.render())
            one_hot = torch.zeros(n_states)
            one_hot[state] = 1.0
            with torch.no_grad():
                action = policy_net(one_hot).argmax().item()
            state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            if done:
                frames.append(environment.render())

        environment.close()
        return frames

except ImportError:
    pass
