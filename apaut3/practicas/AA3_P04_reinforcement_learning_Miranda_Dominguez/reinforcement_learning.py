# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:06:34 2025
@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...

"""
# from tqdm import tqdm

from tqdm.notebook import tqdm
import numpy as np

# [To-Do]: fill in how to select the action 
def greedy_policy(Qtable, state):
    # Exploitation only: take the action with the highest (state, action) value
    action = np.argmax(Qtable[state])

    return action 

# [To-Do]: Define the epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon, environment):
    action = None
    if np.random.random() < epsilon:
        action = environment.action_space.sample()
    else:
        action = np.argmax(Qtable[state])
    return action


# [To-Do]: Impelemt Q-learning
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
    
    for episode in tqdm(range(n_training_episodes)):
     
        # Reduce epsilon (reduce exploration as learning progresses)
        epsilon = ( 
            min_epsilon 
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        )
        
        # Reset the environment
        state, info = environment.reset()

        # Episode loop    
        episode_over = False
        n_steps = 0

        while not episode_over and n_steps < max_steps:
            n_steps += 1

            # Choose action (a) at state (s) using an epsilon-greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon, environment)
            # Take the action (a) and observe the new state(s') and reward (r)
            new_state, reward, terminated, truncated, info = environment.step(action)
            episode_over = terminated or truncated
            # Update Q-table and state with bellman formuula
            Qtable[state, action] = Qtable[state, action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action]
            )
            # Determine whether the episode is over
            state = new_state
            
    return Qtable


# [To-Do]: Implement SARSA 
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
    for episode in tqdm(range(n_training_episodes)):
     
        # Reduce epsilon (reduce exploration as learning progresses)
        epsilon = ( 
            min_epsilon 
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        )
        
        # Reset the environment
        state, info = environment.reset()
        action = epsilon_greedy_policy(Qtable, state, epsilon, environment)
        # Episode loop    
        episode_over = False
        n_steps = 0

        while not episode_over and n_steps < max_steps:
            n_steps += 1
            new_state, reward, terminated, truncated, info = environment.step(action)
            episode_over = terminated or truncated
            # Choose action (a) at state (s) using an epsilon-greedy policy

            next_action = epsilon_greedy_policy(Qtable, new_state, epsilon, environment)
            
            # Update Q-table and state with bellman formuula
            Qtable[state, action] = Qtable[state, action] + learning_rate * (
                reward + gamma * Qtable[new_state, next_action] - Qtable[state, action]
            )
            # Determine whether the episode is over
            state = new_state
            action = next_action
            
    return Qtable