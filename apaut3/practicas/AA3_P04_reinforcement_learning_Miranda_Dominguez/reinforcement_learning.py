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
    action = None
    return action 

# [To-Do]: Define the epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon, environment):
    action = None
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
      
            # Take the action (a) and observe the new state(s') and reward (r)

            # Update Q-table and state
            
            # Determine whether the episode is over
            
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
    return Qtable