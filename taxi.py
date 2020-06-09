# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:21:33 2020

@author: halenur
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env

#Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
#hyperparameter
alpha=0.1
gama=0.9
epsilon=0.1 #explore range
#platting matrix
reward_list = []
dropout_list = []
    
episode_number = 10000

for i in range (1, episode_number):
    #initialize Q table
    state = env.reset()
    reward_count=0
    reward = 0
    dropout = 0
    while True:
        #choose an action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        #perform action
            next_state, reward, done, _ = env.step(action)
            
        #q learning function
            old_value = q_table[state,action]
            max_next_value = np.max(q_table[next_state])
            next_value = (1-alpha)*old_value + alpha*(reward + gama*max_next_value)
        
        #update Q table
            q_table[state,action] = next_value
        
        #state update
            state = next_state
        
        #find wrong dropout
            if reward == -10:
                dropout +=1
        
        reward_count += reward
        if i%100 == 0:
            env.render()
        
        if done:
            break
        
    dropout_list.append(dropout)
    reward_list.append(reward_count)
    print("episode: {}, reward: {}, dropout: {}".format(i, reward_count, dropout))
        
fig, axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")
        
axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropout")

axs[0].grid(True)
axs[1].grid(True)
plt.show()       
        
        
        
        