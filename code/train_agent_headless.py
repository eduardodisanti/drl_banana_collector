from unityagents import UnityEnvironment
import numpy as np

import gym
import random
import torch
import numpy as np
from collections import deque
from collections import defaultdict

from dqn_agent import Agent
from auxs import moving_average

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
#
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

EPS_START = 1           # START EXPLORING A LOT
GAMMA = 0.99            # discount factor - THE OBJECTIVE OF THE GAME IS TO MAXIMIZE REWARDS AT THE END, HAS TO BE HIGH 

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

agent = Agent(state_size=state_size, action_size=action_size, seed=0, gamma=GAMMA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, tau=TAU, lr=LR, update_every=UPDATE_EVERY)

TARGET_AVG_SCORE = 13
NUM_OF_TARGET_EPISODES_FOR_AVG = 100

eps_min = 0.01      # EVEN EXPLORE AFTER MANY EPISODES
eps_decay = 0.995   # DECAY EXPLORE SLOWLY

def choose_action(state, agent, eps=0.):
    
    action = agent.act(state, eps=eps)
    
    return action
    
trained = False
episodes = 0
la = {0:0,1:0,2:0,3:0}
lq = []
consecutives_solved = 0
times_solved = 0
avg = 0
mav = 0
avgs = []
mavgs = []

eps = EPS_START

while not trained:
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = choose_action(state, agent)        # select an action
        la[action]+=1
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        if done:                                       # exit loop if episode finished
            break
        agent.step(state, action, reward, next_state, done)
        eps = max(eps_min, eps_decay*eps)
        state = next_state                             # roll over the state to next time step
            
    episodes += 1
    clear_output(wait=True)
    lq.append(score)
    
    avg = np.average(lq[-NUM_OF_TARGET_EPISODES_FOR_AVG:])
    avgs.append(avg)

    print("act", la)
    print("episodes", episodes, "last score", score, "current eps", eps, "solved", times_solved, "avg", avg)
    if score > 13.1:
        times_solved+=1
    else:
        consecutives_solved = 0
    if avg>TARGET_AVG_SCORE:
        trained = True
        torch.save(agent.qnetwork_local.state_dict(), 'banana_raytracing_eds.pt')
            
print("Score: {}".format(score))   