import os
import numpy as np
import json


actions = np.load("actions.npy").tolist()
states = np.load("states.npy").tolist() # Observation
rewards = np.load("rewards.npy").tolist()
prev_states = np.load("prev_states.npy").tolist()
dones = np.load("done.npy").tolist()


episode_list = []

for i in range(len(rewards)):
    episode_list.append([prev_states[i], actions[i], rewards[i],states[i],dones[i]])


dict = {"0":episode_list}

print(len(rewards),len(states),len(dones),len(prev_states))

#print(dict)

#with open('file.txt', 'w') as file:
#     file.write(json.dumps(dict)) # use `json.loads` to do the reverse
