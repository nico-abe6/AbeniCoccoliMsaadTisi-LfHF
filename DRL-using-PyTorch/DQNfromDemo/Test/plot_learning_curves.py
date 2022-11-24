import numpy as np
import os
from grafico import plot_rewards
import matplotlib.pyplot as plt
from scipy.signal import lfilter

rw_DQN = "rewself.npy"
rw_IL = "reward_IL.npy"
rw_HP = "rew.npy"

reward_DQN = np.load(rw_DQN, allow_pickle=True)
reward_IL = np.load(rw_IL, allow_pickle=True)
reward_HP = np.load(rw_HP, allow_pickle=True)

#print(np.size(episodes_HP))
#print(np.size(reward_HP))

n = 10  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
reward_DQN = lfilter(b, a, reward_DQN)
reward_HP = lfilter(b, a, reward_HP)
reward_IL = lfilter(b, a, reward_IL)

plt.figure(1)

plt.plot(reward_DQN, "b-", linewidth = 1)
plt.plot(reward_HP, "r-", linewidth = 1)
plt.plot(reward_IL, "g-", linewidth = 2)

plt.xlabel("x = Attempts")
plt.ylabel("y = reward")
plt.title("Learning Curves")
plt.legend(["R_DQN", "R_HumanPref", "R_ImitLearn"])
plt.show()
