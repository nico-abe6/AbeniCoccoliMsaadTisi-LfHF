import numpy as np

mode = np.load("mode.npy", allow_pickle=True)
episodes = np.load("episodes.npy", allow_pickle=True)

print("\n\nModes ", mode)
print(np.shape(mode))
print("\n\n episodes", episodes)
print(np.shape(episodes))
