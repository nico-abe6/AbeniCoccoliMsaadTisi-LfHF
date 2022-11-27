"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym
import matplotlib.pyplot as plt
import numpy as np

from tamer.agent import Tamer

async def main():
    env = gym.make('CartPole-v1')

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 5
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.5

    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    await agent.train(model_file_to_save='autosave')
    n_episodes=50
    rew=agent.play(n_episodes, render=True)
    
    N_EPISODE = [i+1 for i in range(n_episodes)]
    
    np.save("rew_tamer.npy",rew)

    plt.plot(N_EPISODE, rew)
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())


