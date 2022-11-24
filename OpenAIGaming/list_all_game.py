import gym.envs
for game_name in gym.envs.registry.env_specs.keys():
    print(game_name)
