import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_management_env1 import SliceManagementEnv1

env = SliceManagementEnv1()


model = DQN.load("dqn_maintain1.zip", env)
#model = DQN.load("/home/mario/Documents/DQN_Models/Model 1/gym-examples2/dqn_slices", env)

obs, info = env.reset()

cont = 0
while cont<99:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
    cont += 1
    if terminated or truncated:
        obs, info = env.reset()