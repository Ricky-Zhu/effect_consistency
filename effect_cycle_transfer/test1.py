import gym
env= gym.make('Ant-v2')
env.action_space.seed(1)
print(env.action_space.sample()[:5])