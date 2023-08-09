from modified_envs import *
import gym

env = gym.make('Panda-v2')
env.reset()
for i in range(1000):
    s, r, d, _ = env.step(env.action_space.sample())
    if d:
        break