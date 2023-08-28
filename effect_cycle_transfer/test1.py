from modified_envs import *
import gym
import numpy as np

rew_all = []

for seed in [0, 8, 10, 42, 100]:
    env = gym.make('UR5e-v2')
    print(env.action_space.shape,env.observation_space.shape)
    env.seed(seed)
    env.reset()
    rew_per = 0.
    while True:
        s, r, d, _ = env.step(env.action_space.sample())
        rew_per += r
        if d:
            break
    rew_all.append(rew_per)
    env.close()

rew_all = np.asarray(rew_all)
print(rew_all.mean(), rew_all.std())
