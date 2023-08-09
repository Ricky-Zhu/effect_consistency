import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class TD3(object):
    def __init__(self, policy_path, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')

        self.clip_range = 5

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


# smooth function
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_xy_pos(path, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    policy = TD3(state_dim=state_dim,
                 action_dim=action_dim,
                 max_action=max_action,
                 policy_path=path)

    rew = 0
    x_pos = []
    y_pos = []
    s = env.reset()
    x_pos.append(env.sim.data.qpos[0])
    y_pos.append(env.sim.data.qpos[1])
    while True:
        a = policy.select_action(s)
        s_, r, done, _ = env.step(a)
        rew += r
        x_pos.append(env.sim.data.qpos[0])
        y_pos.append(env.sim.data.qpos[1])
        if done:
            break
        s = s_
    print(rew)
    # y_pos = smooth(y_pos, 500)
    plt.ylim(-1., 1.)
    plt.plot(x_pos, y_pos)

    plt.show()

    return x_pos, y_pos


def visual_xy(path):
    best_xy = pickle.load(open(path + '/xy_pos_best.txt', 'rb'))
    final_xy = pickle.load(open(path + '/xy_pos_final.txt', 'rb'))
    plt.ylim(-1., 1.)
    plt.plot(best_xy[0], best_xy[1], 'r')
    plt.plot(final_xy[0], final_xy[1], 'g')
    plt.show()


if __name__ == "__main__":
    # path = '/home/ruiqi/projects/Cycle_Dynamics/logs/cross_morphology_effect/HalfCheetah-v2_base/models/TD3_HalfCheetah-v2_0_actor'
    # env = gym.make('HalfCheetah-v2')
    # get_xy_pos(path, env)
    path = '/home/ruiqi/projects/Cycle_Dynamics/logs/cross_morphology_effect/HalfCheetah-v2_HalfCheetah_3leg-v2/exp_2023-06-19-18-29-16'
    visual_xy(path)
