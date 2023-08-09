import os
import random
import numpy as np


class CycleData:
    def __init__(self, opt):
        self.opt = opt
        self.episode_n = opt.episode_n
        self.log_root = opt.log_root

        # source data root
        self.data_root1 = os.path.join(self.log_root, 'data/{}_data'.format(self.opt.env),
                                       '{}'.format(self.opt.data_id1))
        # target data root
        self.data_root2 = os.path.join(self.log_root, 'data/{}_data'.format(self.opt.target_env),
                                       '{}'.format(self.opt.data_id2))
        self.data1 = self.collect(self.data_root1, opt.state_dim1)
        self.data2 = self.collect(self.data_root2, opt.state_dim2)
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------\n')
        self.sample_n1 = self.data1[0].shape[0]
        self.sample_n2 = self.data2[0].shape[0]

    def sample(self, batch_size=32):
        id1 = random.sample(range(self.sample_n1), batch_size)
        sample1 = (self.data1[0][id1], self.data1[1][id1], self.data1[2][id1])
        id2 = random.sample(range(self.sample_n2), batch_size)
        sample2 = (self.data2[0][id2], self.data2[1][id2], self.data2[2][id2])
        return sample1, sample2

    def collect(self, data_folder, state_dim):
        now_path = os.path.join(data_folder, 'now_state.npy')
        nxt_path = os.path.join(data_folder, 'next_state.npy')
        act_path = os.path.join(data_folder, 'action.npy')
        now_obs = np.load(now_path)[:, :state_dim]
        nxt_obs = np.load(nxt_path)[:, :state_dim]
        action = np.load(act_path)

        mean = now_obs.mean(0)
        std = now_obs.std(0)
        std[(abs(std) < 0.1)] = 1
        now_obs = (now_obs - mean) / std
        nxt_obs = (nxt_obs - mean) / std

        return (now_obs, action, nxt_obs)
