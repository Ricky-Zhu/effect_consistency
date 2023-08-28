import numpy as np
import torch
import gym
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from modified_envs import *
import os
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image


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
    def __init__(self, policy_path, state_dim, action_dim, max_action, opt):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.opt = opt
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')
        self.source_env_logs = os.path.join(self.opt.log_root, 'data/{}_data'.format(self.opt.env))
        self.target_env_logs = os.path.join(self.opt.log_root, 'data/{}_data'.format(self.opt.target_env))
        self.clip_range = 5
        self.mean1, self.std1 = self.get_mean_std(self.source_env_logs, opt.data_id1)
        self.mean2, self.std2 = self.get_mean_std(self.target_env_logs, opt.data_id2)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def select_cross_action(self, state, gxmodel, axmodel, return_tran_state=False):
        state = torch.tensor(state).float().cuda()
        state = (state - self.mean2) / self.std2

        # state = gxmodel(state.unsqueeze(0))
        tmp = gxmodel(state[:self.opt.state_dim2].unsqueeze(0))
        state = self.mean1.clone()
        state[:self.opt.state_dim1] = tmp

        state_temp = state.clone().detach()
        state = state * self.std1 + self.mean1
        state = state.cpu().data.numpy()
        action = self.select_action(state)
        action = axmodel(torch.tensor(state_temp).float().cuda(),
                         torch.tensor(action).float().cuda().unsqueeze(0))
        action = action.cpu().data.numpy()
        if return_tran_state:
            return state, action
        return action

    def select_cross_action_reverse(self, state, gxmodel, axmodel, return_tran_state=False):
        state = torch.tensor(state).float().cuda()
        state = (state - self.mean2) / self.std2

        # state = gxmodel(state.unsqueeze(0))
        tmp = gxmodel(state[:self.opt.state_dim1].unsqueeze(0))
        state = self.mean1.clone()
        state[:self.opt.state_dim2] = tmp

        state_temp = state.clone().detach()
        state = state * self.std1 + self.mean1
        state = state.cpu().data.numpy()
        action = self.select_action(state)
        action = axmodel(torch.tensor(state_temp).float().cuda(),
                         torch.tensor(action).float().cuda().unsqueeze(0))
        action = action.cpu().data.numpy()
        if return_tran_state:
            return state, action
        return action

    def get_mean_std(self, prefix, data_id):
        data_path = os.path.join(prefix, '{}'.format(data_id))
        mean_std_path = os.path.join(data_path, 'now_state.npy')
        data = np.load(mean_std_path)
        mean = torch.tensor(data.mean(0)).float().cuda()
        std = torch.tensor(data.std(0)).float().cuda()
        std[(abs(std < 0.1))] = 1
        return mean, std


class CrossPolicy:
    def __init__(self, opt):
        self.opt = opt
        self.env_name = opt.target_env
        if opt.optimal:
            self.policy_path = os.path.join(opt.log_root,
                                            '{}_optimal_base/models/TD3_{}_0_actor'.format(opt.env, opt.env))

        else:
            self.policy_path = os.path.join(opt.log_root,
                                            '{}_base/models/TD3_{}_0_actor'.format(opt.env, opt.env))
        self.state_dim = opt.state_dim1
        self.action_dim = opt.action_dim1
        self.max_action = 1
        self.policy = TD3(self.policy_path,
                          self.state_dim,
                          self.action_dim,
                          self.max_action,
                          self.opt)
        self.env = gym.make(self.env_name)
        self.env.seed(opt.seed)

    def eval_policy_reverse(self,
                            gxmodel=None,
                            axmodel=None,
                            imgpath=None,
                            eval_episodes=10,
                            return_xy_pos=False,
                            err_rec=None,
                            eval_type='mujoco',
                            return_error_mean=False,
                            target_policy_path=None):
        '''
        evaluate the learned mapping for target to source
        Args:
            gxmodel:
            axmodel:
            imgpath:
            eval_episodes:
            return_xy_pos:
            err_rec:
            eval_type:
            return_error_mean:

        Returns:

        '''
        # load oracle target policy
        from copy import copy
        temp_opt = copy(self.opt)
        temp_opt.env = self.opt.target_env
        temp_opt.target_env = self.opt.env

        policy = TD3(target_policy_path,
                     temp_opt.state_dim2,
                     temp_opt.action_dim2,
                     self.max_action,
                     temp_opt)
        x_pos = []
        y_pos = []
        error_mean = []
        success_count = 0
        eval_env = gym.make(temp_opt.target_env)
        state_buffer = []
        action_buffer = []
        avg_reward, new_reward = 0., 0.
        save_flag = False
        if eval_type == 'robot':
            err_rec = None
        if imgpath is not None:
            if not os.path.exists(imgpath):
                os.mkdir(imgpath)
            save_flag = True

        for i in tqdm(range(eval_episodes)):
            if err_rec is not None:
                err_rec.reset()
            err_mean_per = []
            state, done = eval_env.reset(), False
            if return_xy_pos:
                x_pos.append(eval_env.sim.data.qpos[0])
                y_pos.append(eval_env.sim.data.qpos[1])
            if save_flag:
                episode_path = os.path.join(imgpath, 'episode_{}'.format(i))
                if not os.path.exists(episode_path):
                    os.mkdir(episode_path)
            count = 0
            while not done:
                state = np.array(state)

                trans_state, action = policy.select_cross_action_reverse(state, gxmodel, axmodel,
                                                                         return_tran_state=True)
                if err_rec is not None:
                    err_rec(state, trans_state)
                    err_mean_per.append(err_rec.err_mean)
                state_buffer.append(state)
                action_buffer.append(action)
                state, reward, done, info = eval_env.step(action)
                if eval_type == 'robot':
                    if info['success']:
                        success_count += 1

                avg_reward += reward
                if return_xy_pos:
                    x_pos.append(eval_env.sim.data.qpos[0])
                    y_pos.append(eval_env.sim.data.qpos[1])
                # if self.env_name=='HalfCheetah-v2':
                #     avg_reward += info['reward_run']
                # elif self.env_name=='Swimmer-v2':
                #     avg_reward += info['reward_fwd']
                # elif self.env_name=='Ant-v2':
                #     avg_reward += info['reward_forward']

                if save_flag:
                    img = eval_env.sim.render(mode='offscreen', camera_name='track', width=256, height=256)
                    Image.fromarray(img[::-1, :, :]).save(os.path.join(episode_path, 'img_{}.jpg'.format(count)))
                count += 1
            error_mean.append(err_mean_per)
        avg_reward /= eval_episodes

        if eval_type == 'robot':
            if success_count != 0:
                print('success rate {}'.format(success_count / float(eval_episodes)))

        # print("-----------------------------------------------")
        # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        # print("-----------------------------------------------")
        if err_rec is not None:
            print('err mean:{} err var:{} err max:{}'.format(err_rec.err_mean, err_rec.err_var,
                                                             err_rec.err_max))
        if return_xy_pos:
            return avg_reward, (x_pos, y_pos)
        elif return_error_mean:
            return avg_reward, error_mean
        else:
            return avg_reward

    def eval_policy(self,
                    gxmodel=None,
                    axmodel=None,
                    imgpath=None,
                    eval_episodes=10,
                    return_xy_pos=False,
                    err_rec=None,
                    eval_type='mujoco',
                    return_error_mean=False,
                    return_trans_state=False):
        x_pos = []
        y_pos = []
        error_mean = []
        success_count = 0
        eval_env = self.env
        state_buffer = []
        trans_state_buffer = []
        action_buffer = []
        avg_reward, new_reward = 0., 0.
        save_flag = False
        if eval_type == 'robot':
            err_rec = None
        if imgpath is not None:
            if not os.path.exists(imgpath):
                os.mkdir(imgpath)
            save_flag = True

        for i in tqdm(range(eval_episodes)):
            if err_rec is not None:
                err_rec.reset()
            err_mean_per = []
            state, done = eval_env.reset(), False
            if return_xy_pos:
                x_pos.append(eval_env.sim.data.qpos[0])
                y_pos.append(eval_env.sim.data.qpos[1])
            if save_flag:
                episode_path = os.path.join(imgpath, 'episode_{}'.format(i))
                if not os.path.exists(episode_path):
                    os.mkdir(episode_path)
            count = 0
            while not done:
                state = np.array(state)

                trans_state, action = self.policy.select_cross_action(state, gxmodel, axmodel, return_tran_state=True)
                if return_trans_state:
                    trans_state_buffer.append(trans_state)
                if err_rec is not None:
                    err_rec(state, trans_state)
                    err_mean_per.append(err_rec.err_mean)
                state_buffer.append(state)
                action_buffer.append(action)
                state, reward, done, info = eval_env.step(action)
                if eval_type == 'robot':
                    if info['success']:
                        success_count += 1

                avg_reward += reward
                if return_xy_pos:
                    x_pos.append(eval_env.sim.data.qpos[0])
                    y_pos.append(eval_env.sim.data.qpos[1])
                # if self.env_name=='HalfCheetah-v2':
                #     avg_reward += info['reward_run']
                # elif self.env_name=='Swimmer-v2':
                #     avg_reward += info['reward_fwd']
                # elif self.env_name=='Ant-v2':
                #     avg_reward += info['reward_forward']

                if save_flag:
                    img = eval_env.sim.render(mode='offscreen', camera_name='track', width=256, height=256)
                    Image.fromarray(img[::-1, :, :]).save(os.path.join(episode_path, 'img_{}.jpg'.format(count)))
                count += 1
            error_mean.append(err_mean_per)
        avg_reward /= eval_episodes

        if eval_type == 'robot':
            if success_count != 0:
                print('success rate {}'.format(success_count / float(eval_episodes)))

        # print("-----------------------------------------------")
        # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        # print("-----------------------------------------------")
        if err_rec is not None:
            print('err mean:{} err var:{} err max:{}'.format(err_rec.err_mean, err_rec.err_var,
                                                             err_rec.err_max))

        if return_trans_state:
            return state_buffer, trans_state_buffer
        if return_xy_pos:
            return avg_reward, (x_pos, y_pos)
        elif return_error_mean:
            return avg_reward, error_mean
        else:
            return avg_reward
