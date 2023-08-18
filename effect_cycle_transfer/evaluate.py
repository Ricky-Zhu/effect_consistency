import os
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
import pickle
from options import get_options
from cycle.data import CycleData
from cycle.dyncycle import CycleGANModel
from cycle.utils import init_logs
from termcolor import cprint
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from trans_xy_err import error_rec
import pandas as pd
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_mapping_error(model_path, args):
    model = CycleGANModel(args)
    model.load(model_path)

    # set the xy record
    xy_err_rec = error_rec(x_arg=0, y_arg=1)

    # evaluate the avg episode return in the target domain
    rewards, error_mean = model.cross_policy.eval_policy(
        gxmodel=model.netG_2to1,
        axmodel=model.net_action_G_1to2,
        eval_episodes=args.eval_n,
        eval_type=args.eval_type,
        err_rec=xy_err_rec,
        return_error_mean=True)
    print(rewards)
    # evaluate reverse
    rewards, error_mean = model.cross_policy.eval_policy_reverse(
        gxmodel=model.netG_1to2,
        axmodel=model.net_action_G_2to1,
        eval_episodes=args.eval_n,
        eval_type=args.eval_type,
        err_rec=xy_err_rec,
        return_error_mean=True,
        target_policy_path='/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/HalfCheetah_3leg-v2_base/models/TD3_HalfCheetah_3leg-v2_0_actor')

    # fw = open('/home/ruiqi/projects/effect_consistency/results_analysis/ours_xy_err_analysis.txt', 'wb')
    # pickle.dump(error_mean,fw)
    # fw.close()
    # print(len(error_mean))
    # fig, axs = plt.subplots(1,5)
    # x_cor = np.arange(1000)
    # for i in range(5):
    #     temp_error_mean = np.asarray(error_mean[i])
    #     axs[i].plot(x_cor, temp_error_mean)
    # plt.show()
    print(rewards)
    # return rewards
    # return error_mean


if __name__ == "__main__":
    model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/HalfCheetah-v2_HalfCheetah_3leg-v2/exp_2023-08-17-17-14-04/weights'
    args = get_options()
    args.env = 'HalfCheetah-v2'
    args.target_env = 'HalfCheetah_3leg-v2'
    args.eval_n = 1
    args.init_start = False
    # source env information
    env_name = args.env
    env = gym.make(env_name)
    args.state_dim1 = env.observation_space.shape[0]
    args.action_dim1 = env.action_space.shape[0]
    env.close()

    # target env information
    env_name = args.target_env
    env = gym.make(env_name)
    args.state_dim2 = env.observation_space.shape[0]
    args.action_dim2 = env.action_space.shape[0]
    env.close()

    eval_rew = []
    for seed in [10]:
        args.seed = seed
        setup_seed(args.seed)
        rew = evaluate_mapping_error(model_path, args)
        eval_rew.append(rew)
    eval_rew = np.asarray(eval_rew)
    print(eval_rew.mean(), eval_rew.std())
