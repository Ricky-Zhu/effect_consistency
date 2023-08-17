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
from trans_xy_err import error_rec


def evaluate_mapping_error(model_path, args):
    model = CycleGANModel(args)
    model.load(model_path)

    # set the xy record
    xy_err_rec = error_rec(x_arg=0, y_arg=1)

    # evaluate the avg episode return in the target domain
    rewards = model.cross_policy.eval_policy(
        gxmodel=model.netG_2to1,
        axmodel=model.net_action_G_1to2,
        eval_episodes=args.eval_n,
        eval_type=args.eval_type,
        err_rec=xy_err_rec)
    print(rewards)


if __name__ == "__main__":
    model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/HalfCheetah-v2_HalfCheetah_3leg-v2/exp_2023-08-11-20-55-52/weights'
    args = get_options()
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
    evaluate_mapping_error(model_path, args)
