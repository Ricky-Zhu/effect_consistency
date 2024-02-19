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
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def fetch_transformed_traj(model_path, args):
    '''
    fetch the target traj and the transformed traj
    Args:
        model_path:
        args:

    Returns:

    '''
    model = CycleGANModel(args)
    model.load(model_path)
    setup_seed(10)
    target_state, source_state = model.cross_policy.eval_policy(
        gxmodel=model.netG_2to1,
        axmodel=model.net_action_G_1to2,
        eval_episodes=1,
        eval_type=args.eval_type,
        return_trans_state=True)
    target_state = np.asarray(target_state)
    source_state = np.asarray(source_state)
    np.save("/home/ruiqi/projects/effect_consistency/results_analysis/ours_t2s_source_states_train.npy", source_state)
    np.save("/home/ruiqi/projects/effect_consistency/results_analysis/t2s_target_states_train.npy", target_state)


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
    # rewards, error_mean = model.cross_policy.eval_policy_reverse(
    #     gxmodel=model.netG_1to2,
    #     axmodel=model.net_action_G_2to1,
    #     eval_episodes=args.eval_n,
    #     eval_type=args.eval_type,
    #     err_rec=xy_err_rec,
    #     return_error_mean=True,
    #     target_policy_path='/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/HalfCheetah_3leg-v2_base/models/TD3_HalfCheetah_3leg-v2_0_actor')

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

    # return rewards
    # return error_mean


def evaluate_transferred_agent(model_path, args):
    '''
    fetch the target traj and the transformed traj
    Args:
        model_path:
        args:

    Returns:

    '''
    model = CycleGANModel(args)
    model.load(model_path)
    setup_seed(10)
    avg_return = model.cross_policy.eval_policy(
        gxmodel=model.netG_2to1,
        axmodel=model.net_action_G_1to2,
        eval_episodes=args.eval_n,
        eval_type=args.eval_type,
        render=True,
        return_images=False)
    # image_save_path='/home/ruiqi/projects/effect_consistency/effect_cycle_transfer/gif_logs'
    # appendix=f'{args.env}_{args.target_env}'
    # image_save_path = os.path.join(image_save_path,appendix)
    # if not os.path.exists(image_save_path):
    #     os.makedirs(image_save_path)
    # for i in range(len(images)):
    #     if args.eval_type=='robot':
    #         image = np.flipud(images[i])
    #     else:
    #         image = images[i]
    #     if args.eval_type=='mujoco':
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image_array_uint8 = np.uint8(image)
    #
    #     # Save the image using OpenCV
    #     cv2.imwrite(f'{image_save_path}/frame_{i:04d}.jpg', image_array_uint8)

    print(f'evaluate for {args.eval_n} episodes, avg return: {avg_return}')


if __name__ == "__main__":
    args = get_options()
    ENV = 0
    if ENV == 0:
        model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/Ant-v2_Ant_5leg-v2/exp_2024-02-19-18-22-15/weights'
        args.env = 'Ant-v2'
        args.target_env = 'Ant_5leg-v2'
    elif ENV == 1:
        model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/HalfCheetah-v2_HalfCheetah_3leg-v2/exp_2023-08-29-15-31-34/weights'
        args.env = 'HalfCheetah-v2'
        args.target_env = 'HalfCheetah_3leg-v2'
    elif ENV == 2:
        model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/Swimmer-v2_Swimmer_4part-v2/exp_2023-08-11-11-38-22/weights'
        args.env = 'Swimmer-v2'
        args.target_env = 'Swimmer_4part-v2'

    elif ENV == 3:
        model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/Jaco-v2_Kinova3-v2/exp_2023-08-11-16-06-25/weights'
        args.env = 'Jaco-v2'
        args.target_env = 'Kinova3-v2'
        args.eval_type = 'robot'

    elif ENV == 4:
        model_path = '/home/ruiqi/projects/effect_consistency/logs/cross_morphology_effect/UR5e-v2_Panda-v2/exp_2023-08-16-14-58-09/weights'
        args.env = 'UR5e-v2'
        args.target_env = 'Panda-v2'
        args.eval_type = 'robot'

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

    evaluate_transferred_agent(model_path, args)

    # eval_rew = []
    # for seed in [8, 10]:
    #     args.seed = seed
    #     setup_seed(args.seed)
    #     evaluate_mapping_error(model_path, args)
