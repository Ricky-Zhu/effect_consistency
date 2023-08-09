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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_wandb(args):
    current_date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    wandb.login()
    wandb.init(
        project="cdat_det_{}_{}".format(args.env, args.target_env),
        config=vars(args),
        name="cdat_{}".format(current_date)
    )


def add_errors(model, display):
    errors = model.get_current_errors()
    for key, value in errors.items():
        display += '{}:{:.4f}  '.format(key, value)
    return display


def train(args):
    txt_logs, txt_eval_logs, training_args_logs, img_logs, weight_logs, log_dirs = init_logs(args)
    data_agent = CycleData(args)  # normalize and initial the pre-collected source and target domain data
    model = CycleGANModel(args)  # initialize all the needed networks
    model.iengine.train_statef(data_agent.data2)  # train the target inverse dynamics
    model.iengine_1.train_statef(data_agent.data1)  # train the source inverse dynamics

    if args.start_train:
        setup_wandb(args)

    print('evaluate the initial transferred policy in the target domain')
    model.cross_policy.eval_policy(
        gxmodel=model.netG_2to1,
        axmodel=model.net_action_G_1to2,
        eval_episodes=args.eval_n,
        eval_type=args.eval_type)

    best_reward = 0

    for iteration in range(3):
        print('iteration {}'.format(iteration + 1))

        args.lr_Gx = 1e-4
        args.lr_Ax = 1e-4
        model.update(args)
        end_id = 0
        start_id = end_id
        end_id = start_id + args.pair_n

        for batch_id in range(start_id, end_id):
            item = data_agent.sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            # real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> iteration {} \t Batch[{}/{}]'.format(iteration, batch_id + 1, args.pair_n)
                print(display)
                # wandb log the loss
                new_loss_dict = {}
                errs = model.get_current_errors()
                for k, v in errs.items():
                    k_ = 'iter_{}/g/{}'.format(iteration, k)
                    new_loss_dict[k_] = v
                if args.start_train:
                    wandb.log(new_loss_dict)

                model.visual()

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_2to1,
                    axmodel=model.net_action_G_1to2,
                    eval_episodes=args.eval_n,
                    eval_type=args.eval_type
                )

                if reward > best_reward:
                    best_reward = reward
                    model.save(weight_logs)

                if args.start_train:
                    wandb.log({'iter_{}/g/eval'.format(iteration): reward})
                    wandb.log({'best_reward': best_reward})

                eval_display = '\n G part iteration {} best_reward:{:.1f}  cur_reward:{:.1f}'.format(iteration,
                                                                                                     best_reward,
                                                                                                     reward)
                print(eval_display)

        args.init_start = False
        args.lr_Gx = 1e-4
        args.lr_Ax = 1e-4
        model.update(args)
        end_id = 0
        start_id = end_id
        end_id = start_id + args.pair_n
        print('update A phase')
        for batch_id in range(start_id, end_id):
            item = data_agent.sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            # real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id + 1, args.pair_n)
                print(display)
                # wandb log the loss
                new_loss_dict = {}
                errs = model.get_current_errors()
                for k, v in errs.items():
                    k_ = 'iter_{}/a/{}'.format(iteration, k)
                    new_loss_dict[k_] = v
                if args.start_train:
                    wandb.log(new_loss_dict)

                model.visual()

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_2to1,
                    axmodel=model.net_action_G_1to2,
                    eval_episodes=args.eval_n)

                if args.start_train:
                    wandb.log({'iter_{}/a/eval'.format(iteration): reward})
                    wandb.log({'best_reward': best_reward})
                eval_display = '\nA part iteration {} best_reward:{:.1f}  cur_reward:{:.1f}'.format(iteration,
                                                                                                    best_reward,
                                                                                                    reward)
                print(eval_display)


if __name__ == '__main__':
    args = get_options()
    setup_seed(args.seed)

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

    train(args)
