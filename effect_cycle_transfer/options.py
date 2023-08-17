import os
import argparse


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', default='../logs/cross_morphology_effect', type=str)
    parser.add_argument('--exp_id', default=10, type=int)
    parser.add_argument("--env", default="Swimmer-v2")
    parser.add_argument("--target_env", default="Swimmer_4part-v2")
    # parser.add_argument('--data_type1', type=str, default='base', help='data type')
    # parser.add_argument('--data_type2', type=str, default='3leg', help='data type')
    parser.add_argument('--data_id1', type=str, default=str('1'), help='data id')
    parser.add_argument('--data_id2', type=str, default=str(1), help='data id')

    # parser.add_argument('--state_dim1', default=17, type=int)
    # parser.add_argument('--action_dim1', default=6, type=int)

    # parser.add_argument('--state_dim2', default=23, type=int)
    # parser.add_argument('--action_dim2', default=9, type=int)

    parser.add_argument('--cut_state1', default=0, type=int)
    parser.add_argument('--cut_state2', default=0, type=int)

    parser.add_argument('--episode_n', default=200, type=int)
    parser.add_argument('--pair_n', default=100, type=int)  # TODO change the parameter to 7000
    parser.add_argument('--display_gap', default=1, type=int)  # TODO change the parameter to 1000
    parser.add_argument('--eval_gap', default=1, type=int)  # TODO change the parameter to 1000
    parser.add_argument('--eval_n', default=1, type=int)  # TODO change the parameter to 10

    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--istrain', default=True, type=bool)
    parser.add_argument('--pretrain_i', default=1, type=int, help='1 -> ture, 0 -> false')
    parser.add_argument('--start_train', action='store_true')

    parser.add_argument('--lr_Gx', default=1e-4, type=float)
    parser.add_argument('--lr_Ax', default=0., type=float)
    parser.add_argument('--lambda_G0', default=10., type=float)
    parser.add_argument('--lambda_D', default=15., type=float)
    parser.add_argument('--lambda_G1', default=10., type=float)
    parser.add_argument('--lambda_GactA', default=10., type=float)
    parser.add_argument('--lambda_GactB', default=10., type=float)
    parser.add_argument('--lambda_Gcyc', default=30., type=float)
    parser.add_argument('--lambda_F', default=50., type=float)
    parser.add_argument('--init_start', default=1, type=int, help='1 -> ture, 0 -> false')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--iteration', default=3, type=int)
    parser.add_argument('--eval_type', default='mujoco', type=str, help="robot or mujoco")
    parser.add_argument("--optimal", action='store_true', help='whether the source policy is optimal')

    args = parser.parse_args()

    return args
