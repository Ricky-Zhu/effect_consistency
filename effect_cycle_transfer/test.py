import numpy as np
import os


def concat_dataset(path1, path2, save_path, suffix='compound'):
    now_state_1 = np.load(path1 + '/now_state.npy')
    next_state_1 = np.load(path1 + '/next_state.npy')
    action_1 = np.load(path1 + '/action.npy')

    now_state_2 = np.load(path2 + '/now_state.npy')
    next_state_2 = np.load(path2 + '/next_state.npy')
    action_2 = np.load(path2 + '/action.npy')

    now_state = np.concatenate([now_state_1, now_state_2], axis=0)
    action = np.concatenate([action_1, action_2], axis=0)
    next_state = np.concatenate([next_state_1, next_state_2], axis=0)

    save_folder = save_path + '/{}'.format(suffix)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_folder + '/now_state.npy', now_state)
    np.save(save_folder + '/action.npy', action)
    np.save(save_folder + '/next_state.npy', next_state)


if __name__ == "__main__":
    path1 = '../../logs/cross_morphology_effect/data/HalfCheetah-v2_data/1'
    path2 = '../../logs/cross_morphology_effect/data/HalfCheetah-v2_data/1_expert_data'
    save_path = '../../logs/cross_morphology_effect/data/HalfCheetah-v2_data'
    concat_dataset(path1, path2, save_path)
