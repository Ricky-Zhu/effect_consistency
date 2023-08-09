from gym.envs.registration import register

register(
    id='Swimmer_4part-v2',
    entry_point='modified_envs.mujoco:SwimmerModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='HalfCheetah_3leg-v2',
    entry_point='modified_envs.mujoco:HalfCheetahModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='UR5e-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'UR5e'}
)

register(
    id='Panda-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'Panda'}
)
