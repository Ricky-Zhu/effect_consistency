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
    id='Ant_5leg-v2',
    entry_point='modified_envs.mujoco:AntModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='UR5e-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'UR5e', 'gripper': True}
)

register(
    id='Panda-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'Panda', 'gripper': True}
)

register(
    id='Jaco-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'Jaco'}
)

register(
    id='Kinova3-v2',
    entry_point='modified_envs.mujoco:create_robot_env',
    kwargs={'robot_name': 'Kinova3'}
)
