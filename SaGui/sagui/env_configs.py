from gym.envs.registration import register

config_guide1 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'robot_keepout': 0.0,
    'task': 'none',
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'hazards_size': 0.7,
    'hazards_keepout': 0.705,
    'hazards_locations': [(0, 0)]
}

config_guide2 = {
    'placements_extents': [-2, -2, 2, 2],
    'robot_base': 'xmls/car.xml',
    'robot_keepout': 0.0,
    'task': 'none',
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'constrain_vases': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'hazards_locations': [(1.0, 1.0), (1, -1), (-0.2, 0.2), (-1.4, -1.4)],
    'vases_num': 4,
    'vases_size': 0.2,
    'vases_keepout': 0.18,
    'vases_locations': [(-1.0, -1.0), (-1, 1), (0.2, -0.2), (1.4, 1.4)]
}

config_guide3 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'robot_keepout': 0.0,
    'task': 'none',
    'constrain_hazards': True,
    'observe_hazards': True,
    'observe_vases': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 8,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'vases_num': 1,
}

config_guide4 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'robot_keepout': 0.0,
    'task': 'none',
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'hazards_size': 0.7,
    'hazards_keepout': 0.75,
    'hazards_locations': [(0, 0)]
}

config_guide5 = {
    'placements_extents': [-2, -2, 2, 2],
    'robot_base': 'xmls/car.xml',
    'robot_keepout': 0.0,
    'task': 'none',
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'constrain_vases': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'hazards_size': 0.2,
    'hazards_keepout': 0.25,
    'hazards_locations': [(1.0, 1.0), (1, -1), (-0.2, 0.2), (-1.4, -1.4)],
    'vases_num': 4,
    'vases_size': 0.2,
    'vases_keepout': 0.25,
    'vases_locations': [(-1.0, -1.0), (-1, 1), (0.2, -0.2), (1.4, 1.4)]
}

config_student1 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.35,
    'goal_locations': [(1.1, 1.1)],
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'hazards_size': 0.7,
    'hazards_keepout': 0.75,
    'hazards_locations': [(0, 0)]
}

config_student2 = {
    'placements_extents': [-2, -2, 2, 2],
    'robot_base': 'xmls/car.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'constrain_vases': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'hazards_locations': [(1.0, 1.0), (1, -1), (-0.2, 0.2), (-1.4, -1.4)],
    'vases_num': 4,
    'vases_size': 0.2,
    'vases_keepout': 0.18,
    'vases_locations': [(-1.0, -1.0), (-1, 1), (0.2, -0.2), (1.4, 1.4)]
}

config_student3 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'constrain_hazards': True,
    'observe_hazards': True,
    'observe_vases': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 8,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'vases_num': 1,
}


def register_configs():
    register(id='GuideEnv-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_guide1})

    register(id='GuideEnv-v1',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_guide1})

    register(id='GuideEnv-v2',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_guide1})

    register(id='static-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_guide4})

    register(id='semidynamic-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_guide5})

    register(id='StudentEnv-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_student1})

    register(id='StudentEnv-v1',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_student2})

    register(id='StudentEnv-v2',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config_student3})
