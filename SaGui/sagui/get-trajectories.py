#Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
#!/usr/bin/env python
from functools import partial
import os
import numpy as np
import tensorflow as tf
import gym
import time
from sagui.utils.logx import EpochLogger
from sagui.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from sagui.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs
from sagui.utils.load_utils_transfer import load_policy_transfer
from safety_gym.envs.engine import Engine
from gym.envs.registration import register

for _ in range(20):
    print('#####################################')
print('Hi! This is the trajectory code stuff')

EPS = 1e-8

config1 = {
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

register(id='GuideEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config1})

config2 = {
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
        'hazards_locations': [(1.0, 1.0),(1,-1),(-0.2,0.2),(-1.4,-1.4)],
        'vases_num': 4,
        'vases_size': 0.2,
        'vases_keepout': 0.18,
        'vases_locations': [(-1.0, -1.0),(-1,1),(0.2,-0.2),(1.4,1.4)]
        }

register(id='GuideEnv-v1',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config2})

config3 = {
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

register(id='GuideEnv-v2',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config3})

config_static = {
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

register(id='static-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_static})


if __name__ == '__main__':
    import json
    import argparse

    # Params for static environment (Appendix G in https://arxiv.org/abs/2307.14316)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='static-v0')
    parser.add_argument('--hid', type=int, default=32)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--local_start_steps', default=500, type=int)
    parser.add_argument('--local_update_after', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fixed_entropy_bonus', default=1.0, type=float)
    parser.add_argument('--entropy_constraint', type=float, default=-1)
    parser.add_argument('--fixed_cost_penalty', default=None, type=float)
    parser.add_argument('--cost_constraint', type=float, default=None)
    parser.add_argument('--cost_lim', type=float, default=5)
    parser.add_argument('--lr_s', type=int, default=50)
    parser.add_argument('--damp_s', type=int, default=10)
    parser.add_argument('--reward_b', type=float, default=1.0)
    parser.add_argument('--logger_kwargs_str', type=json.loads,
                        default='{"output_dir": "./data"}')
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)

    from sagui.utils.run_utils import setup_logger_kwargs
    
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs= args.logger_kwargs_str

    max_ep_len = 1000

    env, get_logp_a, get_action, sess = load_policy_transfer('data_static-v0/', 4)

    teacher_size = env.obs_flat_size
    teacher_keys = env.obs_space_dict.keys()

    for i in range(100):
        print('Trajectory: ', i)

        o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = env.reset(), 0, False, 0, 0, 0, 0
        positions = [env.robot_pos]
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = get_action(o)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_cost += info.get('cost', 0)
            ep_len += 1
            ep_goals += 1 if info.get('goal_met', False) else 0
            positions.append(env.robot_pos)

        # Specify the directory path
        directory_path = './positions/'

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Now, you can safely open and write to the file
        with open(directory_path + 'positions' + str(i) + '.txt', 'w') as f:
            f.write(str(positions))


    print('SUCCESS :)')
