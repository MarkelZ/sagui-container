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
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
from gym.envs.registration import register
import safety_gym
from sagui.utils.run_utils import setup_logger_kwargs
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
mpi_fork(args.cpu)


logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
logger_kwargs= args.logger_kwargs_str

# max_ep_len = 1000
max_ep_len = 30000

# env, get_logp_a, get_action, sess = load_policy_transfer('data_static-v0/', 4)
env, get_action, sess = load_policy('data_static-v0/', 4, deterministic=True)

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
    
    # Plot trajectory
    positions = np.array(positions)
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    plt.plot(x_positions, y_positions, label=f'Curve {i}')

# Add a red hazard circle
hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
plt.gca().add_patch(hazard_circle)

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')

# plt.legend()

plt.grid()
# plt.show()
plt.savefig('./plot.png')

