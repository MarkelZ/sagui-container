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


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    logp_a = gaussian_likelihood(a, mu, log_std)
    return mu, pi, logp_pi, logp_a

def apply_squashing_func(mu, pi, a, logp_pi, logp_a):
    # Adjustment to log prob
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
    logp_a -= tf.reduce_sum(2*(np.log(2) - a - tf.nn.softplus(-2*a)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi, logp_a


"""
Actors and Critics
"""
def mlp_actor(x, a, name='pi', hidden_sizes=(64,64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi, logp_a = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi, logp_a = apply_squashing_func(mu, pi, a, logp_pi, logp_a)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi, logp_a


def mlp_critic(x, a, pi, name, hidden_sizes=(64,64), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x,a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return critic, critic_pi


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])


if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    # # parser.add_argument('--env', type=str, default='GuideEnv-v2')
    # parser.add_argument('--env', type=str, default='GuideEnv-v0')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--epochs', type=int, default=2)
    # parser.add_argument('--exp_name', type=str, default='sac')
    # # parser.add_argument('--steps_per_epoch', type=int, default=30000)
    # parser.add_argument('--steps_per_epoch', type=int, default=1000)
    # parser.add_argument('--update_freq', type=int, default=100)
    # parser.add_argument('--cpu', type=int, default=1)
    # # parser.add_argument('--render', default=False, action='store_true')
    # parser.add_argument('--render', default=False, action='store_true')
    # parser.add_argument('--local_start_steps', default=500, type=int)
    # parser.add_argument('--local_update_after', default=500, type=int)
    # parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--fixed_entropy_bonus', default=1.0, type=float)
    # parser.add_argument('--entropy_constraint', type=float, default= -1)
    # parser.add_argument('--fixed_cost_penalty', default=None, type=float)
    # parser.add_argument('--cost_constraint', type=float, default=None)
    # parser.add_argument('--cost_lim', type=float, default=20)
    # parser.add_argument('--lr_s', type=int, default=50)
    # parser.add_argument('--damp_s', type=int, default=10)
    # parser.add_argument('--reward_b', type=float, default=1.0)
    # parser.add_argument('--logger_kwargs_str', type=json.loads, default='{"output_dir": "./data"}')
    # args = parser.parse_args()

    # Params for static environment (Appendix G in https://arxiv.org/abs/2307.14316)
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
