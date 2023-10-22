#!/usr/bin/env python
import os
from typing import Callable
import numpy as np
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def modify_constants(env: Engine, coef_dic: dict):
    model = env.model
    for coef, val in coef_dic.items():
        atr = getattr(model, coef)
        for index, _ in np.ndenumerate(atr):
            atr[index] = val


def eval_robust(n, coefs: dict, env: Engine, get_action: Callable[[np.ndarray], np.ndarray]):
    accum_cost = 0
    for _ in range(n):
        o, d, ep_cost = env.reset(), False, 0
        modify_constants(env, coefs)
        while not d:
            a = get_action(o)
            o, _, d, info = env.step(a)
            ep_cost += info.get('cost', 0)

        accum_cost += ep_cost

    return accum_cost / n


def eval_coefs_robust(coef_list: list):
    env, get_action, _ = load_policy('data/', itr=4, deterministic=True)

    res = []
    for coefs in coef_list:
        print(f'Evaluating:\n{coefs}')
        cost = eval_robust(100, coefs, env, get_action)
        v = (coefs, cost)
        res.append(v)

    return res


def plot_trajectories(n):
    # Load from save
    env, get_action, _ = load_policy('data/', itr=4, deterministic=True)

    # Plot trajectories
    for i in range(n):
        print(f'Trajectory {i}')
        o, d, ep_cost = env.reset(), False, 0, 
        positions = [env.robot_pos]
        while not d:
            a = get_action(o)
            o, _, d, info = env.step(a)
            ep_cost += info.get('cost', 0)
            positions.append(env.robot_pos)

        positions = np.array(positions)
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        color = 'blue' if ep_cost == 0 else 'black'
        plt.plot(x_positions, y_positions, color=color)

    # Add dummy trajectories for the legend
    plt.plot([], [], color='blue', label='Safe')
    plt.plot([], [], color='black', label='Unsafe')

    # Draw hazard
    hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
    plt.gca().add_patch(hazard_circle)

    # Plot settings
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Trajectories')
    plt.legend()
    plt.grid()

    # Save plot
    plt.savefig('./plot.png')


# First, plot trajectories to make sure everything works
plot_trajectories(100)

# Number of processes
num_procs = 10

# Create a list of coefficients
coef_list = []
for mass in np.arange(start=0, stop=0.02, step=0.002):
    for fric in np.arange(start=0, stop=0.01, step=0.001):
        coef_dic = {'body_mass' : mass, 'dof_frictionloss' : fric}
        coef_list.append(coef_dic)

# Split the list of coefficients into equal chunks
coef_list = np.array(coef_list)
coef_sublists = np.array_split(coef_list, num_procs)

# Create a thread pool and compute the robustness values
pool: ThreadPool = Pool(num_procs)
print(type(pool))
results = pool.map(eval_coefs_robust, coef_sublists)

# Close the pool and join the threads
pool.close()
pool.join()

# Flatten the results and turn them into a string
res_flat = [x for r in results for x in r]
res_str = '[\n' + ',\n'.join(res_flat) + '\n]'

# Save the results in a text file
with open('./robust_results.txt', 'w') as f:
    f.write(res_str)
