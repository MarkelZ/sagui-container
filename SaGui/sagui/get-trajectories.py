#!/usr/bin/env python
from functools import partial
import os
import numpy as np
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# Load model and environment
env, get_action, sess = load_policy('data/', itr=4, deterministic=True)
env.num_steps = 1000

# Run trajectories
num_eps = 10
for i in range(num_eps):
    print('Trajectory: ', i)

    o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = env.reset(), 0, False, 0, 0, 0, 0
    positions = [env.robot_pos]
    while not d:
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

    if (ep_cost == 0):
        plt.plot(x_positions, y_positions, color='blue')
    else:
        plt.plot(x_positions, y_positions, color='black')

# Add a red hazard circle
hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
plt.gca().add_patch(hazard_circle)

# Add dummy trajectories for the legend
plt.plot([], [], color='blue', label='Safe')
plt.plot([], [], color='black', label='Unsafe')

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')
plt.legend()

# plt.legend()

plt.grid()
# plt.show()
plt.savefig('./plot.png')
