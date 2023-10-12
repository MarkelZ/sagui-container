#!/usr/bin/env python
import numpy as np
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing.dummy import Pool
from time import time


def get_trajectories(num_eps):
    # Load model and environment
    env, get_action, _ = load_policy('data/', itr=4, deterministic=True)
    env.num_steps = 1000

    results = []

    # Run trajectories
    for i in range(num_eps):
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
        
        # Save trajectory
        positions = np.array(positions)
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        ep = (ep_cost == 0, x_positions, y_positions)
        results.append(ep)

    return results



num_procs = 2  # Number of processors
total_eps = 50 # Total number of episodes
proc_eps =  total_eps // num_procs # Number of episodes per processor

show_safety = False # Color trajectories based on safety
show_hazard = True  # Show hazard circle
show_goal = False   # Show goal circle
limit_plot = True   # Only show the region (-1.5, -1.5) to (1.5, 1.5)

# Measure start time
t0 = time()

# Use multithreading to compute the trajectories
pool = Pool(num_procs)
results = pool.map(get_trajectories, [proc_eps for _ in range(num_procs)])

# Close the pool and wait until done
pool.close()
pool.join()

# Flatten the results
results = [ep for r in results for ep in r ]

# Plot each episode
for ep in results:
    safe, x_positions, y_positions = ep

    if show_safety:
        color = 'blue' if safe else 'black'
        plt.plot(x_positions, y_positions, color=color)
    else:
        plt.plot(x_positions, y_positions)


# Add dummy trajectories for the legend
plt.plot([], [], color='blue', label='Safe')
plt.plot([], [], color='black', label='Unsafe')

# Add a red hazard circle
if show_hazard:
    hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
    plt.gca().add_patch(hazard_circle)

# Add a green goal circle
if show_goal:
    goal_circle = Circle((1.1, 1.1), 0.3, color='green', label='Goal')
    plt.gca().add_patch(goal_circle)

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')
plt.legend()

if limit_plot:
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

plt.grid()
# plt.show()
plt.savefig('./plot.png')

t = time() - t0
print('Total time elapsed: ' + '{:.1f}'.format(t) + ' seconds')
print('Num. trajectories: ' + str(num_procs * proc_eps))
