#!/usr/bin/env python
import numpy as np
from sagui.utils.load_utils import load_policy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



# Modify the physics constants of the environment
# WARNING!! THESE VALUES GET RESET WHEN YOU CALL env.reset()
def modify_constants(the_env, coef_dic: dict):
    model = the_env.model
    for coef, val in coef_dic.items():
        atr = getattr(model, coef)
        for index, value in np.ndenumerate(atr):
            atr[index] = value + val


num_eps = 1

show_safety = False # Color trajectories based on safety
show_hazard = True  # Show hazard circle
show_goal = False   # Show goal circle
limit_plot = True   # Only show the region (-1.5, -1.5) to (1.5, 1.5)

accum_cost = 0

# Load model and environment
env, get_action, _ = load_policy('data/', itr=4, deterministic=True)
env.num_steps = 1000

# Always place robot at (1, 1) with angle 0
env.robot_locations = [(1, 1)]
env.robot_rot = 0
env.build_placements_dict()

# Run trajectories
trajectories = []
for i in range(num_eps):
    o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = env.reset(), 0, False, 0, 0, 0, 0
    modify_constants(env, {'body_mass' : 0, 'dof_frictionloss' : 0.005})
    positions = [env.robot_pos]
    print(f'Epoch: {i+1}')
    while not d:
        # a = get_action(o)
        a = [1, -1]
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
    trajectories.append(ep)


# trajectories = []
# for j in range(num_eps):
#     o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = env.reset(), 0, False, 0, 0, 0, 0
#     positions = [env.robot_pos]
#     while not d:
#         o, r, d, info = env.step(get_action(o))
#         ep_ret += r
#         ep_cost += info.get('cost', 0)
#         ep_len += 1
#         ep_goals += 1 if info.get('goal_met', False) else 0
#         positions.append(env.robot_pos)

#     positions = np.array(positions)
#     x_positions = positions[:, 0]
#     y_positions = positions[:, 1]

#     ep = (ep_cost == 0, x_positions, y_positions)
#     trajectories.append(ep)

# Plot each episode
for ep in trajectories:
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
