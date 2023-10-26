#!/usr/bin/env python
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register



# Modify the physics constants of the environment
def modify_constants(the_env, coef_dic: dict):
    model = the_env.model
    for coef, val in coef_dic.items():
        atr = getattr(model, coef)
        for index, value in np.ndenumerate(atr):
            atr[index] = value + val


# Create and regist an env configuration
config = {
    'robot_placements' : [(0, 0, 0, 0)],
    'robot_rot' : 0,
    'robot_base': 'xmls/point.xml',
    'robot_keepout': 0.0,
    'task': 'none',
}
register(id='plot-v0',
        entry_point='safety_gym.envs.mujoco:Engine',
        kwargs={'config': config})

# Make the environment with the registered configuration
env = gym.make('plot-v0')
env.num_steps = 1000

# Create a 4x4 grid of subplots
size = 4
fig, axs = plt.subplots(size, size, figsize=(10, 10))

# Run trajectories
for i, mass in enumerate(np.linspace(0.0001, 0.03, size)):
    for j, fric in enumerate(np.linspace(0, 0.008, size)):
        print(f'Mass: {mass}; Fric: {fric}')

        # Run trajectory
        d = False
        env.reset()
        modify_constants(env, {'body_mass' : mass, 'dof_frictionloss' : fric})
        positions = [env.robot_pos]
        while not d:
            a = [1, 1]
            _, _, d, info = env.step(a)
            positions.append(env.robot_pos)

        # Plot trajectory
        positions = np.array(positions)
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        ax = axs[i, j]
        ax.plot(x_positions, y_positions)
        ax.set_title(f'Mass={"{:.4f}".format(mass)}; Fric={"{:.4f}".format(fric)}')

        # Add labels and title
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        # plt.title('Robot Trajectories')

        # plt.xlim(0.5, 1.5)
        # plt.ylim(0.2, 1.2)

        # plt.grid()
        # # plt.show()
        # plt.savefig('./plot.png')

plt.tight_layout()
plt.savefig('./plot.png')
