#!/usr/bin/env python
import os
import subprocess
import sys
import numpy as np
from sagui.utils.load_utils import load_policy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpi4py import MPI



def get_trajectories(num_eps, num_steps=1000):
    # Load model and environment
    env, get_action, _ = load_policy('data/', itr=4, deterministic=True)
    env.num_steps = num_steps

    results = []

    # Run trajectories
    for _ in range(num_eps):
        o, d, ep_cost = env.reset(), False, 0
        positions = [env.robot_pos]
        while not d:
            a = get_action(o)
            o, _, d, info = env.step(a)
            ep_cost += info.get('cost', 0)
            positions.append(env.robot_pos)
        
        # Save trajectory
        positions = np.array(positions)
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]

        ep = (ep_cost == 0, x_positions, y_positions)
        results.append(ep)

    return results


def mpi_fork(n):
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # Notice: We allow running as root because it is a Docker container!
        args = ["mpirun", "-np", str(n), "--allow-run-as-root"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


# Main
if __name__ == '__main__':
    # Fork using mpi
    num_procs = 8
    mpi_fork(num_procs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    total_eps = 104 # Total number of episodes
    proc_eps =  total_eps // num_procs # Number of episodes per processor

    show_legend = True  # Whether to display the legend
    show_safety = True # Color trajectories based on safety
    show_hazard = True  # Show hazard circle
    show_goal = False   # Show goal circle
    limit_plot = True   # Only show the region (-1.5, -1.5) to (1.5, 1.5)

    # Calculate the results
    results = get_trajectories(proc_eps)

    # Gather results
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten results
        results = [ep for r in all_results for ep in r]

        # Plot each episode
        for ep in results:
            safe, x_positions, y_positions = ep

            if show_safety:
                color = 'blue' if safe else 'black'
                plt.plot(x_positions, y_positions, color=color)
            else:
                plt.plot(x_positions, y_positions)

        if show_legend:
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
        if show_legend:
            plt.legend()

        if limit_plot:
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)

        plt.grid()
        # plt.show()
        plt.savefig('./plot.png')

MPI.Finalize()
