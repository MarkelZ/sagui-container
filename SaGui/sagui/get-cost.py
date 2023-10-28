#!/usr/bin/env python
import os
import subprocess
import sys
import numpy as np
from sagui.utils.load_utils import load_policy
from mpi4py import MPI



def modify_constants(the_env, coef_dic: dict):
    model = the_env.model
    for coef, val in coef_dic.items():
        atr = getattr(model, coef)
        for index, _ in np.ndenumerate(atr):
            atr[index] = val


def get_cum_cost(num_eps, coef_dic, num_steps=1000):
    # Load model and environment
    env, get_action, _ = load_policy('data/', itr=4, deterministic=False)
    env.num_steps = num_steps

    cum_cost = 0
    # Run trajectories
    for _ in range(num_eps):
        o, d, ep_cost = env.reset(), False, 0
        modify_constants(env, coef_dic)
        while not d:
            a = get_action(o)
            o, _, d, info = env.step(a)
            ep_cost += info.get('cost', 0)
        
        # Accumulate cost
        cum_cost += ep_cost
        print(ep_cost)

    return cum_cost


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

    total_eps = 400 # Total number of episodes
    proc_eps =  total_eps // num_procs # Number of episodes per processor

    # Calculate the results
    cum_cost = get_cum_cost(proc_eps, {'body_mass': 1e-8, 'dof_frictionloss': 1e-8})

    # Gather results
    all_cum_costs = comm.gather(cum_cost, root=0)

    if rank == 0:
        # Calculate the average cost
        total_cost = sum(all_cum_costs)
        avg_cost = total_cost / (num_procs * proc_eps)
        print(f'Averge cost: {avg_cost}')

MPI.Finalize()
