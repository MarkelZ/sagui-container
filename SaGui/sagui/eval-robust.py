#!/usr/bin/env python
import os
import subprocess
import sys
from typing import Callable
import numpy as np
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
from mpi4py import MPI


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


def eval_coefs_robust(coef_list: list, rank: int):
    env, get_action, _ = load_policy('data/', itr=4, deterministic=True)

    res = []
    for coefs in coef_list:
        print(f'Proc. {rank} with: {coefs}')
        cost = eval_robust(100, coefs, env, get_action)
        v = (coefs, cost)
        res.append(v)

    return res


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

    # Create a list of coefficients
    coef_list = []
    eps = 1e-9
    for mass in np.arange(start=eps, stop=0.02+eps, step=0.002):
        for fric in np.arange(start=0, stop=0.01, step=0.001):
            coef_dic = {'body_mass': mass, 'dof_frictionloss': fric}
            coef_list.append(coef_dic)

    # Split the list of coefficients into equal chunks
    coef_list = np.array(coef_list)
    coef_sublists = np.array_split(coef_list, num_procs)

    # Select corresponding chunk of data
    coefs_chunk = coef_sublists[rank]

    # Calculate the results
    results = eval_coefs_robust(coefs_chunk, rank)

    # Gather results
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten the results and turn them into a string
        res_flat = [x for r in all_results for x in r]
        res_str = '[\n' + ',\n'.join(res_flat) + '\n]'

        # Save the results in a text file
        with open('./robust_results.txt', 'w') as f:
            f.write(res_str)

    MPI.Finalize()
