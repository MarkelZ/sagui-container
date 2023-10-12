#!/usr/bin/env python
import numpy as np
from sagui.utils.load_utils import load_policy
from safety_gym.envs.engine import Engine
from safety_gym.envs.world import World, Robot


def eval_cost(num_iters=20):
    accum_cost = 0

    for i in range(num_iters):
        print('Trajectory ' + str(i))
        o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = env.reset(), 0, False, 0, 0, 0, 0
        while not d:
            a = get_action(o)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_cost += info.get('cost', 0)
            ep_len += 1
            ep_goals += 1 if info.get('goal_met', False) else 0

        accum_cost += ep_cost

    return accum_cost / num_iters


# Type hints
env: Engine
robot: Robot

# Load model and environment
env, get_action, _ = load_policy('data/', itr=4, deterministic=True)

robot = env.robot
sim = robot.sim
model = sim.model

print()
print(sim.model.body_mass)        # Mass
print(sim.model.dof_frictionloss) # Degree of freedom friction loss
print(sim.model.geom_friction)    # Geometry friction loss

n = len(sim.model.body_mass)
for i in range(n):
    sim.model.body_mass[i] *= 100

n = len(sim.model.dof_frictionloss)
for i in range(n):
    sim.model.dof_frictionloss[i] *= 0

n = len(sim.model.geom_friction)
for i in range(n):
    m = len(sim.model.geom_friction[i])
    for j in range(m):
        sim.model.geom_friction[i, j] *= 0

print()
print(sim.model.body_mass)        # Mass
print(sim.model.dof_frictionloss) # Degree of freedom friction loss
print(sim.model.geom_friction)    # Geometry friction loss

print(eval_cost())
