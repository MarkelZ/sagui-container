import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# Path to saves
path = Path('./robust_positions/')

entries = {}
# Iterate through the positions directories
for dir in path.iterdir():
    if not dir.is_dir():
        continue

    # Find the parameter value pairs
    params = dir.name.split(';')[1:]
    param_vals = [float(p.split('=')[1]) for p in params]

    # Iterate through each file in the directory
    # to sum up all the cost values of the trajectories
    total_cost = 0
    count = 0
    for item in dir.iterdir():
        if not item.is_file():
            continue

        # The first line contains the cost of the trajectory
        with open(item, 'r') as file:
            cost = file.readline()[:-1]

        total_cost += float(cost)
        count += 1

    # Average cost
    avg_cost = total_cost / count

    # Add new entry for the heatmap
    entries[(param_vals[0], param_vals[1])] = avg_cost


# Extract x and y coordinates and values from the dictionary
x = [k[0] for k in entries.keys()]
y = [k[1] for k in entries.keys()]
values = list(entries.values())

# Create the heatmap
plt.scatter(x, y, c=values, cmap='viridis', s=500)
plt.colorbar()
plt.xlabel('Body mass')
plt.ylabel('DOF friction')
plt.title('Average cost of the deterministic policy')

# plt.show()
plt.savefig('plot.png')
