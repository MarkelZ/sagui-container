import matplotlib.pyplot as plt
from pathlib import Path


# Path to saves
with open('./robust_results.txt') as f:
    lines = f.readlines()

# Reconstruct values dictionary from source
source = ''.join(lines)
entries = eval(source)
entries.sort(reverse=True, key=lambda v:v[1])
print('Worst case dynamics:')
print(entries[0])
print(entries[1])
print(entries[2])

# Extract x and y coordinates and values from the dictionary
x = [v[0]['body_mass'] for v in entries]
y = [v[0]['dof_frictionloss'] for v in entries]
values = [v[1] for v in entries]

# Create the heatmap
plt.scatter(x, y, c=values, cmap='viridis', vmin=0, vmax=40, s=500)
plt.colorbar()
plt.xlabel('Body mass')
plt.ylabel('DOF friction')
plt.title('Avg. cost of the deterministic policy')

# Adjust plot to fit 
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
margin = 0.001
plt.xlim(x_min-margin, x_max+margin)
plt.ylim(y_min-margin, y_max+margin)

# plt.show()
plt.savefig('plot.png')
