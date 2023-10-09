import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from matplotlib.patches import Circle


# Lists for safe and unsafe trajectories
safe_positions = []
unsafe_positions = []

# Read ./positions/ file
folder_path = './positions/'
for i in range(100):
    filename = os.path.join(folder_path, f'positions{i}.txt')
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        cost = float(lines[0][:-1])
        positions = eval(lines[1])
        
    positions = np.array(positions)
    if cost == 0:
        safe_positions.append(positions)
    else:
        unsafe_positions.append(positions)

# Plot safe trajectories
for positions in safe_positions:
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    plt.plot(x_positions, y_positions, color='blue')

# Create dummy line for the legend
plt.plot([], [], label='Safe', color='blue')

# Plot unsafe trajectories
for positions in unsafe_positions:
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    plt.plot(x_positions, y_positions, color='black')

# Create dummy line for the legend
plt.plot([], [], label='Unsafe', color='black')

# Add a red hazard circle
hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
plt.gca().add_patch(hazard_circle)

# Add a green goal circle
# goal_circle = Circle((1.1, 1.1), 0.3, color='lime', label='Goal')
# plt.gca().add_patch(goal_circle)


# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)

# Show the plot
plt.legend()
plt.grid()
# plt.show()
plt.savefig('./plot.png')

