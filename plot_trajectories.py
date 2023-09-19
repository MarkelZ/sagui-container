import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from matplotlib.patches import Circle

# Create a list to store all the curve data
all_positions = []

# Loop through files in the ./positions/ folder
folder_path = './positions/'
for i in range(32):
    filename = os.path.join(folder_path, f'positions{i}.txt')
    
    with open(filename, 'r') as f:
        lines = ''.join(f.readlines())
        positions = eval(lines)
        
    positions = np.array(positions)
    all_positions.append(positions)

# Create a line plot for each curve
for i, positions in enumerate(all_positions):
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    plt.plot(x_positions, y_positions, label=f'Curve {i}')

# Add a red hazard circle
hazard_circle = Circle((0, 0), 0.7, color='red', label='Hazard')
plt.gca().add_patch(hazard_circle)

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()

