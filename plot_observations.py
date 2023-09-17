from numpy import array
import numpy as np
import matplotlib.pyplot as plt

with open('robot_locations.txt', 'r') as f:
    lines = ''.join(f.readlines())
    positions = eval(lines)

positions = np.array(positions)
x_positions = positions[:, 0]
y_positions = positions[:, 1]

# Create a line plot of x and y positions
plt.plot(x_positions, y_positions, label='Robot Trajectory',
         color='blue', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectory')

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()
