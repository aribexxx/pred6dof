import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial arrow
arrow_length = 1.0
arrow = np.array([[0, 0, 0], [arrow_length, 0, 0]]).T

# Function to update the arrow based on the rotation matrix
def update_arrow(rotation_matrix):
    transformed_arrow = rotation_matrix @ arrow
    return transformed_arrow

# Plot the initial arrow
line, = ax.plot([0, arrow_length], [0, 0], [0, 0], lw=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Sample quaternion data stream
quaternion_data = [
    [0, 0, 0, 1],
    [0.707, 0, 0, 0.707],
    [0.5, 0.5, 0.5, 0.5],
    [0, 0.707, 0, 0.707],
    # Add more quaternion data as needed
]

def update(frame):
    q = quaternion_data[frame % len(quaternion_data)]
    rotation_matrix = quaternion_to_rotation_matrix(q)
    transformed_arrow = update_arrow(rotation_matrix)
    line.set_data(transformed_arrow[0, :], transformed_arrow[1, :])
    line.set_3d_properties(transformed_arrow[2, :])
    return line,

ani = FuncAnimation(fig, update, frames=len(quaternion_data), interval=500, blit=True)

plt.show()
