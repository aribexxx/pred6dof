import pandas as pd
import numpy as np

# Read the CSV data
df = pd.read_csv('../data/milis/oscar_all.csv')

# Ensure the data is sorted by timestamp
df = df.sort_values(by='timestamp')

# Calculate the differences in time, position, and orientation
time_diff = df['timestamp'].diff().values / 1e6  # Convert microseconds to seconds

# Calculate position differences
px_diff = df.iloc[::100, :]['px'].diff().values
print(px_diff)
py_diff = df['py'].diff(10).values
pz_diff = df['pz'].diff(10).values

# Calculate orientation differences
qx_diff = df['qx'].diff().values
qy_diff = df['qy'].diff().values
qz_diff = df['qz'].diff().values
qw_diff = df['qw'].diff().values

# # Calculate velocities (m/s) and angular velocities (rad/s)
# velocity_px = px_diff / time_diff
# velocity_py = py_diff / time_diff
# velocity_pz = pz_diff / time_diff

# angular_velocity_qx = qx_diff / time_diff
# angular_velocity_qy = qy_diff / time_diff
# angular_velocity_qz = qz_diff / time_diff
# angular_velocity_qw = qw_diff / time_diff

# # Calculate accelerations (m/s^2) and angular accelerations (rad/s^2)
# acceleration_px = np.diff(velocity_px) / time_diff[1:]
# acceleration_py = np.diff(velocity_py) / time_diff[1:]
# acceleration_pz = np.diff(velocity_pz) / time_diff[1:]

# angular_acceleration_qx = np.diff(angular_velocity_qx) / time_diff[1:]
# angular_acceleration_qy = np.diff(angular_velocity_qy) / time_diff[1:]
# angular_acceleration_qz = np.diff(angular_velocity_qz) / time_diff[1:]
# angular_acceleration_qw = np.diff(angular_velocity_qw) / time_diff[1:]

# print(f"velocity_px is ",velocity_px)
# print(f"velocity_px is ",velocity_py)
# print(f"velocity_px is ",velocity_p)
# Add new columns to the DataFrame
df1 = pd.DataFrame()
df1['px_diff'] = np.append([0],px_diff)  # Pad the first value with 0
# df1['velocity_py'] = np.append([0], velocity_py)
# df1['velocity_pz'] = np.append([0], velocity_pz)

# df['angular_velocity_qx'] = np.append([0], angular_velocity_qx)
# df['angular_velocity_qy'] = np.append([0], angular_velocity_qy)
# df['angular_velocity_qz'] = np.append([0], angular_velocity_qz)
# df['angular_velocity_qw'] = np.append([0], angular_velocity_qw)

# df['acceleration_px'] = np.append([0, 0], acceleration_px)  # Pad the first two values with 0
# df['acceleration_py'] = np.append([0, 0], acceleration_py)
# df['acceleration_pz'] = np.append([0, 0], acceleration_pz)

# df['angular_acceleration_qx'] = np.append([0, 0], angular_acceleration_qx)
# df['angular_acceleration_qy'] = np.append([0, 0], angular_acceleration_qy)
# df['angular_acceleration_qz'] = np.append([0, 0], angular_acceleration_qz)
# df['angular_acceleration_qw'] = np.append([0, 0], angular_acceleration_qw)

# Write the updated DataFrame ssto a new CSV file
df1.to_csv('new.csv', index=False)