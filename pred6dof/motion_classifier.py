import numpy as np

#TODO: a class that classify different motion patterns by entropy, variance

# Generate example trajectory data
num_points = 20
trajectory = np.random.randint(0, 10, size=num_points)  # Random integer points in the range [0, 10)
trajectory = [1,1,1,1,1,1,1,1,1,1,1,1,1]
print("trajectory:", trajectory)

# Compute the actual entropy
def compute_actual_entropy(trajectory):
    actual_entropy = 0
    for t in range(1, len(trajectory) + 1):
        # Find the shortest non-repeated sub-sequence starting at time t
        shortest_subsequence = len(trajectory) + 1  # Initialize with a large value
        for j in range(t, len(trajectory)):
            if trajectory[j] in trajectory[t:j]:
                break  # Stop if a repeated point is found
            shortest_subsequence = min(shortest_subsequence, j - t + 1)
        actual_entropy += np.log2(shortest_subsequence)
    actual_entropy /= len(trajectory)
    return -actual_entropy

# Compute and print the actual entropy
entropy = compute_actual_entropy(trajectory)
print("Actual Entropy:", entropy)


