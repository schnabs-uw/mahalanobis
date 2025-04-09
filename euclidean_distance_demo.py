import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Define two points
point_a = (1, 2)
point_b = (4, 6)

# Calculate the Euclidean distance
distance = euclidean_distance(point_a, point_b)

# Print the result
print(f"The Euclidean distance between {point_a} and {point_b} is {distance:.2f}")

# Visualization
plt.figure(figsize=(8, 6))

# Plot the points
plt.scatter(*point_a, color='red', label=f'Point A {point_a}')
plt.scatter(*point_b, color='blue', label=f'Point B {point_b}')

# Plot the line connecting the points
plt.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color='green', linestyle='--', label='Distance Line')

# Annotate the distance
midpoint = ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)
plt.text(midpoint[0], midpoint[1], f"{distance:.2f}", fontsize=12, color='purple', ha='center')

# Add labels and legend
plt.title("Euclidean Distance Demonstration")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()

# Show the plot
plt.show()
