import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from matplotlib.patches import Ellipse

def calculate_mahalanobis_distance(point, data):
    """Calculate the Mahalanobis distance of a point from a dataset."""
    data = np.array(data)
    point = np.array(point)
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    # Add a small value to the diagonal to regularize the covariance matrix
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return mahalanobis(point, mean, inv_cov_matrix)

def plot_covariance_ellipse(mean, cov_matrix, ax, n_std=2.0, **kwargs):
    """Plot an ellipse representing the covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def plot_covariance_ellipses(mean, cov_matrix, ax, n_std_list=[1, 2, 3], **kwargs):
    """Plot multiple ellipses representing different Mahalanobis distance thresholds."""
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    for n_std in n_std_list:
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

# Rename variables for clarity
# Update the dataset to include even more diverse points
dataset_points = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 2], [6, 3], [7, 8], [8, 1], [9, 5], [10, 6],
    [11, 7], [12, 8], [13, 9], [14, 10], [15, 5], [16, 4], [17, 3], [18, 2], [19, 1], [20, 0]
])

# Define even more test points
test_points = [
    [5, 6], [3, 3], [6, 7], [2, 1], [8, 9], [7, 2], [10, 10], [1, 1],
    [12, 12], [14, 14], [16, 16], [18, 18], [20, 20], [0, 0], [5, 15], [10, 5]
]

# Recalculate distances for all test points
distances = [calculate_mahalanobis_distance(pt, dataset_points) for pt in test_points]

# Update references in the visualization
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Plot the dataset points
for i, point in enumerate(dataset_points):
    plt.scatter(*point, color='blue', label='Dataset Points' if i == 0 else "")

# Plot the test points and annotate distances
for i, (pt, dist) in enumerate(zip(test_points, distances)):
    plt.scatter(*pt, color='red', label='Test Points' if i == 0 else "")
    plt.text(pt[0], pt[1], f"{dist:.2f}", fontsize=10, color='purple', ha='center')

# Plot multiple covariance ellipses
mean = np.mean(dataset_points, axis=0)
cov_matrix = np.cov(dataset_points, rowvar=False)

# Update the legend to include values for each covariance ellipse with unique colors
ellipse_colors = ['green', 'blue', 'purple']
for idx, n_std in enumerate([1, 2, 3]):
    plt.plot([], [], color=ellipse_colors[idx], linestyle='--', label=f'{n_std}-std Covariance Ellipse')

# Update the plot_covariance_ellipses function call to use unique colors
for idx, n_std in enumerate([1, 2, 3]):
    plot_covariance_ellipse(mean, cov_matrix, ax, n_std=n_std, edgecolor=ellipse_colors[idx], facecolor='none', linestyle='--')

# Overlay eigenvectors and update the legend to include eigenvalues for all eigenvectors
eigvals, eigvecs = np.linalg.eigh(cov_matrix)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
for i in range(len(eigvals)):
    vec = eigvecs[:, order[i]] * np.sqrt(eigvals[i]) * 3  # Scale for visualization
    plt.quiver(mean[0], mean[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='orange',
               label=f'Eigenvector {i+1} (Eigenvalue: {eigvals[i]:.2f})')

# Add labels and legend
plt.title("Dataset with Eigenvectors and Covariance Ellipses")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

# Show the combined plot
plt.show()
