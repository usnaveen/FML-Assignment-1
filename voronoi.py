"""
voronoi.py: Plot Voronoi regions for Question 2b.
TODO:
- Implement def plot_voronoi(centers, data_min, data_max, file_name=None):
  - Since from scratch, approximate Voronoi on a grid (scipy.spatial.Voronoi may not be allowed - check; if not, implement grid-based).
  - xx, yy = np.meshgrid(np.linspace(data_min[0], data_max[0], 500), np.linspace(data_min[1], data_max[1], 500))
  - grid = np.c_[xx.ravel(), yy.ravel()]
  - labels = KMeans.predict(grid)  # But need KMeans instance; pass centers.
  - Actually, for given centers, compute dist to each point in grid, assign min dist color.
  - plt.contourf(xx, yy, labels.reshape(xx.shape), cmap='viridis')
  - plt.scatter(centers[:,0], centers[:,1], c='red')
  - Add data points if needed, labels, save to file.
Assume data_min/max from np.min(data, axis=0), np.max(data, axis=0).
No unauthorized libraries.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_voronoi(centers, data_min, data_max, file_name=None, data=None):
    """
    Plot Voronoi regions given cluster centers.
    
    Parameters:
    - centers: numpy array of shape (k, 2) containing cluster centers
    - data_min: numpy array of shape (2,) containing minimum values for each dimension
    - data_max: numpy array of shape (2,) containing maximum values for each dimension
    - file_name: optional string for saving the plot
    - data: optional numpy array of original data points to overlay on the plot
    """
    # Create a grid of points
    xx, yy = np.meshgrid(
        np.linspace(data_min[0], data_max[0], 500),
        np.linspace(data_min[1], data_max[1], 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # For given centers, compute distance to each point in grid, assign min dist color
    dist = np.array([np.linalg.norm(grid - c, axis=1)**2 for c in centers])
    labels = np.argmin(dist, axis=0)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot Voronoi regions
    plt.contourf(xx, yy, labels.reshape(xx.shape), cmap='viridis', alpha=0.3)
    
    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    # Plot original data points if provided
    if data is not None:
        plt.scatter(data[:, 0], data[:, 1], c='black', marker='o', s=20, alpha=0.6)
    
    plt.title('Voronoi Diagram')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    if file_name:
        plt.savefig(file_name)
    plt.show()