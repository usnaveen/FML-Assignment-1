"""
utils.py: Common helper functions for the assignment.
TODO:
- Implement def load_data(file_path): Load 2D dataset from CSV file (assume 1000 points in R^2; use numpy.loadtxt or manual parsing since from scratch, no pandas if not allowed).
- Implement def compute_covariance_matrix(data): Center data (subtract mean), compute covariance as (data.T @ data) / (n-1).
- Implement def compute_eig(matrix): Use numpy.linalg.eig (allowed) to compute eigenvectors and eigenvalues, return sorted descending by eigenvalues.
- Implement def plot_scatter(points, colors=None, title='', xlabel='PC1', ylabel='PC2', file_name=None): Use matplotlib.pyplot.scatter for 2D plots, add labels, save to file if provided.
- Implement def plot_line(x, y, title='', xlabel='Iterations', ylabel='Error', file_name=None): Use matplotlib.pyplot.plot for line graphs (e.g., error vs iterations), save to file.
- Optional: def compute_kernel(x1, x2, kernel_type='linear', degree=3, sigma=1.0): Single kernel computation for shared use (linear: x1@x2, poly: (x1@x2 +1)**degree, rbf: exp(-||x1-x2||^2 / (2*sigma^2))).
Ensure only allowed libraries: import numpy as np, import matplotlib.pyplot as plt.
No other inbuilt computations.
"""
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load 2D dataset from a CSV file."""
    data = np.loadtxt(file_path, delimiter=',')
    return data

def compute_covariance_matrix(data):
    """Compute the covariance matrix of the dataset."""
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    # Compute covariance matrix
    cov_matrix = np.dot(data_centered.T, data_centered) / (data.shape[0] - 1)
    return cov_matrix

def compute_eig(matrix):
    """Compute eigenvalues and eigenvectors of a matrix, sorted in descending order."""
    eigvals, eigvecs = np.linalg.eig(matrix)
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    return eigvals, eigvecs

def plot_scatter(points, colors=None, title='', xlabel='PC1', ylabel='PC2', file_name=None):
    """Plot a scatter plot of 2D points."""
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if file_name:
        plt.savefig(file_name)
    plt.show()

def plot_line(x, y, title='', xlabel='Iterations', ylabel='Error', file_name=None):
    """Plot a line graph."""
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if file_name:
        plt.savefig(file_name)
    plt.show()

def compute_kernel(x1, x2, kernel_type='linear', degree=3, sigma=1.0):
    """Compute the kernel between two vectors."""
    if kernel_type == 'linear':
        return np.dot(x1, x2)
    elif kernel_type == 'polynomial':
        return (np.dot(x1, x2) + 1) ** degree
    elif kernel_type == 'rbf':
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError("Unsupported kernel type. Choose from 'linear', 'polynomial', or 'rbf'.")


