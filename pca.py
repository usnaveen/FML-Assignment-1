"""
pca.py: Implementation of PCA algorithm from scratch for Question 1a.
TODO:
- Implement class PCA:
  - def __init__(self, n_components=2):
    self.n_components = n_components
    self.components = None
    self.explained_variance = None
  - def fit(self, data): 
    - Use utils.compute_covariance_matrix(data) to get cov_matrix.
    - Use utils.compute_eig(cov_matrix) to get eigvals, eigvecs (sorted descending).
    - Set self.components = eigvecs[:, :self.n_components]
    - Set self.explained_variance = eigvals / np.sum(eigvals)
  - def transform(self, data):
    - Center data.
    - Return data @ self.components
  - def explained_variance_ratio(self): Return self.explained_variance[:self.n_components]
For 1a: After fit, print explained_variance_ratio for each PC (since data in R^2, expect 2 values).
No unauthorized libraries.
"""
import numpy as np
from utils import compute_covariance_matrix, compute_eig

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.explained_variance = None

    def fit(self, data):
        # Compute covariance matrix
        cov_matrix = compute_covariance_matrix(data)
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = compute_eig(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        # Select the top n_components
        self.components = eigvecs[:, :self.n_components]
        self.explained_variance = eigvals / np.sum(eigvals)

    def transform(self, data):
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        # Project the data onto principal components
        return np.dot(data_centered, self.components)

    def explained_variance_ratio(self):
        return self.explained_variance[:self.n_components]
