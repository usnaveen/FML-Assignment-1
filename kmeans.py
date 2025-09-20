"""
kmeans.py: Implementation of K-means from scratch for Question 2a,2b.
TODO:
- Implement class KMeans:
  - def __init__(self, k, max_iters=100, tol=1e-4):
    self.k = k
    self.max_iters = max_iters
    self.tol = tol
    self.centers = None
    self.labels = None
    self.error_history = []
  - def fit(self, data, random_state=None):
    - np.random.seed(random_state) if random_state
    - Initialize centers: self.centers = data[np.random.choice(len(data), self.k, replace=False)]
    - For iter in range(max_iters):
      - Compute distances: dist = np.array([np.linalg.norm(data - c, axis=1)**2 for c in self.centers])
      - self.labels = np.argmin(dist, axis=0)
      - error = np.sum(np.min(dist, axis=0))
      - self.error_history.append(error)
      - new_centers = np.array([data[self.labels == i].mean(axis=0) if np.sum(self.labels==i)>0 else self.centers[i] for i in range(self.k)])
      - if np.all(np.abs(new_centers - self.centers) < tol): break
      - self.centers = new_centers
  - def predict(self, data): 
    - Compute dist to self.centers, return argmin
For 2a: Multiple random inits (different random_state), fit, get error_history, plot.
No unauthorized libraries.
"""
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        self.labels = None
        self.error_history = []

    def fit(self, data, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize centers by randomly selecting k unique points from the data
        self.centers = data[np.random.choice(len(data), self.k, replace=False)]
        
        for iter in range(self.max_iters):
            # Compute distances from data points to cluster centers
            dist = np.array([np.linalg.norm(data - c, axis=1)**2 for c in self.centers])
            # Assign labels based on closest center
            self.labels = np.argmin(dist, axis=0)
            # Compute current error (sum of squared distances to closest center)
            error = np.sum(np.min(dist, axis=0))
            self.error_history.append(error)
            
            # Update centers
            new_centers = np.array([
                data[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0 else self.centers[i]
                for i in range(self.k)
            ])
            
            # Check for convergence
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break
            
            self.centers = new_centers

    def predict(self, data):
        dist = np.array([np.linalg.norm(data - c, axis=1)**2 for c in self.centers])
        return np.argmin(dist, axis=0)
    
    def get_error_history(self):
        return self.error_history
    
    def get_centers(self):
        return self.centers
    
    def get_labels(self):
        return self.labels
    
    def get_inertia(self):
        if self.labels is None or self.centers is None:
            raise ValueError("Model has not been fitted yet.")
        dist = np.array([np.linalg.norm(data - c, axis=1)**2 for c in self.centers])
        return np.sum(np.min(dist, axis=0))
    
def compute_inertia(data, labels, centers):
    inertia = 0.0
    for i in range(len(centers)):
        cluster_points = data[labels == i]
        inertia += np.sum((cluster_points - centers[i]) ** 2)
    return inertia 

def computer_kernel_matrix(data, kernel_type='linear', **kernel_params):
    n = data.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if kernel_type == 'linear':
                K[i, j] = np.dot(data[i], data[j])
            elif kernel_type == 'polynomial':
                degree = kernel_params.get('degree', 3)
                K[i, j] = (np.dot(data[i], data[j]) + 1) ** degree
            elif kernel_type == 'rbf':
                sigma = kernel_params.get('sigma', 1.0)
                K[i, j] = np.exp(-np.linalg.norm(data[i] - data[j]) ** 2 / (2 * sigma ** 2))
    return K


def run_multiple_inits(data, k, n_inits=5, max_iters=100, tol=1e-4):
  all_error_histories = []
  for i in range(n_inits):
    kmeans = KMeans(k=k, max_iters=max_iters, tol=tol)
    kmeans.fit(data, random_state=i)
    all_error_histories.append(kmeans.get_error_history())
  return all_error_histories

def plot_error_histories(error_histories):
  plt.figure(figsize=(8, 5))
  for idx, history in enumerate(error_histories):
    plt.plot(history, label=f'Init {idx+1}')
  plt.xlabel('Iteration')
  plt.ylabel('Error (Inertia)')
  plt.title('K-Means Error History for Multiple Random Initializations')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
