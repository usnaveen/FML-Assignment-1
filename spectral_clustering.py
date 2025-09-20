import numpy as np

def compute_kernel_matrix(data, kernel_type='linear', **kernel_params):
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

class KernelPCA:
    
    
    def __init__(self, n_components=2, kernel_type='linear', **kernel_params):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.alphas = None
        self.eigenvalues = None
        self.X_fit = None
        
    def fit(self, X):
        
        self.X_fit = X
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = compute_kernel_matrix(X, self.kernel_type, **self.kernel_params)
        
        # Center the kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only top n_components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components]
        
        # Normalize eigenvectors by eigenvalues
        for i in range(self.n_components):
            if self.eigenvalues[i] > 0:
                self.alphas[:, i] = self.alphas[:, i] / np.sqrt(self.eigenvalues[i])
        
        return self

def spectral_clustering(data, k=4, kernel_type='rbf', sigma=1.0):
    # Import KMeans here to avoid circular imports
    from kmeans import KMeans
    
    # Use KernelPCA to get eigenvectors (top-k)
    kpca = KernelPCA(n_components=k, kernel_type=kernel_type, sigma=sigma)
    kpca.fit(data)
    evecs = kpca.alphas  # Top-k eigenvectors
    
    # Normalize rows: evecs = evecs / np.linalg.norm(evecs, axis=1)[:, np.newaxis]
    row_norms = np.linalg.norm(evecs, axis=1)
    # Avoid division by zero
    row_norms[row_norms == 0] = 1
    evecs = evecs / row_norms[:, np.newaxis]
    
    # Apply K-means on the normalized eigenvectors
    km = KMeans(k=k)
    km.fit(evecs)
    
    return km.labels

def alternative_mapping(eigenvectors, k):
    # For each data point i, label = np.argmax(eigenvectors[i, :k])
    labels = np.argmax(eigenvectors[:, :k], axis=1)
    return labels