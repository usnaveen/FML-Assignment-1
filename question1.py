"""
question1.py: Main script to run Question 1 parts.
TODO:
- Import necessary modules.
- data = utils.load_data('Dataset1.csv')  # Assume CSV with 1000 rows, 2 columns.
- For 1a:
  - pca = PCA(n_components=2)
  - pca.fit(data)
  - print('Variance explained:', pca.explained_variance_ratio())
- For 1b:
  - kernels = [('linear', {}), ('polynomial', {'degree':2}), ('polynomial', {'degree':3}), ('rbf', {'sigma':0.1}), ('rbf', {'sigma':1}), ('rbf', {'sigma':10})]  # Explore
  - for kernel_type, params in kernels:
    - kpca = KernelPCA(n_components=2, kernel_type=kernel_type, **params)
    - kpca.fit(data)
    - projected = kpca.transform(data)
    - file_name = f'q1b_{kernel_type}_{params.get("degree") or params.get("sigma") or ""}.png'
    - utils.plot_scatter(projected, title=f'Kernel PCA: {kernel_type}', file_name=file_name)
- For 1c: Manually analyze plots, choose best (e.g., one that shows clear separation), explain in Report.pdf.
Run: python question1.py to generate prints and plots.
"""
import numpy as np
from utils import load_data, plot_scatter
from pca import PCA
from kernel_pca import KernelPCA

if __name__ == "__main__":
    # Load data
    data = load_data('Dataset1.csv')

    # Question 1a: PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    print('Variance explained by each principal component:', pca.explained_variance_ratio())

    # Question 1b: Kernel PCA with different kernels
    kernels = [
        ('linear', {}),
        ('polynomial', {'degree': 2}),
        ('polynomial', {'degree': 3}),
        ('rbf', {'sigma': 0.1}),
        ('rbf', {'sigma': 1}),
        ('rbf', {'sigma': 10})
    ]

    for kernel_type, params in kernels:
        kpca = KernelPCA(n_components=2, kernel_type=kernel_type, **params)
        kpca.fit(data)
        projected = kpca.transform(data)
        param_str = '_'.join(f'{k}{v}' for k, v in params.items())
        file_name = f'q1b_{kernel_type}_{param_str}.png' if param_str else f'q1b_{kernel_type}.png'
        plot_scatter(projected, title=f'Kernel PCA: {kernel_type} {params}', file_name=file_name)
        