"""
question2.py: Main script to run Question 2 parts.
TODO:
- Import necessary modules.
- data = utils.load_data('Dataset2.csv')
- data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
- For 2a:
  - for init in range(5):
    - km = KMeans(k=4)
    - km.fit(data, random_state=init)
    - utils.plot_line(range(len(km.error_history)), km.error_history, title=f'Error vs Iters Init {init}', file_name=f'q2a_error_init{init}.png')
    - utils.plot_scatter(data, colors=km.labels, title=f'Clusters Init {init}', file_name=f'q2a_clusters_init{init}.png')
- For 2b:
  - fixed_seed = 0  # Fix init
  - for k in [2,3,4,5]:
    - km = KMeans(k=k)
    - km.fit(data, random_state=fixed_seed)
    - voronoi.plot_voronoi(km.centers, data_min, data_max, file_name=f'q2b_voronoi_k{k}.png')
- For 2c:
  - kernel_type = 'rbf'  # Choose based on data/experiments
  - sigma = 1.0  # Tune
  - labels = spectral_clustering(data, k=4, kernel_type=kernel_type, sigma=sigma)
  - utils.plot_scatter(data, colors=labels, title='Spectral Clustering', file_name='q2c.png')
  - Explain choice in Report.pdf
- For 2d:
  - kpca = KernelPCA(n_components=4, kernel_type=kernel_type, sigma=sigma)
  - kpca.fit(data)
  - evecs = kpca.alphas  # Top-4 eigenvectors
  - labels_alt = alternative_mapping(evecs, k=4)
  - utils.plot_scatter(data, colors=labels_alt, title='Alternative Mapping', file_name='q2d.png')
  - Compare to 2c in Report.pdf
Run: python question2.py to generate plots.
"""
import numpy as np
import utils
from kmeans import KMeans
import voronoi
from spectral_clustering import spectral_clustering, alternative_mapping
from kernel_pca import KernelPCA

def main():
    print("Starting Question 2 Analysis...")
    
    # Load the data
    print("Loading data...")
    data = utils.load_data('Dataset2.csv')
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    print(f"Data shape: {data.shape}")
    print(f"Data range: min={data_min}, max={data_max}")
    
    # Question 2a: K-means with different random initializations
    print("\n=== Question 2a: K-means with different initializations ===")
    for init in range(5):
        print(f"Running K-means with initialization {init}...")
        km = KMeans(k=4)
        km.fit(data, random_state=init)
        
        # Plot error history
        utils.plot_line(
            range(len(km.error_history)), 
            km.error_history, 
            title=f'Error vs Iterations (Init {init})', 
            xlabel='Iterations',
            ylabel='Error (Inertia)',
            file_name=f'q2a_error_init{init}.png'
        )
        
        # Plot clusters
        utils.plot_scatter(
            data, 
            colors=km.labels, 
            title=f'K-means Clusters (Init {init})', 
            file_name=f'q2a_clusters_init{init}.png'
        )
        
        print(f"  Final error: {km.error_history[-1]:.2f}")
    
    # Question 2b: Voronoi diagrams for different k values
    print("\n=== Question 2b: Voronoi diagrams for different k values ===")
    fixed_seed = 0  # Fix initialization
    for k in [2, 3, 4, 5]:
        print(f"Running K-means with k={k}...")
        km = KMeans(k=k)
        km.fit(data, random_state=fixed_seed)
        
        # Plot Voronoi diagram
        voronoi.plot_voronoi(
            km.centers, 
            data_min, 
            data_max, 
            file_name=f'q2b_voronoi_k{k}.png',
            data=data
        )
        
        print(f"  Final error for k={k}: {km.error_history[-1]:.2f}")
    
    # Question 2c: Spectral clustering
    print("\n=== Question 2c: Spectral clustering ===")
    # Choose RBF kernel based on the spiral nature of the data
    kernel_type = 'rbf'  # RBF is good for non-linear, circular patterns
    sigma = 1.0  # May need tuning based on data scale
    
    print(f"Running spectral clustering with {kernel_type} kernel (sigma={sigma})...")
    labels = spectral_clustering(data, k=4, kernel_type=kernel_type, sigma=sigma)
    
    # Plot spectral clustering results
    utils.plot_scatter(
        data, 
        colors=labels, 
        title='Spectral Clustering (RBF Kernel)', 
        file_name='q2c.png'
    )
    
    print(f"  Spectral clustering completed. Unique labels: {np.unique(labels)}")
    
    # Question 2d: Alternative mapping
    print("\n=== Question 2d: Alternative mapping ===")
    print("Running Kernel PCA and alternative mapping...")
    kpca = KernelPCA(n_components=4, kernel_type=kernel_type, sigma=sigma)
    kpca.fit(data)
    evecs = kpca.alphas  # Top-4 eigenvectors
    
    labels_alt = alternative_mapping(evecs, k=4)
    
    # Plot alternative mapping results
    utils.plot_scatter(
        data, 
        colors=labels_alt, 
        title='Alternative Mapping (Argmax of Eigenvectors)', 
        file_name='q2d.png'
    )
    
    print(f"  Alternative mapping completed. Unique labels: {np.unique(labels_alt)}")
    
    # Compare the two approaches
    print("\n=== Comparison between 2c and 2d ===")
    print("Spectral clustering (2c) uses K-means on normalized eigenvectors")
    print("Alternative mapping (2d) uses argmax on eigenvector components")
    
    # Calculate how many points have the same label in both methods
    same_labels = np.sum(labels == labels_alt)
    total_points = len(labels)
    agreement = same_labels / total_points * 100
    print(f"Agreement between methods: {agreement:.1f}% ({same_labels}/{total_points} points)")
    
    print("\nAll plots have been saved. Check the generated PNG files.")
    print("Analysis complete!")

if __name__ == "__main__":
    main()