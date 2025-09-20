# Foundations of Machine Learning - Assignment 1

This repository contains solutions for Assignment 1 of the Foundations of Machine Learning course (MTech, Semester 1).

## Folder Structure

```
Assignment 1.pdf
files.sh
Solutions_DA25M020.zip
Solutions_DA25M020/
    Assignment 1.ipynb
    Dataset1.csv
    Dataset2.csv
    kernel_pca.py
    kmeans.py
    pca.py
    spectral_clustering.py
    voronoi.py
    utils.py
    question1.py
    question2.py
    Report.pdf
```

## Contents

- **Assignment 1.pdf**: Assignment instructions.
- **files.sh**: Script to generate starter code and directory structure.
- **Solutions_DA25M020/**: Main solution folder (replace DA25M020 with your roll number).
    - **Dataset1.csv, Dataset2.csv**: Datasets for PCA, Kernel PCA, K-means, and Spectral Clustering.
    - **pca.py**: PCA implementation from scratch.
    - **kernel_pca.py**: Kernel PCA implementation from scratch.
    - **kmeans.py**: K-means clustering implementation.
    - **spectral_clustering.py**: Spectral clustering and alternative mapping.
    - **voronoi.py**: Voronoi diagram plotting for K-means clusters.
    - **utils.py**: Helper functions (data loading, plotting, etc.).
    - **question1.py**: Script for PCA and Kernel PCA experiments.
    - **question2.py**: Script for K-means, Voronoi, and Spectral Clustering experiments.
    - **Assignment 1.ipynb**: (Optional) Jupyter notebook for interactive exploration.
    - **Report.pdf**: Your written analysis and explanations.

## How to Run

1. Install requirements (only `numpy` and `matplotlib` are needed).
2. Run the scripts:
    - For PCA/Kernel PCA:  
      ```sh
      python question1.py
      ```
    - For K-means, Voronoi, Spectral Clustering:  
      ```sh
      python question2.py
      ```

## Notes

- All code is implemented from scratch, without using high-level machine learning libraries.
- See `Report.pdf` for analysis and explanations of results.

---