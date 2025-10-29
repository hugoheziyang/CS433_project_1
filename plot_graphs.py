import numpy as np
import matplotlib.pyplot as plt

### Plot the explained variance ratio vs number of PCA components

def plot_explained_variance(pca_model):
    """
    Plot the explained variance ratio vs number of PCA components.
    Args:
        pca_model: dict returned by pca_fit function
    """
    explained_variance_ratio = pca_model["explained_variance_ratio"]
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of PCA Components (k)')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of PCA Components')
    plt.grid()
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')
    plt.legend()
    plt.show()