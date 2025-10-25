import numpy as np

### Standardization

def standardize_fit(X):
    """
    Fit per-feature mean and std on training data X (N x D).
    Returns (mean, std), where std zeros are replaced by 1 to avoid division by zero.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)  # unbiased std
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe

def standardize_transform(X, mean, std):
    """Apply training mean/std to any data (train/val/test)."""
    return (X - mean) / std



### PCA via SVD

def pca_fit(X, k):
    """
    Fit PCA on training data X (do not add column of 1s in X).
    Uses economy SVD: Xc = U S V^T, principal axes are columns of V.
    Returns a dict with components, mean, explained_variance, explained_variance_ratio.
    """
    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD; works for both N>=D and N<D
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Principal directions (D x k), each column is a component
    components = Vt[:k].T

    # Explained variance of each PC; same as eigenvalues of covariance matrix Sigma = (Xc^T Xc) / (N - 1)
    # Var along pc_i = S_i^2 / (N - 1)
    N = X.shape[0]
    eigvals = (S**2) / max(N - 1, 1)    # divide by N - 1 instead of N to get unbiased estimate 
    explained_variance = eigvals[:k]
    explained_variance_ratio = explained_variance / eigvals.sum()

    return {
        "mean": mean,
        "components": components,              # shape: (D, k)
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }

def pca_transform(X, pca_model):
    """
    Project any data onto the k PCs learned from training data.
    Args:
        X: data to project, shape (N, D)
        pca_model: dict returned by pca_fit, i.e. input pca_model = pca_fit(X, k)
    Returns:
        X_pca: projected data, shape (N, k)
    """
    Xc = X - pca_model["mean"]
    X_pca = Xc @ pca_model["components"]
    return X_pca  # shape: (N, k); X_pca[i, :] gives the coordinate of the i'th sample in PCA space