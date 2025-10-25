import numpy as np

def nan_feature_filter(X, cutoff_ratio=0.8):
    """
    Removes features (columns) from X that have a proportion of NaN values greater than or equal to cutoff_ratio.
    Args:
        X (np.ndarray): Input dataset of shape (n_samples, n_features).
        cutoff_ratio (float): Threshold ratio for removing features (default 0.8).
    Returns:
        np.ndarray: Dataset with features removed.
        np.ndarray: Boolean mask of kept features (True = kept, False = removed).
    """
    nan_proportions = np.mean(np.isnan(X), axis=0)
    keep_mask = nan_proportions < cutoff_ratio
    X_new = X[:, keep_mask]
    return X_new, keep_mask

def nan_datapoint_filter(X, cutoff_ratio=0.8):
    """
    Removes datapoints (rows) from X that have a proportion of NaN values greater than or equal to cutoff_ratio.
    Args:
        X (np.ndarray): Input dataset of shape (n_samples, n_features).
        cutoff_ratio (float): Threshold ratio for removing datapoints (default 0.8).
    Returns:
        np.ndarray: Dataset with datapoints removed.
        np.ndarray: Boolean mask of kept datapoints (True = kept, False = removed).
    """
    nan_proportions = np.mean(np.isnan(X), axis=1)
    keep_mask = nan_proportions < cutoff_ratio
    X_new = X[keep_mask, :]
    return X_new, keep_mask
