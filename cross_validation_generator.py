import numpy as np

def kfold_indices(N, K=5, shuffle=True, seed=0):
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        N (int): total number of samples
        K (int): number of folds
        shuffle (bool): whether to shuffle before splitting
        seed (int): random seed for reproducibility

    Yields:
        (train_idx, val_idx): two 1D numpy arrays of indices

    Example usage of generator:
        for fold, (train_idx, val_idx) in enumerate(kfold_indices(N, K=5, shuffle=True, seed=42)):
            print(f"Fold {fold+1}")
            print("  train:", train_idx)
            print("  val:  ", val_idx)

    Another example:
        for train_idx, val_idx in kfold_indices_list(N, K=5):
            # do something        
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, K)

    for i in range(K):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
        yield train_idx, val_idx
