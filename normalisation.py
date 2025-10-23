import numpy as np

def normalize(X):

    # Compute the mean and std of each feature (column) on the training data,
    # Then normalize.

    # Parameters
    # ----------
    # X : np.ndarray of shape (n_samples, n_features)
    #     Data to normalize.

    # Returns
    # -------
    # X_norm : np.ndarray (n_samples, n_features)
    #     Normalized data.

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)   # Need to change ddof to 1 if we want to use Bessel's correction and compute sample STD instead of STD
                                        # The divisor is (N - ddof) with N the number of elements.
                                         
    sigma[sigma == 0] = 1.0             # avoid divide-by-zero    

    return (X - mu) / sigma             # normalize
