import numpy as np

def normalize(X):

    # Function
    # ---------
    # Computes the mean and std of each feature (column) on the training data,
    # then normalizes.

    # Inputs
    # ----------
    # X : (N x M) with N samples, M features
    #     Data to normalize.

    # Outputs
    # -------
    # X_norm : (N x M)
    #          Normalized data.

    mu = np.mean(X, axis=0)
    
    sigma = np.std(X, axis=0, ddof=0)   # Need to change ddof to 1 if we want to use Bessel's correction and compute sample STD instead of STD
                                        # The divisor is (N - ddof) with N the number of elements.
                                         
    sigma[sigma == 0] = 1.0             # avoid divide-by-zero    

    X_norm = (X - mu) / sigma             # normalize

    return X_norm
