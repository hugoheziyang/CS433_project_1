def compute_pca(X):

    # Function
    # ---------
    # Implementation of Principal Component Analysis

    # Inputs
    # ---------
    # X : (N x M) with N samples, M features
    
    # Outputs
    # ---------
    # eigen_val     : (M x 1), Eigenvalues of Covariance Matrix
    # eigen_vect    : (M x M), Eigenvectors of Covariance Matrix

    # Get the number of samples N 
    N = X.shape[0]                                         # X.shape returns [#lines, #columns] of X

    # Compute Covariance Matrix C
    C = (X.T @ X) / (N - 1)                                 # @ is Python's matrix multiplication 
                                                            # C has size (M,M)
                                                            # Covariance matrices are symmetric positive semidefinite

    # Compute eigenvalues and eigenvectors 
    eigen_val, eigen_vect = np.linalg.eigh(C)                # Find eigenvalues & eigenvectors for symmetric matrices
    
    idx = np.argsort(eigen_val)[::-1]                        # Sort the eigenvalues in descending order (explain largest to smallest variance)
    eigen_val = eigen_val[idx]

    eigen_vect = eigen_vect[:, idx]                          # Sort the eigenvectors in the same order as the eigenvalues

    
    return eigen_val, eigen_vect


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def pca_explained_variance(eigvals, var_threshold=None, k=None):

    # Function
    # ---------
    # Compute the amount of variance in the dataset explained by the selected principal components 

    # Inputs
    # ---------
    # eigen_val     : (M x 1), Eigenvalues of Covariance Matrix (already sorted descending)
    # eigen_vect    : (M x M), corresponding Eigenvectors of Covariance Matrix 
    #
    # Options (name-value):
    # var_threshold : scalar in [0,1]   -> choose p s.t. CumVar(p) >= threshold
    # k             : positive integer  -> fixed number of components
    
    # Outputs
    # ---------
    # ExpVar   : (N x 1) explained variance ratios per component
    # CumVar   : (N x 1) cumulative explained variance
    # p_opt    : optimal p for the given VarThreshold (NaN if not provided)
    # var_at_k : cumulative variance captured by first K components (NaN if K not provided)


    exp_var = np.asarray(eigvals).ravel()
    exp_var = np.maximum(exp_var, 0.0)                         # Security check
    exp_var = np.sort(exp_var)[::-1]                           # Descending order of explained variance

    s = exp_var.sum()
    if s <= 0:
        ExpVar = np.zeros_like(exp_var)
        CumVar = np.zeros_like(exp_var)
        return ExpVar, CumVar, None, None

    ExpVar = exp_var / s
    CumVar = np.cumsum(ExpVar)

    p_opt = None
    var_at_k = None

    if var_threshold is not None:
        vt = float(np.clip(var_threshold, 0.0, 1.0))
        idx = np.searchsorted(CumVar, vt, side="left")
        p_opt = int(idx + 1) if idx < len(CumVar) else len(CumVar)

    if k is not None:
        kk = int(min(len(exp_var), max(1, int(round(k)))))
        var_at_k = float(CumVar[kk-1])

    return ExpVar, CumVar, p_opt, var_at_k


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


def project_pca(X, Mu, V, p):
    
    # Project data onto the selected p principal components.

    # Parameters
    # ----------
    # X : (N x M) with N samples, M features
    # Mu : (n_features,) or (1, n_features)
    #     Mean vector from original data
    # V : (n_features, n_features)
    #     Eigenvector matrix from PCA (columns = eigenvectors)
    # p : int
    #     Number of components to keep

    # Returns
    # -------
    # Y  : (n_samples, p)
    #     Projected data in p-dimensional space
    # Ap : (p, n_features)
    #     Projection matrix (maps original space to reduced space)
    # 


    # 1) Center
    Xc = X - Mu

    # 2) Projection matrix
    Ap = V[:, :p].T       # shape (p, n_features)

    # 3) Project
    Y = Xc @ Ap.T         # (n_samples, p)

    return Y, Ap


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------



def run_pca(X, var_threshold=None, k=None):

    # Function
    # ---------
    # Run PCA and return projected data for either a target variance or a fixed number of PCs

    # Inputs
    # ---------
    # X : (N x M) with N samples, M features
    # var_threshold : float, optional
    #     Desired proportion of total variance to retain (e.g., 0.95 for 95%).
    #     The function will automatically select the minimum number of components
    #     needed to reach this lexp_varel of explained variance.
    # k : int, optional
    #     Fixed number of principal components to keep, regardless of explained variance.
    
    # Outputs
    # ---------
    # ExpVar   : (N x 1) explained variance ratios per component
    # CumVar   : (N x 1) cumulative explained variance
    # p_opt    : optimal p for the given var_threshold (NaN if var_threshold not provided)
    # var_at_k : cumulative variance captured by first K components (NaN if k not provided)

    Mu = np.mean(X, axis=0, keepdims=True)

    eigvals, eigvecs = compute_pca(X)

    ExpVar, CumVar, p_opt, var_at_k = pca_explained_variance(eigvals, var_threshold, k)

    if var_threshold is not None:
        p = p_opt
        print(f"Keeping {p} components for {var_threshold*100:.1f}% variance.")
    elif k is not None:
        p = k
        print(f"Keeping {p} components ({var_at_k*100:.1f}% variance explained).")
    else:
        raise ValueError("Provide either var_threshold or k.")

    Y, Ap = project_pca(X, Mu, eigvecs, p)
    return Y, Ap, eigvals, eigvecs, ExpVar, CumVar