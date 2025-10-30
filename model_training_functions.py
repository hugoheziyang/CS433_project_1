import matplotlib.pyplot as plt
import numpy as np
from implementations import reg_logistic_regression, logistic_regression, NLL_loss, sigmoid

### K-Fold Cross-Validation Indices Generator

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

def choose_k(X, cutoff=0.9):
    """
    Choose the number of PCA components k to retain at least `cutoff` fraction of variance.
    Args:
        X: data matrix, shape (N, D)
        cutoff: float in (0, 1), fraction of variance to retain
    Returns:
        k: integer, number of PCA components to retain at least `cutoff` variance
    """
    D = X.shape[1]
    pca = pca_fit(X, D)
    explained_variance_ratio = pca["explained_variance_ratio"]
    cumulative = np.cumsum(explained_variance_ratio)
    k = np.searchsorted(cumulative, cutoff) + 1
    return k  



### Add intercept term of column of 1s after PCA transform

def add_intercept(X_pca):
    """
    Add a column of 1s to the input data Z (N x k) for intercept term.
    Args:
        X_pca: input data of shape (N, k)
    Returns:
        tx: design matrix; data with intercept term, shape (N, k + 1)
    """
    tx = np.c_[np.ones((X_pca.shape[0], 1)), X_pca]
    return tx



# Cross-validation to select best pair (gamma, lambda) for logistic regression

def cv_logreg(
    X, y_pm1, gamma_list,lambda_list=(1e-3,), K=5, k=90, seed=0, standardize=True,
    use_regularization=False, max_iters=2000

):
    """
    Args:
        X: data matrix, shape (N, D)
        y_pm1: labels in {-1, +1}, shape (N, )
        gamma_list: of learning rates (step sizes) to try
        lambda_list: list of regularization strengths to try (if use_regularization is True)
        K: number of cross-validation folds
        seed: random seed for shuffling in K-fold (for reproducibility)
        use_regularization: whether to use regularized logistic regression
        max_iters: max iterations for logistic regression training

    Returns: 
        best_gamma: float, best learning rate
        best_lambda: float, best regularization (0.0 if no regularization)
        cv_loss: dict mapping (gamma, lambda) tuples to mean validation loss
    """

    # Remap original labels from ±1 to 0/1
    y = (y_pm1 + 1) / 2.0  

    # Setup
    N = X.shape[0]
    cv_loss = {}
    best_score = np.inf
    best_gamma, best_lambda = None, None 

    # Input lambda_grid, execpt if no regulariztaiton
    lambda_grid = (lambda_list if use_regularization else (0.0,))

    # Prepare K-fold splits once
    folds = list(kfold_indices(N, K=K, shuffle=True, seed=seed))

    # Grid search the hyperparameters over the K folds to optimize their value
    for gamma in gamma_list:
        for lam in lambda_grid:
            fold_losses = []

            # Extract the fold samples
            for tr_idx, val_idx in folds:
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                # Standardize on this K-fold's TRAIN set only, to avoid leakage
                if standardize:
                    m, s = standardize_fit(X_tr)
                    X_tr_std = standardize_transform(X_tr, m, s)
                    X_val_std = standardize_transform(X_val, m, s)
                else:
                    X_tr_std, X_val_std = X_tr, X_val

                # PCA fit on this K-TRAIN only (to avoid leakage), then transform both 
                n_train, D = X_tr.shape                 # First check k does not exceed feasible values
                if k > min(D, n_train):
                    raise ValueError(f"k={k} exceeds min(D={D}, n_train={n_train}).")
            
                pca = pca_fit(X_tr_std, k=k)
                Z_tr = pca_transform(X_tr_std, pca)     # PCA-transformed training data
                Z_val = pca_transform(X_val_std, pca)   # PCA-transformed validation data

                # Build design matrices (add intercept)
                tx_tr = add_intercept(Z_tr)
                tx_val = add_intercept(Z_val)

                # Train with your functions 
                initial_w = np.zeros(tx_tr.shape[1]) # initial weights to do logistic regression
                if use_regularization:
                    w, _ = reg_logistic_regression(y_tr, tx_tr, lam, initial_w, max_iters, gamma)  
                else:
                    w, _ = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)          

                # Validation loss with NLL_loss
                val_loss = NLL_loss(y_val, tx_val, w)  # no reg term in score 
                
                # Safety check: handle NaN or inf losses
                if not np.isfinite(val_loss):
                    val_loss = np.inf

                fold_losses.append(val_loss)

            mean_loss = float(np.mean(fold_losses))
            cv_loss[(gamma, lam)] = mean_loss

            if mean_loss < best_score:
                best_score = mean_loss
                best_gamma, best_lambda = gamma, lam

    return best_gamma, (best_lambda if use_regularization else None), cv_loss



### Train final logistic regression model with chosen k (and lambda if regularized)

def train_final_logreg_model(
    X_train, y_train_pm1, k,
    use_regularization=False, lambda_=1e-3,
    max_iters=2000, gamma=0.1, standardize=True
):
    """
    Fit final logistic regression model with chosen k principal components.
    Args:
        X_train: training data matrix, shape (N, D)
        y_train_pm1: training labels in {-1, +1}, shape (N, )
        k: number of PCA components to use
        use_regularization: whether to use regularized logistic regression
        lambda_: regularization strength (if use_regularization is True)
        max_iters: max iterations for logistic regression training
        gamma: step size for logistic regression training
        standardize: whether to standardize features before PCA
    Returns:
        model: dict containing trained model and preprocessing info; see below for keys
            "w": optimal weights, numpy array of shape(D,), D is the number of features after PCA + intercept.
            "pca_model": dict returned by pca_fit on training data
            "standardize_mean": mean used for standardization (None if standardize is False)
            "standardize_std": std used for standardization (None if standardize is False)
            "k": number of PCA components used
            "use_regularization": whether regularization was used
            "lambda_": regularization strength used (None if use_regularization is False)
            "max_iters": max iterations used in training
            "gamma": step size used in training
            "standardize": whether standardization was used
    """

    # Fit on training set
    if standardize:
        m_s, s_s = standardize_fit(X_train)
        Xtr_std = standardize_transform(X_train, m_s, s_s)
    else:
        m_s, s_s = None, None
        Xtr_std = X_train

    # Fit PCA on standardized training data
    pca = pca_fit(Xtr_std, k)
    Ztr = pca_transform(Xtr_std, pca)

    # Add intercept of column of 1s
    tx_tr = add_intercept(Ztr)

    # Convert labels from {-1,1} → {0,1}
    y_tr = (y_train_pm1 + 1) / 2.0

    # Train logistic regression 
    initial_w = np.zeros(tx_tr.shape[1])
    if use_regularization:
        w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
    else:
        w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)

    # Return trained model and preprocessing info
    return {
        "w": w,
        "pca_model": pca,
        "standardize_mean": m_s,
        "standardize_std": s_s,
        "k": k,
        "use_regularization": use_regularization,
        "lambda_": lambda_,
        "max_iters": max_iters,
        "gamma": gamma,
        "standardize": standardize,
    }



### Plot mean validation loss over (gamma, lambda) grid

def plot_cv_surface(cv_loss, best_gamma, best_lambda, fontsize=13):
    """
    Visualize mean validation loss over (gamma, lambda) grid.
    Args:
        cv_loss: dict {(gamma, lambda): mean_loss}
        best_gamma: float
        best_lambda: float
        fontsize: base font size for labels, legend, etc.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect unique grid values in sorted order
    gammas  = sorted({g for (g, l) in cv_loss.keys()})
    lambdas = sorted({l for (g, l) in cv_loss.keys()})

    # Sanity: log10 requires positive values
    if any(g <= 0 for g in gammas) or any(l <= 0 for l in lambdas):
        raise ValueError("All gamma and lambda values must be > 0 for log10 plotting.")

    # Build loss matrix in the same ordering
    M, N = len(gammas), len(lambdas)
    loss_matrix = np.empty((M, N))
    for i, g in enumerate(gammas):
        for j, l in enumerate(lambdas):
            loss_matrix[i, j] = cv_loss[(g, l)]

    # Log coordinates
    gx = np.log10(np.array(gammas))
    lx = np.log10(np.array(lambdas))

    # Avoid singular extents if one side has length 1
    eps = 1e-9
    x0, x1 = (lx[0] - eps, lx[0] + eps) if N == 1 else (lx.min(), lx.max())
    y0, y1 = (gx[0] - eps, gx[0] + eps) if M == 1 else (gx.min(), gx.max())

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        loss_matrix,
        origin='lower',
        aspect='auto',
        extent=[x0, x1, y0, y1],
        cmap='viridis',
        interpolation='nearest',
    )
    cbar = plt.colorbar(im)
    cbar.set_label('Mean validation loss', fontsize=fontsize)

    # Best point
    plt.scatter(np.log10(best_lambda), np.log10(best_gamma),
                color='red', marker='x', s=120, label='Best (γ, λ)')

    # Labels / title
    plt.xlabel('log10(lambda)', fontsize=fontsize)
    plt.ylabel('log10(gamma)',  fontsize=fontsize)
    plt.title('Cross-Validation Loss Surface', fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize)

    # Ticks as scientific notation (works for 1-value cases too)
    plt.xticks(lx if N > 1 else [lx[0]], [f"{v:.0e}" for v in lambdas],
               fontsize=fontsize - 2, rotation=45)
    plt.yticks(gx if M > 1 else [gx[0]], [f"{v:.0e}" for v in gammas],
               fontsize=fontsize - 2)

    plt.tight_layout()
    plt.show()



### Classify test data using trained logistic regression model 

def classify_test_data(X_test, model):
    """
    Apply a trained logistic regression model (with PCA + standardization) to test data.

    Args:
        X_test: test data matrix, shape (N, D)
        model: dict containing trained model and preprocessing info, i.e. output of train_final_logreg_model in log_reg_training.py
    
    Returns:
        results: dict with keys
            "yhat_prob": predicted probabilities P(y=1|x) on test set, shape (N, )
            "yhat_label_pm1": predicted labels in {-1, +1} on test set, shape (N, )
    """

    # Extract artifacts of trained model
    w = model["w"]
    pca = model["pca_model"]
    m_s, s_s = model["standardize_mean"], model["standardize_std"]
    standardize = model["standardize"]

    # Standardize test data from trained model, apply PCA, add intercept
    if standardize:
        Xte_std = (X_test - m_s) / s_s
    else:
        Xte_std = X_test

    Zte = pca_transform(Xte_std, pca)
    tx_te = add_intercept(Zte)

    # Evaluate predictions
    prob = sigmoid(tx_te @ w)                 # P(y=1 | x)
    yhat01 = (prob >= 0.5).astype(int)      # predicted labels in {0, 1}, threshold 0.5 assumes balanced classes
    yhat_pm1 = 2 * yhat01 - 1

    return {
        "yhat_prob": prob,
        "yhat_label_pm1": yhat_pm1,
    }