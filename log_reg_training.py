import numpy as np
from implementations import logistic_regression, reg_logistic_regression, NLL_loss, sigmoid
from cross_validation_generator import *
from pca_functions import *

### Choose the best k (number of PCA components) via K-fold cross-validation

def cv_logreg_find_k(
    X, y_pm1, k_list, K=5, seed=0,
    use_regularization=False, lambda_list=(1e-3,), max_iters=2000, gamma=0.1,
    standardize=True
):
    """
    Args:
        X: data matrix, shape (N, D)
        y_pm1: labels in {-1, +1}, shape (N, )
        k_list: list/tuple of integers, numbers of PCA components to try
        K: number of CV folds
        seed: random seed for shuffling in K-fold
        use_regularization: whether to use regularized logistic regression
        lambda_list: list/tuple of regularization strengths to try (if use_regularization is True)
        max_iters: max iterations for logistic regression training
        gamma: step size for logistic regression training
        standardize: whether to standardize features before PCA

    Returns: 
        best_k: integer, best number of PCA components
        best_lambda: float or None, best regularization strength (None if use_regularization is False)
        cv_scores: dict mapping (k, lambda) tuples to mean validation loss
    """

    # Remap original labels from ±1 to 0/1
    y = (y_pm1 + 1) / 2.0  

    N = X.shape[0]
    cv_scores = {}
    best_score = np.inf
    best_k, best_lambda = None, None

    for k in k_list:
        lam_grid = (lambda_list if use_regularization else (0.0,))
        for lam in lam_grid:
            fold_losses = []
            for tr_idx, val_idx in kfold_indices(N, K=K, shuffle=True, seed=seed):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                # Standardize on TRAIN only
                if standardize:
                    m, s = standardize_fit(X_tr)
                    X_tr_std = standardize_transform(X_tr, m, s)
                    X_val_std = standardize_transform(X_val, m, s)
                else:
                    X_tr_std, X_val_std = X_tr, X_val

                # PCA fit on TRAIN only, then transform both
                pca = pca_fit(X_tr_std, k=k)
                Z_tr = pca_transform(X_tr_std, pca)
                Z_val = pca_transform(X_val_std, pca)

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
                val_loss = NLL_loss(y_val, tx_val, w)  # no reg term in score  :contentReference[oaicite:6]{index=6}
                fold_losses.append(val_loss)

            mean_loss = float(np.mean(fold_losses))
            cv_scores[(k, lam)] = mean_loss

            if mean_loss < best_score:
                best_score = mean_loss
                best_k, best_lambda = k, lam

    return best_k, (best_lambda if use_regularization else None), cv_scores



### Train final model on full data with best hyperparameters

def train_final_logreg_model(
    X_train, y_train_pm1, k,
    *, use_regularization=False, lambda_=1e-3,
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
