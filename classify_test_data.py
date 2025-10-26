import numpy as np
from implementations import NLL_loss, sigmoid
from pca_functions import *

### Classify test data using trained logistic regression model from train_final_logreg_model in log_reg_training.py

def classify_test_data(X_test, y_test_pm1, model):
    """
    Apply a trained logistic regression model (with PCA + standardization)
    to test data and evaluate NLL and accuracy.

    Args:
        X_test: test data matrix, shape (N, D)
        y_test_pm1: test labels in {-1, +1}, shape (N, )
        model: dict containing trained model and preprocessing info, i.e. output of train_final_logreg_model in log_reg_training.py
    
    Returns:
        results: dict with keys
            "test_nll": NLL loss on test set (float)
            "test_acc": accuracy on test set (float)
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

    # Convert labels {-1,1} → {0,1} for NLL
    y_te = (y_test_pm1 + 1) / 2.0

    # Evaluate predictions and loss
    test_nll = float(NLL_loss(y_te, tx_te, w))      # NLL on test set
    prob = sigmoid(tx_te @ w)                 # P(y=1 | x)
    yhat01 = (prob >= 0.5).astype(int)      # predicted labels in {0, 1}, threshold 0.5 assumes balanced classes
    yhat_pm1 = 2 * yhat01 - 1

    test_acc = float(np.mean(yhat_pm1 == y_test_pm1))   # proportion of correct predictions

    return {
        "test_nll": test_nll,
        "test_acc": test_acc,
        "yhat_prob": prob,
        "yhat_label_pm1": yhat_pm1,
    }