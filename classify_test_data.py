import numpy as np
from implementations import sigmoid
from model_training_functions import pca_transform, add_intercept

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