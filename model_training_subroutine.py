# --- Complete model training on train data: preprocess data, cross-validation, standardization, PCA transformation, 
# hyperparameter count determination, train logistic regression model ---
import h5py
import numpy as np
from preprocessing_subroutine import preprocess_data
from model_training_functions import *

def model_training():
    ### Step 1: Preprocess data
    x_train_final, x_test_final, y_train, train_ids, test_ids = preprocess_data()

    ### Step 2: Train final model by following the procedure:
    # 1. Split data into K folds to prepare for cross validation
    # 2. Set up a list of candidate k values (number of principal components) amd a number of lambda values (regularization strengths) 
    #    For each k candidate:
    #        For each lambda candidate:
    #            For each training/validation split:
    #               (i) Fit PCA on the training fold only.
    #               (ii) Transform both training and validation data.
    #               (iii) Train logistic regression on training fold.
    #               (iv) Evaluate negative log likelihood (NLL) loss on validation fold.
    # 3. Average the validation losses over folds for each (k, lambda) pair.
    # 4. Pick the value of k which minimises the average validation loss.
    # 5. Retrain on full preprocessed training set x_train_final using the chosen value of k and cache the model to .h5 file

    # Variables subject to modification
    K = 5   # number of folds for cross-validation 
    k_list = [5, 10, 15, 20, 25, 30]    # candidate k values 
    seed = 42  # random seed for reproducibility in K-fold cross validation splitting 
    use_regularization = True  # whether to use ridge regularization in logistic regression
    lambda_list = [0.01, 0.1, 1.0]  # candidate lambda values for ridge regularization
    max_iters = 2000  # maximum number of iterations for logistic regression training
    gamma = 0.1  # step size for logistic regression training
    standardize = True  # whether to standardize features before PCA

    # Cross-validation to find best k (and lambda if regularized)
    best_k, best_lambda, cv_loss = cv_logreg_find_k(x_train_final, y_train, k_list, K, 
                                                    seed, use_regularization, lambda_list, 
                                                    max_iters, gamma, standardize)
    
    # Train final model with best k (and lambda if regularized) and store it in the dictionary model with the following keys: 
    #   "w": optimal weights, numpy array of shape(D,), D is the number of features after PCA + adding column of 1s.
    #   "pca_model": dict returned by pca_fit on training data
    #   "standardize_mean": mean used for standardization (None if standardize is False)
    #   "standardize_std": std used for standardization (None if standardize is False)
    #   "k": number of PCA components used
    #   "use_regularization": whether regularization was used
    #   "lambda_": regularization strength used (None if use_regularization is False)
    #   "max_iters": max iterations used in training
    #   "gamma": step size used in training
    #   "standardize": whether standardization was used
    model = train_final_logreg_model(x_train_final, y_train, best_k,
                                    use_regularization, best_lambda,
                                    max_iters, gamma, standardize)
    
    # Save model to .h5 file
    with h5py.File("final_logreg_model.h5", "w") as f:
        for key, value in model.items():
            f.create_dataset(key, data=value)
    
    # To load the model back, use:
    # with h5py.File("final_logreg_model.h5", "r") as f:
    #     loaded_model = {key: f[key][()] for key in f.keys()}