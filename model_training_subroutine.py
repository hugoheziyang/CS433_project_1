# --- Complete model training on train data: preprocess data, cross-validation, standardization, PCA transformation,
# hyperparameter count determination, train logistic regression model ---
import numpy as np
import pickle
from preprocessing_subroutine import preprocess_data
from model_training_functions import *
import argparse

def model_training(k=None, gamma_list=None, lambda_list=None, verbose=True):
    ### Step 1: Preprocess data
    x_train_final, x_test_final, y_train, train_ids, test_ids = preprocess_data(
        verbose=True
    )

    ### Step 2: Train final model by following the procedure:
    # 1. Split data into K folds to prepare for cross validation
    # 2. Set up a list of candidate gamma values (learning rates) and a number of lambda values (regularization strengths)
    #    For each gamma candidate:
    #        For each lambda candidate:
    #            For each training/validation split:
    #               (i) Fit PCA on the training fold only.
    #               (ii) Transform both training and validation data.
    #               (iii) Train logistic regression on training fold.
    #               (iv) Evaluate F1 score on validation fold.
    # 3. Average the F1 scores over folds for each (gamma, lambda) pair.
    # 4. Pick the optimal pair (gamma, lambda) which maximises the F1 score.
    # 5. Retrain on full preprocessed training set x_train_final using the chosen value of k and cache the model to .h5 file
    
    # Variables subject to modification
    if gamma_list is None:
        # use log scale for gamma values
        gamma_list = np.logspace(-3, -0.3, 5)  # candidate gamma (step-size) values for gradient descent
    
    if lambda_list is None:
        lambda_list = np.logspace(-3, -0.07, 8)  # candidate lambda (regularization strength) values for ridge regularization
 
    if k is None:
        k = 90  # PCA : fix k the number of selected principal components
    
    K = 5   # number of folds for cross-validation   
    seed = 42  # random seed for reproducibility in K-fold cross validation splitting 
  
    standardize = True  # whether to standardize features before PCA
    use_regularization = (
        True  # whether to use ridge regularization in logistic regression
    )
    max_iters = 200  # maximum number of iterations for logistic regression training

    if verbose:
        print(f"model_training: starting cross-validation with k={k}, gamma_list={gamma_list}, lambda_list={lambda_list}", flush=True)

    # Cross-validation for (and gamma, lambda)
    best_gamma, best_lambda, cv_f1 = cv_logreg(
        X=x_train_final,
        y_pm1=y_train,
        gamma_list=gamma_list,
        lambda_list=lambda_list,
        K=K,
        k=k,
        seed=seed,
        standardize=standardize,
        use_regularization=use_regularization,
        max_iters=max_iters,
        verbose=True,
    )

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

    if verbose:
        print(f"model_training: training final model with k={k}, lambda={best_lambda}, gamma={best_gamma}", flush=True)
    
    model = train_final_logreg_model(X_train=x_train_final,
        y_train_pm1=y_train,
        k=k,
        use_regularization=use_regularization,
        lambda_=best_lambda,
        max_iters=max_iters,
        gamma=best_gamma,
        standardize=standardize,
        verbose=True,
    )

    # Save model using Python's pickle (standard library). This stores nested
    # dictionaries and arbitrary Python objects without wrapping them in 0-d
    # object arrays (unlike np.savez). The file will be named final_logreg_model.pkl.
    with open(f"k={k}_lambda={lambda_list[0]}_logreg_model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    # To load the model back, use:
    # import pickle
    # with open("final_logreg_model.pkl", "rb") as fh:
    #     loaded_model = pickle.load(fh)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression model with optional k and lambda from CLI.")
    parser.add_argument("--k", type=int, help="Number of PCA components to keep (overrides default).")
    parser.add_argument("--lambda", dest="lambda_val", type=float,
                        help="Single regularization strength to use (will be passed as a one-element list).")
    parser.add_argument("--lambda-list", dest="lambda_list", nargs="+", type=float,
                        help="Space-separated list of candidate lambda values for CV.")
    parser.add_argument("--gamma-list", dest="gamma_list", nargs="+", type=float,
                        help="Space-separated list of candidate gamma (step-size) values for CV.")
    args = parser.parse_args()

    k = args.k if args.k is not None else None

    if args.lambda_list is not None:
        lambda_list = args.lambda_list
    elif args.lambda_val is not None:
        lambda_list = [args.lambda_val]
    else:
        lambda_list = None

    gamma_list = args.gamma_list if args.gamma_list is not None else None

    model = model_training(k=k, gamma_list=gamma_list, lambda_list=lambda_list)

    # Print Loss on training data
    print(f"Training Loss: {model['loss']}")

    # Compute F1 score on training data
    compute_f1_on_train(model=model, verbose=True)