# --- Complete preprocessing: load, filter, split, impute, recombine ---
import h5py
import numpy as np
from nan_filtering import nan_feature_filter, nan_datapoint_filter
from is_categorical_or_continuous import is_categorical
from mean_mode_imputation import impute_train_test


def preprocess_data():
    with h5py.File("data/dataset/cached_data.h5", "r") as f:
        x_train = f["x_train"][:]
        x_test = f["x_test"][:]
        y_train = f["y_train"][:]
        train_ids = f["train_ids"][:]
        test_ids = f["test_ids"][:]


    # Step 1: Feature filtering (remove features with >=80% NaNs)
    x_train_filtered, feature_mask = nan_feature_filter(x_train, cutoff_ratio=0.2)
    x_test_filtered = x_test[:, feature_mask]  # Apply same mask to test set

    # If we want to also filter datapoints with many NaNs, uncomment below:
    # x_train_filtered, datapoint_mask = nan_datapoint_filter(x_train_filtered, cutoff_ratio=0.2)
    # y_train = y_train[datapoint_mask]

    # Step 2: Split into continuous and categorical columns
    continuous_mask, categorical_mask = is_categorical(x_train.shape[1], feature_mask)
    x_train_cont = x_train_filtered[:, continuous_mask]
    x_test_cont = x_test_filtered[:, continuous_mask]
    x_train_cat = x_train_filtered[:, categorical_mask]
    x_test_cat = x_test_filtered[:, categorical_mask]

    # Step 3: Impute missing values (mean for continuous, mode for categorical)
    (x_train_cont_imp, x_test_cont_imp,
    x_train_cat_imp, x_test_cat_imp,
    means, modes) = impute_train_test(x_train_cont, x_test_cont, x_train_cat, x_test_cat)

    # Step 4: Combine imputed columns back together
    x_train_final = np.empty_like(x_train_filtered)
    x_test_final = np.empty_like(x_test_filtered)
    x_train_final[:, continuous_mask] = x_train_cont_imp
    x_train_final[:, categorical_mask] = x_train_cat_imp
    x_test_final[:, continuous_mask] = x_test_cont_imp
    x_test_final[:, categorical_mask] = x_test_cat_imp

    return x_train_final, x_test_final, y_train