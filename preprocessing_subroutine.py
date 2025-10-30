# --- Complete preprocessing: load, filter, split, impute, recombine ---
import numpy as np
import os
from helpers import load_csv_data
from preprocessing_functions import *

def preprocess_data(verbose=False):
    if verbose:
        print("preprocess_data: start", flush=True)
    # Use a single NumPy .npz archive for caching (simpler and atomic).
    cache_dir = "data/dataset"
    cache_dir2 = "projects/project1/data/dataset"
    cache_path = os.path.join(cache_dir, "cached_data.npz")
    cache_path2 = os.path.join(cache_dir2, "cached_data.npz")

    if os.path.exists(cache_path):
        # Load data from cached .npz archive
        with np.load(cache_path) as data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            train_ids = data["train_ids"]
            test_ids = data["test_ids"]
        if verbose:
            print(f"preprocess_data: loaded cached data from {cache_path}", flush=True)
    elif os.path.exists(cache_path2):
        # Load data from cached .npz archive
        with np.load(cache_path2) as data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            train_ids = data["train_ids"]
            test_ids = data["test_ids"]
        if verbose:
            print(f"preprocess_data: loaded cached data from {cache_path2}", flush=True)
    else:
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset", sub_sample=False)

        # Save to a single .npz archive for faster subsequent loads
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, x_train=x_train, x_test=x_test, y_train=y_train,
                 train_ids=train_ids, test_ids=test_ids)
        if verbose:
            print(f"preprocess_data: loaded data from CSV and saved cache to {cache_path}", flush=True)
    # NOTE: many helper functions (replace_weird_values, specific replacement maps,
    # and the index-based exceptions) assume original column indexing. To keep
    # indices consistent we must run replacements on the original full dataset
    # first, then suppress and filter columns while maintaining masks that map
    # back to the original feature indices.

    # Step 0: Create an empty collumn 0 in x to match feature indices:
    x_train = np.hstack((np.empty((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.empty((x_test.shape[0], 1)), x_test))

    # Step 1: Replace useless values with NaNs on full original data
    x_train_replaced = replace_weird_values(x_train)
    x_test_replaced = replace_weird_values(x_test)
    if verbose:
        print(f"preprocess_data: replaced weird values (train shape {x_train_replaced.shape}, test shape {x_test_replaced.shape})", flush=True)

    # Step 2: Suppress unuseful features (uses original indices)
    # We only need the suppressed training matrix here; the mask is stored in
    # `suppressed_mask` and will be used later to build feature-level masks.
    x_train_suppressed, suppressed_mask = suppress_features(x_train_replaced)
    P_orig = x_train.shape[1]
    kept_after_suppress = suppressed_mask.sum()
    if verbose:
        print(f"preprocess_data: suppressed features -> kept {kept_after_suppress}/{P_orig} features", flush=True)

    # Step 3: Feature filtering (remove features with too many NaNs)
    x_train_filtered, feature_mask = nan_feature_filter(x_train_suppressed, cutoff_ratio=0.2, prev_mask=suppressed_mask)
    # feature_mask is a boolean mask relative to the original feature indices
    x_test_filtered = x_test_replaced[:, feature_mask]  # Apply same mask to test set (indexing over original columns)
    kept_after_nan_filter = feature_mask.sum()
    if verbose:
        print(f"preprocess_data: nan feature filter -> kept {kept_after_nan_filter}/{P_orig} original features", flush=True)

    # If we want to also filter datapoints with many NaNs, uncomment below:
    # x_train_filtered, feature_mask = nan_datapoint_filter(x_train_filtered, cutoff_ratio=0.2, prev_mask=feature_mask)
    # y_train = y_train[feature_mask]

    # Step 4: Split into continuous and categorical columns
    continuous_mask, categorical_mask = is_categorical(x_train.shape[1], feature_mask)
    x_train_cont = x_train_filtered[:, continuous_mask]
    x_test_cont = x_test_filtered[:, continuous_mask]
    x_train_cat = x_train_filtered[:, categorical_mask]
    x_test_cat = x_test_filtered[:, categorical_mask]
    if verbose:
        print(f"preprocess_data: split cont/cat -> cont {continuous_mask.sum()}, cat {categorical_mask.sum()} (relative to filtered features)", flush=True)

    # Step 5: One-hot encode categorical columns
    # Build one-hot encodings and vocabs from the TRAIN set
    x_train_cat_enc, train_vocabs = one_hot_encode_columns(
        x_train_cat, np.ones(x_train_cat.shape[1], dtype=bool), return_vocabs=True)
    # Encode test using train vocabs to guarantee matching columns
    x_test_cat_enc = encode_with_vocabs(x_test_cat, train_vocabs)

    if verbose:
        print(f"preprocess_data: one-hot encoded categorical -> train enc shape {x_train_cat_enc.shape}, test enc shape {x_test_cat_enc.shape}", flush=True)
    # features_with_vocabs_above_threshold(feature_mask, categorical_mask, train_vocabs)

    # Step 6: Impute missing values (mean for continuous, mode for categorical)
    (x_train_cont_imp, x_test_cont_imp,
    x_train_cat_imp, x_test_cat_imp,
    means, modes) = impute_train_test(x_train_cont, x_test_cont, x_train_cat_enc, x_test_cat_enc)
    if verbose:
        print(f"preprocess_data: imputation done. cont shapes: train {x_train_cont_imp.shape}, test {x_test_cont_imp.shape}; cat shapes: train {x_train_cat_imp.shape}, test {x_test_cat_imp.shape}", flush=True)

    # Step 7: Combine imputed columns back together
    x_train_final = np.hstack((x_train_cont_imp, x_train_cat_imp))
    x_test_final = np.hstack((x_test_cont_imp, x_test_cat_imp))
    if verbose:
        print(f"preprocess_data: recombined final arrays -> train {x_train_final.shape}, test {x_test_final.shape}", flush=True)
    if verbose:
        print(f"preprocess_data: finished (total NaNs in train final: {np.isnan(x_train_final).sum()}, test final: {np.isnan(x_test_final).sum()})", flush=True)

    return x_train_final, x_test_final, y_train, train_ids, test_ids

if __name__ == "__main__":
    preprocess_data()