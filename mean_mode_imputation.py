import numpy as np

def mean_imputation(x, means=None, fill_all_nan_with=0.0):
    """
    Impute NaNs in continuous data by column means.

    Args:
        x: array of shape (N, D), can contain np.nan
        means: optional array of shape (D,). If None, compute from x.
        fill_all_nan_with: value used when a column has all NaNs in the 'fit' data

    Returns:
        x_filled: imputed copy of x
        means: column means used (shape (D,))
    """
    x = x.astype(float, copy=True)

    if means is None:
        means = np.nanmean(x, axis=0)
        # Handle columns that are all-NaN (nanmean -> NaN)
        all_nan_cols = np.isnan(means)
        if np.any(all_nan_cols):
            means[all_nan_cols] = fill_all_nan_with

    # Replace NaNs with the corresponding column mean
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = means[nan_cols]
    return x, means



def mode_imputation(x, modes=None, fill_all_nan_with=0):
    """
    Impute NaNs in categorical data by column modes.

    Args:
        x: array of shape (N, D), can contain np.nan (=> dtype will be float)
        modes: optional array of shape (D,). If None, compute from x.
        fill_all_nan_with: value used when a column has all NaNs in the 'fit' data

    Returns:
        x_filled: imputed copy of x
        modes: column modes used (shape (D,))
    """
    x = x.copy()

    if modes is None:
        D = x.shape[1]
        modes = np.empty(D)

        for j in range(D):
            col = x[:, j]
            col_no_nan = col[~np.isnan(col)]
            if col_no_nan.size == 0:
                # all NaN -> fallback
                modes[j] = fill_all_nan_with
                continue

            # Compute mode via counts; tie-break by smallest value
            vals, counts = np.unique(col_no_nan, return_counts=True)
            maxc = counts.max()
            candidates = vals[counts == maxc]
            modes[j] = np.min(candidates)  # tie-break: smallest value

    # Fill NaNs using modes
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = modes[nan_cols]
    return x, modes



def impute_train_test(x_train_cont, x_test_cont, x_train_cat, x_test_cat,
                      cast_categorical_to_int=True):
    """
    Fit imputation stats on train; apply to both train and test.

    Returns:
        x_train_cont_imp, x_test_cont_imp, x_train_cat_imp, x_test_cat_imp, means, modes
    """
    # Continuous
    x_train_cont_imp, means = mean_imputation(x_train_cont, means=None)
    x_test_cont_imp, _ = mean_imputation(x_test_cont, means=means)

    # Categorical
    x_train_cat_imp, modes = mode_imputation(x_train_cat, modes=None)
    x_test_cat_imp, _ = mode_imputation(x_test_cat, modes=modes)

    if cast_categorical_to_int:
        # Safe to cast back to integer type after mode imputation (no NaNs remain)
        x_train_cat_imp = x_train_cat_imp.astype(int, copy=False)
        x_test_cat_imp = x_test_cat_imp.astype(int, copy=False)

    return (x_train_cont_imp, x_test_cont_imp,
            x_train_cat_imp, x_test_cat_imp,
            means, modes)