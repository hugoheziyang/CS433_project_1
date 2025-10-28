import numpy as np 

def detect_variable_types(x_train, cat_threshold=20, int_tolerance=1e-6):
    """
    Automatically detect continuous vs categorical columns in a numpy array.
    Heuristic:
        - Few unique values → categorical
        - Integers with small range → categorical
        - Otherwise → continuous

    Args:
        x_train (np.ndarray): The training data (samples × features).
        cat_threshold (int): Maximum number of unique values to be considered categorical.
        int_tolerance (float): Tolerance for detecting integer-like float values.

    Returns:
        cont_ids (list[int]): Indices of continuous variables.
        cat_ids (list[int]): Indices of categorical variables.
    """
    n_cols = x_train.shape[1]
    cont_ids, cat_ids = [], []

    for i in range(n_cols):
        col = x_train[:, i]
        col_nonan = col[~np.isnan(col)]

        # Skip empty columns
        if col_nonan.size == 0:
            cat_ids.append(i)
            continue

        unique_vals = np.unique(col_nonan)
        nunique = len(unique_vals)

        # Check if mostly integers
        int_like = np.all(np.abs(col_nonan - np.round(col_nonan)) < int_tolerance)

        # Apply heuristic
        if nunique <= cat_threshold:    # few unique values → categorical
            cat_ids.append(i)
        elif int_like and nunique < 100:    # small range integers → categorical
            cat_ids.append(i)
        else:
            cont_ids.append(i)

    return cont_ids, cat_ids
