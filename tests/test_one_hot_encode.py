import os
import sys
import unittest
import numpy as np

# Ensure the project module directory (parent of this tests folder) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing_functions import one_hot_encode
from preprocessing_functions import one_hot_encode_columns


class TestOneHotEncode(unittest.TestCase):
    def test_numeric_with_nan(self):
        g = np.array([1, 2, np.nan, np.nan, 3])
        one_hot, vocab = one_hot_encode(g)

        # vocab must contain no NaN entries
        self.assertFalse(np.isnan(vocab).any(), msg=f"vocab contains NaN: {vocab}")

        # number of columns must match vocabulary length
        self.assertEqual(one_hot.shape[1], len(vocab))

        # rows corresponding to NaN input must be all zeros
        nan_mask = np.isnan(g)
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                self.assertTrue(
                    np.all(one_hot[i] == 0), msg=f"row {i} expected all zeros"
                )
            else:
                # non-NaN rows should have exactly one '1'
                self.assertEqual(one_hot[i].sum(), 1)

    def test_all_non_nan(self):
        g = np.array([1, 2, 1, 3])
        one_hot, vocab = one_hot_encode(g)

        self.assertFalse(np.isnan(vocab).any())
        self.assertEqual(one_hot.shape, (4, len(vocab)))
        # vocabulary should contain exactly the unique values {1,2,3}
        self.assertEqual(set(vocab.tolist()), {1, 2, 3})

    def test_multi_column_encode(self):
        # Create a dataset with 3 columns; encode columns 0 and 2
        X = np.array([[1, 10, 2], [2, 11, np.nan], [1, 12, 3], [3, 13, 2]], dtype=float)

        # mask: True for columns to encode
        mask = [True, False, True]

        X_new, vocabs = one_hot_encode_columns(X, mask, return_vocabs=True)

        # vocabs should have two entries (for columns 0 and 2)
        self.assertEqual(len(vocabs), 2)

        # Check vocab content and shapes
        vocab0 = vocabs[0]
        vocab1 = vocabs[1]
        # vocab0 should contain {1,2,3}
        self.assertEqual(set(vocab0.tolist()), {1, 2, 3})
        # vocab1 should contain {2,3} (nan dropped)
        self.assertEqual(set(vocab1.tolist()), {2, 3})

        # Number of columns in X_new should equal: len(vocab0) + 1 (kept col1) + len(vocab1)
        expected_cols = len(vocab0) + 1 + len(vocab1)
        self.assertEqual(X_new.shape, (4, expected_cols))

        # Check that original col1 is intact
        col1_original = X[:, 1]
        col1_new = X_new[:, len(vocab0)]
        np.testing.assert_array_equal(col1_original, col1_new)

        # Rows that originally had NaN in col2 (row index 1) should have zeros in the block
        # corresponding to vocab1 (the last len(vocab1) columns)
        last_block = X_new[:, -len(vocab1) :]
        self.assertTrue(np.all(last_block[1] == 0))


if __name__ == "__main__":
    unittest.main()
