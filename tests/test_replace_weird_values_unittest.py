import unittest
import numpy as np
import importlib.util


def _load_module(path):
    spec = importlib.util.spec_from_file_location("pf", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ReplaceWeirdValuesUnitTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path

        # tests/ sits inside the root folder; the module file lives one level up
        mod_path = Path(__file__).parent.parent / "preprocessing_functions.py"
        mod = _load_module(str(mod_path))
        self.replace_weird_values = mod.replace_weird_values

        rows = 3
        cols = 306
        self.X = np.ones((rows, cols), dtype=float)

        # Generic replacements (not in exception lists): col 10
        self.X[0, 10] = 7  # should become NaN (general nan_values)
        self.X[1, 10] = 8  # should become 0 (general zero_values)

        # Column in full no-replace list: col 1 should stay unchanged
        self.X[0, 1] = 7
        self.X[0, 248] = 8

        self.X[0, 28] = 9  # should stay unchanged

        # Specific replacements
        self.X[0, 82] = 555  # should become 0 (82 in [82..87])
        self.X[0, 83] = 555

        # 196: replace 97->NaN and 98->0 per rules
        self.X[0, 196] = 97
        self.X[1, 196] = 98

        # 89: replace 98->NaN
        self.X[0, 89] = 98

        # 263: replace 900->NaN
        self.X[0, 263] = 900

        # 265: replace 99900->NaN
        self.X[0, 265] = 99900

        # 294: replace 99000->NaN
        self.X[0, 294] = 99000

    def test_replacements(self):
        Y = self.replace_weird_values(self.X.copy())

        # Generic
        self.assertTrue(np.isnan(Y[0, 10]), "col 10 value 7 should become NaN")
        self.assertEqual(Y[1, 10], 0, "col 10 value 8 should become 0")

        # No-replace-all
        self.assertEqual(Y[0, 1], 7, "col 1 is in no-replace-all and should remain 7")
        self.assertEqual(
            Y[0, 248], 8, "col 248 is in no-replace-all and should remain 8"
        )

        self.assertEqual(Y[0, 28], 9, "col 280 should remain 9")

        # Specific replacements
        self.assertEqual(Y[0, 82], 0)
        self.assertEqual(Y[0, 83], 0)

        self.assertTrue(np.isnan(Y[0, 196]))
        self.assertEqual(Y[1, 196], 0)

        self.assertTrue(np.isnan(Y[0, 89]))
        self.assertTrue(np.isnan(Y[0, 263]))
        self.assertTrue(np.isnan(Y[0, 265]))
        self.assertTrue(np.isnan(Y[0, 294]))


if __name__ == "__main__":
    unittest.main()
