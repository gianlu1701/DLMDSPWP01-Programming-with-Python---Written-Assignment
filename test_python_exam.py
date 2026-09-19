import os
import tempfile
import unittest

import pandas as pd

from python_exam import (
    Dataset,
    IdealFunctionSelector,
    TestDataMapper,
    ProjectError,
    DataMismatchError,
    NoMatchingFunctionError,
    SelectionResult
)

# =============================================================================
# EXCEPTION HIERARCHY
# =============================================================================

class TestExceptionHierarchy(unittest.TestCase):
    """
    Verifies that the custom exceptions form the required inheritance chain
    and can be caught either individually or via their common base class.
    """

    def test_data_mismatch_error_is_a_project_error(self):
        self.assertTrue(issubclass(DataMismatchError, ProjectError))

    def test_no_matching_function_error_is_a_project_error(self):
        self.assertTrue(issubclass(NoMatchingFunctionError, ProjectError))

    def test_project_error_is_a_standard_exception(self):
        self.assertTrue(issubclass(ProjectError, Exception))

    def test_data_mismatch_error_can_be_caught_as_project_error(self):
        with self.assertRaises(ProjectError):
            raise DataMismatchError("x-grids differ")

# =============================================================================
# DATASET LOADING
# =============================================================================

class TestDataset(unittest.TestCase):
    """
    Verifies that Dataset correctly loads a CSV file into a DataFrame.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_loads_csv_into_dataframe(self):
        path = os.path.join(self.tmpdir, "sample.csv")
        pd.DataFrame({"x": [0, 1], "y": [0.0, 1.0]}).to_csv(path, index=False)

        dataset = Dataset(path)

        self.assertEqual(dataset.name, "sample")
        self.assertEqual(len(dataset.df), 2)
        self.assertListEqual(list(dataset.df.columns), ["x", "y"])

    def test_missing_file_raises_file_not_found(self):
        missing_path = os.path.join(self.tmpdir, "does_not_exist.csv")
        with self.assertRaises(FileNotFoundError):
            Dataset(missing_path)

# =============================================================================
# IDEAL FUNCTION SELECTION
# =============================================================================

class TestIdealFunctionSelector(unittest.TestCase):
    """
    Verifies the least-squares selection logic with a small, hand-checkable
    dataset (3 candidate functions instead of 50).
    """

    def setUp(self):
        x = [0, 1, 2, 3]
        # training data is close to y = 2x, with a small offset
        self.train_df = pd.DataFrame({"x": x, "y1": [0.1, 2.1, 4.1, 6.1]})
        self.ideal_df = pd.DataFrame({
            "x": x,
            "y1": [0.0, 1.0, 2.0, 3.0],   # y = x
            "y2": [0.0, 2.0, 4.0, 6.0],   # y = 2x  <- best fit
            "y3": [0.0, 3.0, 6.0, 9.0],   # y = 3x
        })

    def test_selects_the_closest_ideal_function(self):
        selector = IdealFunctionSelector(self.train_df, self.ideal_df)
        results = selector.select_best_functions()

        self.assertEqual(results["y1"].ideal_column, "y2")
        self.assertAlmostEqual(results["y1"].max_deviation, 0.1, places=6)

    def test_mismatched_x_grid_raises_data_mismatch_error(self):
        shifted_ideal_df = self.ideal_df.copy()
        shifted_ideal_df["x"] = shifted_ideal_df["x"] + 1

        with self.assertRaises(DataMismatchError):
            IdealFunctionSelector(self.train_df, shifted_ideal_df)

# =============================================================================
# TEST DATA MAPPING
# =============================================================================

class TestTestDataMapper(unittest.TestCase):
    """
    Verifies the sqrt(2) mapping criterion, including points that qualify
    and points that must be rejected.
    """

    def setUp(self):

        x = [0, 1, 2, 3]
        self.ideal_df = pd.DataFrame({"x": x, "y2": [0.0, 2.0, 4.0, 6.0]})
        # a max_deviation of 0.5 was "observed" between training and y2 during selection
        self.selected = {
            "y1": SelectionResult(train_column="y1", ideal_column="y2", sse=0.1, max_deviation=0.5)
        }

    def test_point_within_tolerance_is_mapped(self):
        # ideal value at x=1 is 2; deviation 0.6 <= 0.5*sqrt(2) (~0.707) -> should map
        test_df = pd.DataFrame({"x": [1], "y": [2.6]})
        mapper = TestDataMapper(test_df, self.ideal_df, self.selected)
        mapped, discarded = mapper.map_test_data()

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped.iloc[0]["ideal_function"], "y2")
        self.assertAlmostEqual(mapped.iloc[0]["delta_y"], 0.6, places=6)

    def test_point_outside_tolerance_is_dropped(self):
        # ideal value at x=1 is 2; deviation 1.0 > 0.5*sqrt(2) (~0.707) -> should NOT map
        test_df = pd.DataFrame({"x": [1], "y": [3.0]})
        mapper = TestDataMapper(test_df, self.ideal_df, self.selected)
        mapped, discarded = mapper.map_test_data()

        self.assertEqual(len(mapped), 0)
        self.assertEqual(len(discarded), 1)
        self.assertAlmostEqual(discarded.iloc[0]["x"], 1.0, places=6)


    def test_x_not_in_ideal_grid_is_excluded_without_raising(self):
        # off-grid points are now reported and skipped, not a hard failure
        test_df = pd.DataFrame({"x": [99], "y": [1.0]})
        mapper = TestDataMapper(test_df, self.ideal_df, self.selected)  # should not raise
        mapped, discarded = mapper.map_test_data()

        self.assertEqual(len(mapped), 0)

    def test_discarded_values_are_not_mixed_up_between_points(self):
        # a discarded point followed by a mapped point must not leak values across iterations
        # (e.g. the mapped point must not end up in `discarded`, and vice versa)
        test_df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.6]})  # x=1 discarded, x=2 mapped
        mapper = TestDataMapper(test_df, self.ideal_df, self.selected)
        mapped, discarded = mapper.map_test_data()

        self.assertEqual(len(mapped), 1)
        self.assertEqual(len(discarded), 1)
        self.assertEqual(mapped.iloc[0]["x"], 2)
        self.assertEqual(discarded.iloc[0]["x"], 1)
        self.assertAlmostEqual(discarded.iloc[0]["y"], 3.0, places=6)

    def test_off_grid_point_does_not_block_valid_points(self):
        # one bad point (x=99, not on the grid) should not prevent a good
        # point (x=1, within tolerance) from being mapped normally
        test_df = pd.DataFrame({"x": [99, 1], "y": [1.0, 2.6]})
        mapper = TestDataMapper(test_df, self.ideal_df, self.selected)
        mapped, discarded = mapper.map_test_data()

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped.iloc[0]["x"], 1)
        

if __name__ == "__main__":
    unittest.main()
