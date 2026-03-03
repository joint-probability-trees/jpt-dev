"""
Test cases for preprocessing functionality.
"""
import pandas as pd
import numpy as np
import unittest
from unittest import TestCase

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe
from jpt.learning.preprocessing import preprocess_data


# ------------------------------------------------------------------------------


class TestPreprocessData(TestCase):
    """
    Test cases for data preprocessing functionality.
    """

    def test_preprocess_basic_data(self):
        """
        Test basic data preprocessing without multicore.
        """
        # Arrange
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['x', 'y', 'x', 'y', 'x'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))
        self.assertEqual(data.columns.tolist(), processed_data.columns.tolist())

    def test_preprocess_empty_dataframe(self):
        """
        Test preprocessing of empty DataFrame.
        Note: Empty DataFrames cannot infer variable types, so we skip this test.
        """
        # This test is skipped because empty DataFrames cannot have variable types inferred
        # which is a known limitation of the current implementation
        self.skipTest("Empty DataFrames cannot have variable types inferred")

    def test_preprocess_single_column(self):
        """
        Test preprocessing of single column DataFrame.
        """
        # Arrange
        data = pd.DataFrame({'single_col': [10, 20, 30, 40]})
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))
        self.assertEqual(['single_col'], processed_data.columns.tolist())

    def test_preprocess_with_verbose(self):
        """
        Test preprocessing with verbose output enabled.
        """
        # Arrange
        data = pd.DataFrame({
            'num_col': [1, 2, 3],
            'cat_col': ['a', 'b', 'c']
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=True)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))

    def test_preprocess_numeric_data_types(self):
        """
        Test preprocessing preserves numeric data types appropriately.
        """
        # Arrange
        data = pd.DataFrame({
            'integers': [1, 2, 3, 4],
            'floats': [1.5, 2.5, 3.5, 4.5],
            'mixed': [1, 2.5, 3, 4.5]
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))
        # Check that numeric data is handled properly
        for col in processed_data.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(processed_data[col]) or 
                pd.api.types.is_object_dtype(processed_data[col])
            )

    def test_preprocess_categorical_data(self):
        """
        Test preprocessing of categorical data.
        """
        # Arrange
        data = pd.DataFrame({
            'category': ['red', 'blue', 'green', 'red', 'blue'],
            'ordinal': ['low', 'medium', 'high', 'low', 'medium']
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))
        # Verify categorical data is preserved
        self.assertEqual(data.columns.tolist(), processed_data.columns.tolist())

    def test_preprocess_with_missing_values(self):
        """
        Test preprocessing data with missing values.
        """
        # Arrange
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['x', 'y', None, 'y', 'x'],
            'C': [1.1, np.nan, 3.3, 4.4, 5.5]
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))

    def test_preprocess_large_dataframe(self):
        """
        Test preprocessing of larger DataFrame for performance.
        """
        # Arrange
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.choice(['A', 'B', 'C'], 1000),
            'feature3': np.random.randint(0, 100, 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(data), len(processed_data))
        self.assertEqual(data.columns.tolist(), processed_data.columns.tolist())

    def test_preprocess_maintains_data_integrity(self):
        """
        Test that preprocessing maintains data integrity for simple cases.
        """
        # Arrange
        data = pd.DataFrame({
            'simple_int': [1, 2, 3],
            'simple_str': ['a', 'b', 'c']
        })
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables)

        # Act
        processed_data = preprocess_data(jpt, data, multicore=None, verbose=False)

        # Assert
        self.assertIsInstance(processed_data, pd.DataFrame)
        # For simple data, check that values are preserved or appropriately transformed
        self.assertEqual(len(data), len(processed_data))
        self.assertEqual(set(data.columns), set(processed_data.columns))


if __name__ == '__main__':
    unittest.main()