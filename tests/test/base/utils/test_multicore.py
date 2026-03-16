"""
Test cases for multicore processing functionality.
"""
import threading
import unittest
from unittest import TestCase

from jpt.base.utils.multicore import InheritLocalDataThread


# ------------------------------------------------------------------------------


class TestInheritLocalDataThread(TestCase):
    """
    Test cases for InheritLocalDataThread functionality.
    """

    def test_basic_thread_execution(self):
        """
        Test basic thread execution without local data.
        """
        # Arrange
        result = []

        def target_function(value):
            result.append(value)

        # Act
        thread = InheritLocalDataThread(
            target=target_function,
            args=("test_value",),
            local=None
        )
        thread.start()
        thread.join()

        # Assert
        self.assertEqual(["test_value"], result)

    def test_thread_with_local_data_inheritance(self):
        """
        Test thread execution with local data inheritance.
        """
        # Arrange
        local_data = threading.local()
        local_data.test_attribute = "inherited_value"
        result = []

        def target_function():
            # Access the inherited local data
            if hasattr(local_data, 'test_attribute'):
                result.append(local_data.test_attribute)

        # Act
        thread = InheritLocalDataThread(
            target=target_function,
            args=(),
            local=local_data
        )
        thread.start()
        thread.join()

        # Assert
        self.assertEqual(["inherited_value"], result)

    def test_thread_without_target_function(self):
        """
        Test thread behavior when no target function is provided.
        """
        # Arrange & Act
        thread = InheritLocalDataThread(
            target=None,
            args=(),
            local=None
        )
        thread.start()
        thread.join()

        # Assert
        # Should complete without error (target is deleted after run)
        self.assertTrue(True)  # Just verify no exception was raised

    def test_empty_local_data(self):
        """
        Test thread execution with empty local data.
        """
        # Arrange
        local_data = threading.local()
        result = []

        def target_function():
            result.append("executed")

        # Act
        thread = InheritLocalDataThread(
            target=target_function,
            args=(),
            local=local_data
        )
        thread.start()
        thread.join()

        # Assert
        self.assertEqual(["executed"], result)


if __name__ == '__main__':
    unittest.main()