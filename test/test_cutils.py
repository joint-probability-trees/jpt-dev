from unittest import TestCase

import numpy as np

try:
    from jpt.base.cutils import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.cutils import test_sort


class CUtilsTest(TestCase):

    def check_sorted(self, data, indices, length):
        prev = -np.inf
        for j in range(length):
            self.assertGreaterEqual(data[indices[j]], prev)
            prev = data[indices[j]]

    @staticmethod
    def gen_data(n):
        return np.random.uniform(-100, 100, n), np.array([i for i in range(n)])

    def test_sort(self):
        data, indices = CUtilsTest.gen_data(3079828)
        orig_data = np.array(data)
        test_sort(data, indices)
        self.check_sorted(orig_data, indices, data.shape[0])

    def test_partial(self):
        data, indices = CUtilsTest.gen_data(3079828)
        orig_data = np.array(data)
        test_sort(data, indices, 50)
        self.check_sorted(orig_data, indices, 50)

    def test_duplicates(self):
        data = np.array([5.] * 200 + [3., 3., 3., 3., 1.])
        indices = np.array([_ for _ in range(data.shape[0])])
        orig_data = np.array(data)
        test_sort(data, indices)
        self.check_sorted(orig_data, indices, orig_data.shape[0])

    def test_from_file(self):
        import pickle
        with open('resources/nparray-sort-test.dat', 'rb') as f:
            arr = pickle.load(f).astype(np.float64)
            orig_data = np.array(arr)
        indices = np.array([_ for _ in range(arr.shape[0])])
        test_sort(arr, indices)
        self.check_sorted(orig_data, indices, indices.shape[0])
