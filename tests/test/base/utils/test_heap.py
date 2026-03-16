from unittest import TestCase

from jpt.base.utils.heap import Heap


class HeapTest(TestCase):

    def test_iterator(self):
        """Verify Heap iteration yields elements in ascending order."""
        # Arrange
        h = Heap(data=[5, 4, 8])

        # Act
        result = list(iter(h))

        # Assert
        self.assertEqual([4, 5, 8], result)

    def test_reverse(self):
        """Verify reversed Heap iteration yields elements in descending order."""
        # Arrange
        h = Heap(data=[5, 4, 8])

        # Act
        result = list(reversed(h))

        # Assert
        self.assertEqual([8, 5, 4], result)

    def test_push_pop(self):
        """Verify push/pop maintains min-heap ordering."""
        # Arrange
        h = Heap()
        for v in [3, 1, 4, 1, 5, 9, 2, 6]:
            h.push(v)

        # Act
        result = [h.pop() for _ in range(len(h))]

        # Assert
        self.assertEqual([1, 1, 2, 3, 4, 5, 6, 9], result)

    def test_push_pop_with_key(self):
        """Verify key function controls heap ordering."""
        # Arrange
        h = Heap(key=lambda x: -x)  # max-heap via negation
        for v in [3, 1, 4, 1, 5]:
            h.push(v)

        # Act
        result = [h.pop() for _ in range(len(h))]

        # Assert
        self.assertEqual([5, 4, 3, 1, 1], result)

    def test_len_and_bool(self):
        """Verify __len__ and __bool__ reflect heap state."""
        h = Heap()
        self.assertFalse(h)
        self.assertEqual(0, len(h))
        h.push(1)
        self.assertTrue(h)
        self.assertEqual(1, len(h))

    def test_index(self):
        """Verify index() returns the position of an item in the internal data list."""
        h = Heap(data=[10, 20, 30])
        idx = h.index(10)
        self.assertEqual(10, h[idx])

    def test_index_missing(self):
        """Verify index() raises ValueError for absent items."""
        h = Heap(data=[1, 2, 3])
        self.assertRaises(ValueError, h.index, 99)

    def test_delitem(self):
        """Verify __delitem__ removes an element at a given index."""
        h = Heap(data=[1, 2, 3])
        idx = h.index(2)
        del h[idx]
        self.assertEqual(2, len(h))
        self.assertNotIn(2, list(h))

    def test_popleft(self):
        """Verify popleft is an alias for pop — returns the minimum."""
        h = Heap(data=[7, 3, 5])
        self.assertEqual(3, h.popleft())
        self.assertEqual(2, len(h))

    def test_popright(self):
        """Verify popright removes and returns the tail of the internal array."""
        # After heapify the tail is not the max, but it is a valid heap element.
        h = Heap(data=[7, 3, 5])
        before = set(h)
        item = h.popright()
        self.assertIn(item, before)
        self.assertEqual(2, len(h))
        self.assertNotIn(item, list(h))

    def test_getitem(self):
        """Verify __getitem__ accesses the item at a raw internal index."""
        h = Heap(data=[10, 20, 30])
        # The minimum is always at index 0 in a min-heap.
        self.assertEqual(10, h[0])

    def test_repr(self):
        """Verify __repr__ contains the Heap prefix and listed items."""
        h = Heap(data=[2, 1])
        r = repr(h)
        self.assertTrue(r.startswith('<Heap '))
        self.assertIn('1', r)
        self.assertIn('2', r)

    def test_inc_fifo_tiebreak(self):
        """Verify inc=1 (default) breaks ties in insertion order (FIFO)."""
        h = Heap(key=lambda x: 0)  # all keys equal
        for v in [10, 20, 30]:
            h.push(v)
        self.assertEqual([10, 20, 30], [h.pop() for _ in range(len(h))])

    def test_inc_lifo_tiebreak(self):
        """Verify inc=-1 breaks ties in reverse insertion order (LIFO)."""
        h = Heap(key=lambda x: 0, inc=-1)
        for v in [10, 20, 30]:
            h.push(v)
        self.assertEqual([30, 20, 10], [h.pop() for _ in range(len(h))])

    def test_empty_iteration(self):
        """Verify iterating an empty heap yields no elements."""
        self.assertEqual([], list(Heap()))
        self.assertEqual([], list(reversed(Heap())))
