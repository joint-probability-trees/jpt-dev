import heapq
from typing import Callable, Iterable, Any

from dnutils import ifnone


def _count(start: int = 0, inc: int = 1):
    value = start
    while 1:
        yield value
        value += inc


class Heap:
    """Min-heap with a custom key function and stable insertion-order tiebreaking.

    Elements are stored as ``(key(item), insertion_index, item)`` triples so
    that ``heapq`` never needs to compare ``item`` objects directly.  The
    ``inc`` parameter controls how insertion indices advance:

    * ``inc=1``  (default) — older elements have smaller indices and are
      therefore popped first when keys are equal (**FIFO** tiebreaking).
    * ``inc=-1`` — newer elements have smaller indices and are therefore
      popped first when keys are equal (**LIFO** tiebreaking).

    The public interface mirrors :class:`collections.deque` where it makes
    sense: :meth:`pop` / :meth:`popleft` remove the *minimum*, while
    :meth:`popright` removes the element at the tail of the internal array
    (useful for lazy-deletion patterns, but does **not** guarantee maximum).

    Example::

        >>> h = Heap(data=[5, 1, 3])
        >>> h.pop()
        1
        >>> h.push(0)
        >>> list(h)
        [0, 3, 5]
    """

    # ------------------------------------------------------------------------------------------------------------------

    class Iterator:
        """Forward or reverse iterator over the items in heap storage order.

        Iterates the underlying ``_data`` list directly — i.e. in heap-array
        order, *not* in sorted order — without disturbing the heap.

        :param heap:    The :class:`Heap` to iterate.
        :param reverse: If ``True``, iterate from the tail to the head of the
                        internal array.
        """

        def __init__(self, heap: 'Heap', reverse: bool = False):
            self.heap = heap
            self._list_iterator = (reversed if reverse else iter)(self.heap._data)

        def __next__(self):
            return next(self._list_iterator)[2]

        def __iter__(self):
            return self

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
            self,
            data: Iterable[Any] = None,
            key: Callable = None,
            inc: int = 1
    ):
        """Initialise the heap.

        :param data: Optional iterable of initial elements.  All items are
                     inserted at once and the heap is built in O(n) time via
                     :func:`heapq.heapify`.
        :param key:  A one-argument callable used to derive the sort key from
                     each item.  Defaults to the identity function so that
                     items are compared directly.
        :param inc:  Increment applied to the internal insertion counter after
                     each :meth:`push`.  Use ``1`` for FIFO tiebreaking
                     (default) and ``-1`` for LIFO tiebreaking.
        """
        self._key = ifnone(key, lambda x: x)
        self._index = 0
        self._inc = inc
        if data:
            self._data = [(self._key(item), i, item) for i, item in zip(_count(0, self._inc), data)]
            self._index = len(self._data) * self._inc
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item: Any) -> None:
        """Push *item* onto the heap.

        :param item: The element to add.  Its sort key is computed via the
                     ``key`` function supplied at construction time.
        """
        heapq.heappush(
            self._data,
            (self._key(item), self._index, item)
        )
        self._index += self._inc

    def pop(self) -> Any:
        """Remove and return the smallest element.

        :raises IndexError: If the heap is empty.
        """
        return heapq.heappop(self._data)[2]

    def popleft(self) -> Any:
        """Alias for :meth:`pop` — remove and return the smallest element.

        Provided for API symmetry with :class:`collections.deque`.

        :raises IndexError: If the heap is empty.
        """
        return self.pop()

    def popright(self) -> Any:
        """Remove and return the element at the tail of the internal array.

        .. warning::
            This is **not** guaranteed to return the maximum element.  In a
            min-heap the tail of the backing array is a leaf node, which can
            hold any value.  This method is intended for lazy-deletion: locate
            an item with :meth:`index`, then remove it with
            ``del heap[idx]`` or swap-and-``popright``.

        :raises IndexError: If the heap is empty.
        """
        return self._data.pop()[2]

    def index(self, item: Any) -> int:
        """Return the position of *item* in the internal storage array.

        Uses identity comparison (``==``).  When multiple equal items exist
        the index of the first match in array order is returned.

        :param item: The element to search for.
        :returns:    Zero-based index into the backing array.
        :raises ValueError: If *item* is not present.
        """
        for idx, (_, _, i) in enumerate(self._data):
            if i == item:
                return idx
        else:
            raise ValueError(
                'Item %s not found.' % item
            )

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._data)

    def __getitem__(self, item: int) -> Any:
        """Return the element stored at raw array index *item*.

        Index ``0`` always holds the minimum element (heap invariant).

        :param item: Zero-based index into the backing array.
        :raises IndexError: If *item* is out of range.
        """
        return self._data[item][2]

    def __delitem__(self, item: int) -> None:
        """Delete the element at raw array index *item*.

        .. note::
            Removing an element other than the root breaks the heap invariant.
            The backing array is mutated directly without re-heapifying.
            Prefer :meth:`pop` for ordinary removal; use this only when you
            hold a valid index from :meth:`index` and accept the invariant
            violation (e.g. inside a rebuild loop).

        :param item: Zero-based index into the backing array.
        :raises IndexError: If *item* is out of range.
        """
        del self._data[item]

    def __bool__(self) -> bool:
        """Return ``True`` if the heap contains at least one element."""
        return bool(self._data)

    def __repr__(self) -> str:
        return '<Heap %s>' % [item for _, _, item in self._data]

    def __iter__(self) -> 'Heap.Iterator':
        """Return a forward iterator over items in backing-array order."""
        return Heap.Iterator(self)

    def __reversed__(self) -> 'Heap.Iterator':
        """Return a reverse iterator over items in backing-array order."""
        return Heap.Iterator(self, reverse=True)
