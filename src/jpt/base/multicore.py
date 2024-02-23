import os
import queue
import threading
import multiprocessing as mp
from typing import Callable, Any, Tuple, Optional


# noinspection PyUnresolvedReferences
class InheritLocalDataThread(threading.Thread):
    '''
    Custom subclass of ``threading.Thread`` that allows to inherit
    ``threading.local`` data from its parent thread.
    '''

    def __init__(
            self,
            target: Callable,
            args: Tuple[Any, ...],
            local: Optional[threading.local]
    ):
        super().__init__(target=target, args=args)
        self._local = local
        self._local_data = dict(local.__dict__) if local is not None else None

    def run(self):
        try:
            if self._target:
                # Copy the data from self._local_data to the forked thread
                if self._local is not None:
                    for key, value in self._local_data.items():
                        setattr(self._local, key, value)

                self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


# ----------------------------------------------------------------------------------------------------------------------
# This worker function has been copied from ``multiprocessing.pool.py`` and has been extended as
# to support an additional callback function ``terminator`` that is called when
# the worker process has finished its tasks and it is about to terminate. It can be used
# to free resources that have been allocated in the ``initializer`` function, such as database connections,
# shared memory and the like.

# noinspection PyProtectedMember,PyUnresolvedReferences
def worker(
        inqueue,
        outqueue,
        initializer=None,
        initargs=(),
        maxtasks=None,
        wrap_exception=False,
        terminator=None,
        termargs=()
):
    if (maxtasks is not None) and not (isinstance(maxtasks, int)
                                       and maxtasks >= 1):
        raise AssertionError("Maxtasks {!r} is not valid".format(maxtasks))
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        try:
            task = get()
        except (EOFError, OSError):
            mp.util.debug('worker got EOFError or OSError -- exiting')
            break

        if task is None:
            mp.util.debug('worker got sentinel -- exiting')
            break

        job, i, func, args, kwds = task
        try:
            result = (True, func(*args, **kwds))
        except Exception as e:
            if wrap_exception and func is not mp.pool._helper_reraises_exception:
                e = mp.pool.ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)
        try:
            put((job, i, result))
        except Exception as e:
            wrapped = mp.pool.MaybeEncodingError(e, result[1])
            mp.util.debug("Possible encoding error while sending result: %s" % (
                wrapped))
            put((job, i, (False, wrapped)))

        task = job = result = func = args = kwds = None
        completed += 1

    if terminator is not None:
        terminator(*termargs)

    mp.util.debug('worker exiting after %d tasks' % completed)


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyUnresolvedReferences
class Pool(mp.pool.Pool):
    '''
    Class which supports an async version of applying functions to arguments.
    '''

    def __init__(
            self,
            processes: Optional[int] = None,
            initializer: Optional[Callable] = None,
            initargs: Tuple = (),
            terminator: Optional[Callable] = None,
            termargs: Tuple = (),
            maxtasksperchild: Optional[int] = None,
            local: Optional[threading.local] = None
    ):
        # Attributes initialized early to make sure that they exist in
        # __del__() if __init__() raises an exception
        self._pool = []
        self._state = mp.pool.INIT

        self._ctx = mp.get_context()
        self._setup_queues()
        self._taskqueue = queue.SimpleQueue()
        # The _change_notifier queue exist to wake up self._handle_workers()
        # when the cache (self._cache) is empty or when there is a change in
        # the _state variable of the thread that runs _handle_workers.
        self._change_notifier = self._ctx.SimpleQueue()
        self._cache = mp.pool._PoolCache(notifier=self._change_notifier)
        self._maxtasksperchild = maxtasksperchild
        self._initializer = initializer
        self._initargs = initargs
        self._terminator = terminator
        self._termargs = termargs

        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        if initializer is not None and not callable(initializer):
            raise TypeError('initializer must be a callable')

        self._processes = processes
        try:
            self._repopulate_pool()
        except Exception:
            for p in self._pool:
                if p.exitcode is None:
                    p.terminate()
            for p in self._pool:
                p.join()
            raise

        sentinels = self._get_sentinels()

        self._worker_handler = InheritLocalDataThread(
            target=Pool._handle_workers,
            args=(self._cache, self._taskqueue, self._ctx, self.Process,
                  self._processes, self._pool, self._inqueue, self._outqueue,
                  self._initializer, self._initargs, self._maxtasksperchild,
                  self._wrap_exception, sentinels, self._change_notifier),
            local=local
            )
        self._worker_handler.daemon = True
        self._worker_handler._state = mp.pool.RUN
        self._worker_handler.start()

        self._task_handler = threading.Thread(
            target=Pool._handle_tasks,
            args=(self._taskqueue, self._quick_put, self._outqueue,
                  self._pool, self._cache)
            )
        self._task_handler.daemon = True
        self._task_handler._state = mp.pool.RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=Pool._handle_results,
            args=(self._outqueue, self._quick_get, self._cache)
            )
        self._result_handler.daemon = True
        self._result_handler._state = mp.pool.RUN
        self._result_handler.start()

        self._terminate = mp.util.Finalize(
            self, self._terminate_pool,
            args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                  self._change_notifier, self._worker_handler, self._task_handler,
                  self._result_handler, self._cache),
            exitpriority=15
            )
        self._state = mp.pool.RUN

    def _repopulate_pool(self):
        return self._repopulate_pool_static(
            self._ctx,
            self.Process,
            self._processes,
            self._pool,
            self._inqueue,
            self._outqueue,
            self._initializer,
            self._initargs,
            self._maxtasksperchild,
            self._wrap_exception,
            self._terminator,
            self._termargs
        )

    @staticmethod
    def _repopulate_pool_static(
            ctx,
            Process,
            processes,
            pool,
            inqueue,
            outqueue,
            initializer,
            initargs,
            maxtasksperchild,
            wrap_exception,
            terminator,
            termargs
    ):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(processes - len(pool)):
            w = Process(
                ctx,
                target=worker,
                args=(
                    inqueue, outqueue,
                    initializer,
                    initargs,
                    maxtasksperchild,
                    wrap_exception,
                    terminator,
                    termargs
                )
            )
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            pool.append(w)
            mp.util.debug('added worker')
