import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from queue import Empty, Queue
import multiprocessing as mp
from dev_engine import logging as log

from pathos.multiprocessing import Pool, ProcessPool, ThreadingPool
from tqdm import tqdm


def _run_sequential(iterable, func, desc="", verbose=True):
    pbar = tqdm(total=len(iterable), desc=desc, disable=not verbose)
    ret = []
    for it in iterable:
        ret.append(func(it))
        pbar.update(1)
    pbar.close()
    return ret


class imap_async:
    """
    A parallel mapping implementation that supports both thread and process pools.
    It provides safe queuing of tasks with retry capability and proper resource management.

    This class implements an iterator interface that yields (input, result) pairs
    as tasks complete, not necessarily in the order they were submitted.
    """

    def __init__(
        self,
        iterable,
        func,
        num_workers=32,
        fail_condition=lambda x: x is None,
        mode="thread",
        max_retries=3,
    ):
        """
        Initialize the async mapper.

        Args:
            iterable: Input data to be processed
            func: Processing function to be applied to each item
            num_workers: Size of thread/process pool
            fail_condition: Function that returns True if result should trigger retry
            mode: 'thread' or 'process' - determines pool type
            max_retries: Maximum number of retry attempts per item
        """
        self.iterable = iterable
        self.func = func
        self.num_workers = num_workers
        self.fail_condition = fail_condition
        self.max_retries = max_retries

        # Initialize queues based on execution mode
        if mode == "process":
            # Use process-safe queues when using process pool
            manager = mp.Manager()
            self.task_queue = manager.Queue()
            self.not_done = manager.Queue()
        else:
            # Use thread-safe queues for thread pool
            self.task_queue = Queue()
            self.not_done = Queue()

        self.iterable_is_over = False
        self.producer_thread = None

        # Initialize the appropriate executor pool unless parallel execution is disabled
        if not os.getenv("DISABLE_PARALLEL", False):
            if mode == "thread":
                self.executor = ThreadingPool(num_workers)
            elif mode == "process":
                self.executor = ProcessPool(num_workers)
            else:
                raise ValueError("mode must be either 'thread' or 'process'")
        else:
            self.executor = None
        
        # https://github.com/uqfoundation/pathos/issues/111
        self.executor.restart()

    def _producer(self):
        """
        Producer thread function that feeds items into the task queue.
        Each item is paired with its retry count, starting at 0.
        Signals completion by putting None into the queue.
        """
        try:
            for item in self.iterable:
                self.task_queue.put((0, item))  # (retry_count, item)
            # Signal that all original items have been queued
            self.task_queue.put(None)
        except Exception as e:
            log.error(f"Producer thread error: {str(e)}")
            # Ensure we signal completion even on error
            self.task_queue.put(None)
            raise

    def start_producer(self):
        """Start the producer thread to begin feeding items into the task queue."""
        self.producer_thread = threading.Thread(target=self._producer)
        self.producer_thread.daemon = (
            True  # Allow program to exit if thread is still running
        )
        self.producer_thread.start()

    def _apipe_wait(self, timeout):
        """
        Check incomplete futures and return completed ones.

        Args:
            timeout: Maximum time to wait for futures to complete

        Returns:
            list: Completed futures
        """
        done = []
        temp_not_done = Queue()  # Temporary queue for incomplete futures

        # Process all futures in not_done queue
        while True:
            try:
                future = self.not_done.get_nowait()
                if future.ready():
                    done.append(future)
                else:
                    temp_not_done.put(future)
            except Empty:
                break

        # Return incomplete futures to not_done queue
        while True:
            try:
                future = temp_not_done.get_nowait()
                self.not_done.put(future)
            except Empty:
                break

        return done

    def _is_not_done_empty(self):
        """
        Safely check if the not_done queue is empty.
        Returns:
            bool: True if queue is empty, False otherwise
        """
        try:
            item = self.not_done.get_nowait()
            self.not_done.put(item)
            return False
        except Empty:
            return True

    def __iter__(self):
        """
        Iterator implementation that yields (input, result) pairs as they complete.
        Handles task submission, monitoring, retries, and resource cleanup.
        """
        # Handle synchronous execution if parallel processing is disabled
        if os.getenv("DISABLE_PARALLEL", False):
            for item in self.iterable:
                try:
                    yield item, self.func(item)
                except Exception as e:
                    log.error(f"Error processing item {item}: {str(e)}")
            return

        self.start_producer()

        try:
            while True:
                # Submit new tasks from the queue
                while True:
                    try:
                        task = self.task_queue.get_nowait()
                        if task is None:
                            self.iterable_is_over = True
                            break

                        retry_count, item = task
                        future = self.executor.apipe(self.func, item)
                        future._input = item
                        future._retry_count = retry_count
                        self.not_done.put(future)
                    except Empty:
                        break

                # Process completed tasks
                done_futures = self._apipe_wait(timeout=0.1)

                for future in done_futures:
                    try:
                        # Get result with timeout to avoid hanging
                        ret = future.get(timeout=0.1)
                        if self.fail_condition(ret):
                            if future._retry_count < self.max_retries:
                                # Requeue failed tasks that haven't exceeded retry limit
                                self.task_queue.put(
                                    (future._retry_count + 1, future._input)
                                )
                            else:
                                log.warning(
                                    f"Task {future._input} exceeded max retries"
                                )
                        else:
                            yield future._input, ret
                    except Exception as e:
                        log.error(
                            f"Error processing task {future._input}: {str(e)}"
                        )

                # Check if all work is complete
                if (
                    self.iterable_is_over
                    and self.task_queue.empty()
                    and self._is_not_done_empty()
                ):
                    break

                # Small sleep to prevent busy-waiting
                time.sleep(0.1)

        except Exception as e:
            log.error(f"Error in parallel execution: {str(e)}")
            raise
        finally:
            self.close()

    def _clear_queue(self, queue):
        """
        Safely clear all items from a queue.

        Args:
            queue: Queue instance to be cleared
        """
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break

    def close(self):
        """
        Clean up resources by clearing queues and shutting down the executor.
        Waits for producer thread to complete with timeout.
        """
        self._clear_queue(self.task_queue)
        self._clear_queue(self.not_done)

        if self.executor:
            self.executor.close()
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=1.0)


def map_async(
    iterable,
    func,
    num_workers=30,
    chunksize=1,
    desc="",
    total=None,
    test_flag=False,
    verbose=True,
    mode="thread",
):
    if mode == "thread":
        return _map_async_with_thread(
            iterable,
            func,
            num_thread=num_workers,
            desc=desc,
            verbose=verbose,
            test_flag=test_flag,
            total=total,
        )
    elif mode == "process":
        return _map_async(
            iterable,
            func,
            num_process=num_workers,
            chunksize=chunksize,
            desc=desc,
            total=total,
            test_flag=test_flag,
            verbose=verbose,
        )
    else:
        raise ValueError("mode should be either 'thread' or 'process'")


def _map_async(
    iterable,
    func,
    num_process=30,
    chunksize=1,
    desc: object = "",
    total=None,
    test_flag=False,
    verbose=True,
):
    """while test_flag=True, run sequentially"""
    if test_flag or os.getenv("DISABLE_PARALLEL", False):
        return _run_sequential(iterable, func, desc=desc, verbose=verbose)
    else:
        p = Pool(num_process)

        ret = p.map_async(
            func=func,
            iterable=iterable,
            chunksize=chunksize,
        )

        if "__len__" in dir(iterable):
            total = len(iterable)

        if total is not None:
            total = (len(iterable) + chunksize - 1) // chunksize

        pbar = tqdm(total=total, desc=desc, disable=not verbose)

        while ret._number_left > 0:
            if verbose:
                pbar.n = total - ret._number_left
                pbar.refresh()
            time.sleep(0.1)

        pbar.close()

        return ret.get()


def _map_async_with_thread(
    iterable,
    func,
    num_thread=30,
    desc="",
    verbose=True,
    test_flag=False,
    total=None,
):
    if test_flag or os.getenv("DISABLE_PARALLEL", False):
        return _run_sequential(iterable, func, desc=desc, verbose=verbose)

    with ThreadPoolExecutor(num_thread) as executor:

        results = []

        not_done = set()
        if "__len__" in dir(iterable):
            total = len(iterable)
        pbar = tqdm(total=total, desc=desc, disable=not verbose)

        for i, x in enumerate(iterable):
            future = executor.submit(func, x)
            future.index = i
            not_done.add(future)

        results = {}

        while len(not_done) > 0:
            done, not_done = wait(not_done, return_when="FIRST_COMPLETED")
            pbar.update(len(done))
            for future in done:
                results[future.index] = future.result()

        pbar.close()

        return [results[i] for i in range(len(iterable))]


class ApipeWait:
    def __init__(self):
        self.st = time.time()

    def __call__(self, not_done, timeout=5.0):
        done = set(future for future in not_done if future.ready())
        not_done = not_done - done
        while time.time() - self.st < timeout:
            pass

        self.st = time.time()

        return done, not_done


apipe_wait = ApipeWait()
