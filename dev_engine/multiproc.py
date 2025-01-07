import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from queue import Empty, Queue
import multiprocessing as mp

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
    def __init__(
        self,
        iterable,
        func,
        num_workers=32,
        fail_condition=lambda x: x is None,
        mode="thread",
    ):
        self.iterable = iterable
        self.func = func
        self.num_workers = num_workers
        self.fail_condition = fail_condition
        self.task_queue = mp.Manager().Queue() if mode == "process" else Queue()
        self.not_done = set()
        self.iterable_is_over = False
        self.producer_thread = None

        if not os.getenv("DISABLE_PARALLEL", False):
            if mode == "thread":
                self.executor = ThreadPoolExecutor(num_workers)
            elif mode == "process":
                self.executor = ProcessPool(num_workers)
            else:
                raise ValueError("mode should be either 'thread' or 'process'")
        else:
            self.executor = None

    def _producer(self):
        for item in self.iterable:
            self.task_queue.put(item)
        self.task_queue.put(None)

    def start_producer(self):
        self.producer_thread = threading.Thread(target=self._producer)
        self.producer_thread.start()

    def __iter__(self):
        if os.getenv("DISABLE_PARALLEL", False):
            for it in self.iterable:
                yield it, self.func(it)
            return

        self.start_producer()

        while True:
            # Fetch new tasks and submit to the process pool
            while True:
                try:
                    item = self.task_queue.get_nowait()
                    if item is None:
                        self.iterable_is_over = True
                        break

                    future = self.executor.apipe(self.func, item)
                    future._input = item
                    self.not_done.add(future)
                except Empty:
                    break

            # Collect results from completed tasks
            done, self.not_done = self._apipe_wait(self.not_done, timeout=0.1)
            for future in done:
                ret = future.get()
                if self.fail_condition(ret):
                    self.task_queue.put(future._input)  # Re-add failed tasks
                    continue
                yield future._input, ret

            # Exit condition: all tasks completed and iterable is exhausted
            if len(self.not_done) == 0 and self.iterable_is_over:
                break

            time.sleep(1)

        self.producer_thread.join()

    def _apipe_wait(self, not_done, timeout):
        done = set()
        remaining = set()
        for future in not_done:
            if future.ready():
                done.add(future)
            else:
                remaining.add(future)
        return done, remaining

    def close(self):
        if self.executor:
            self.executor.close()
        if self.producer_thread:
            self.producer_thread.join()


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
