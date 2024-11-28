import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait

from pathos.multiprocessing import ProcessPool, ThreadingPool, Pool
from tqdm import tqdm
import threading
from queue import Queue, Empty


def _run_sequential(iterable, func, desc="", verbose=True):
    pbar = tqdm(total=len(iterable), desc=desc, disable=not verbose)
    ret = []
    for it in iterable:
        ret.append(func(it))
        pbar.update(1)
    pbar.close()
    return ret


def imap_async(iterable, func, num_process=32):
    if os.getenv("DISABLE_PARALLEL", False):
        for it in iterable:
            yield it, func(it)

    task_queue = Queue()

    def producer():
        for item in iterable:
            task_queue.put(item)
        task_queue.put(None)

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    with ProcessPool(num_process) as executor:
        not_done = set()
        iterable_is_over = False

        while True:
            while True:
                try:
                    item = task_queue.get_nowait()
                    print(item)
                    if item is None:
                        iterable_is_over = True
                        break

                    future = executor.apipe(func, item)
                    future._input = item
                    not_done.add(future)
                except Empty:
                    break

            done, not_done = apipe_wait(not_done, timeout=0.1)
            for future in done:
                yield future._input, future.get()

            if len(not_done) == 0 and iterable_is_over:
                break

        time.sleep(1)

    producer_thread.join()


def imap_async_with_thread(iterable, func, num_thread=32):
    if os.getenv("DISABLE_PARALLEL", False):
        for it in iterable:
            yield it, func(it)

    task_queue = Queue()

    def producer():
        for item in iterable:
            task_queue.put(item)
        task_queue.put(None)

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    with ThreadingPool(num_thread) as executor:
        not_done = set()
        iterable_is_over = False

        while True:
            while True:
                try:
                    item = task_queue.get_nowait()
                    if item is None:
                        iterable_is_over = True
                        break
                    future = executor.apipe(func, item)
                    future._input = item
                    not_done.add(future)
                except Empty:
                    break

            done, not_done = apipe_wait(not_done, timeout=0.1)
            for future in done:
                yield future._input, future.get()

            if len(not_done) == 0 and iterable_is_over:
                break

        time.sleep(1)

    producer_thread.join()


def map_async(
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


def map_async_with_thread(
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
