import functools
import inspect
import os
from functools import partial

_GLOBAL_COUNTER = {}


def run_first_n(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.stack()[1]
            caller_info = (caller_frame.filename, caller_frame.lineno)

            name = (hash(func), caller_info)

            current_count = _GLOBAL_COUNTER.get(name, 0)
            if current_count < n:
                func(*args, **kwargs)
            _GLOBAL_COUNTER[name] = current_count + 1

        return wrapper

    return decorator


def run_every_n(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.stack()[1]
            caller_info = (caller_frame.filename, caller_frame.lineno)

            name = (hash(func), caller_info)

            current_count = _GLOBAL_COUNTER.get(name, 0)
            if current_count % n == 0:
                func(*args, **kwargs)
            _GLOBAL_COUNTER[name] = current_count + 1

        return wrapper

    return decorator


def rprint(*args, rank=0, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[rank{rank}]", *args, **kwargs)


def print_every_n(n):
    return run_every_n(n)(print)


def rprint_every_n(n, rank=0):
    return run_every_n(n)(partial(rprint, rank=rank))
