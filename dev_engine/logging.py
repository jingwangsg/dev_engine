import os
from functools import partial
import functools
import atexit
import os
import sys
from typing import Any, Optional
import torch
import numpy as np
import inspect
from icecream import IceCreamDebugger, argumentToString

import torch.distributed as dist
from loguru._logger import Core, Logger

_GLOBAL_COUNTER = {}


def run_first_n(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.stack()[1]
            caller_info = (caller_frame.filename, caller_frame.lineno)

            name = caller_info

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

            name = caller_info

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


def print_first_n(n):
    return run_first_n(n)(print)


def rprint_first_n(n, rank=0):
    return run_first_n(n)(partial(rprint, rank=rank))


RANK0_ONLY = eval(os.getenv("LOGURU_RANK0_ONLY", "True"))
LEVEL = os.environ.get("LOGURU_LEVEL", "INFO")

logger = Logger(
    core=Core(),
    exception=None,
    depth=1,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

atexit.register(logger.remove)


def _add_relative_path(record: dict[str, Any]) -> None:
    start = os.getcwd()
    record["extra"]["relative_path"] = os.path.relpath(record["file"].path, start)


*options, _, extra = logger._options  # type: ignore
logger._options = tuple([*options, [_add_relative_path], extra])  # type: ignore


def init_loguru_stdout() -> None:
    logger.remove()
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        sys.stdout,
        level=LEVEL,
        format="[<green>{time:MM-DD HH:mm:ss}</green>|"
        f"{machine_format}"
        f"{message_format}",
        filter=_rank0_only_filter,
    )


def init_loguru_file(path: str) -> None:
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        path,
        encoding="utf8",
        level=LEVEL,
        format="[<green>{time:MM-DD HH:mm:ss}</green>|"
        f"{machine_format}"
        f"{message_format}",
        rotation="100 MB",
        filter=lambda result: _rank0_only_filter(result) or not RANK0_ONLY,
        enqueue=True,
    )


def get_machine_format() -> str:
    node_id = os.environ.get("NODE_RANK", "0")
    num_nodes = int(os.environ.get("NNODES", "1"))
    machine_format = ""
    rank = 0
    if dist.is_available():
        if not RANK0_ONLY and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            machine_format = (
                f"<red>[Node{node_id:<3}/{num_nodes:<3}][RANK{rank:<5}/{world_size:<5}]"
                + "[{process.name:<8}]</red>| "
            )
    return machine_format


def get_message_format() -> str:
    message_format = "<level>{level}</level>|<cyan>{extra[relative_path]}:{line}:{function}</cyan>] {message}"
    return message_format


def _rank0_only_filter(record: Any) -> bool:
    is_rank0 = record["extra"].get("rank0_only", True)
    if _get_rank() == 0 and is_rank0:
        return True
    if not is_rank0:
        record["message"] = f"[RANK {_get_rank()}]" + record["message"]
    return not is_rank0


def custom_args_to_string(obj: Any) -> str:
    meta_info = []

    if "__class__" in dir(obj):
        meta_info = [f"{obj.__class__.__name__}"]
    elif isinstance(obj, (int, float, str)):
        meta_info += [f"{obj}"]

    if torch.is_tensor(obj):
        meta_info += [
            f"shape: {list(obj.shape)}",
            f"dtype: {obj.dtype}",
            f"has_nan: {torch.isnan(obj).any().item()}",
            f"has_inf: {torch.isinf(obj).any().item()}",
            f"requires_grad: {obj.requires_grad}",
            f"min,max: {obj.min().item(), obj.max().item()}",
            f"device: {obj.device}",
        ]
        if obj.is_floating_point():
            meta_info.append(f"mean: {obj.mean().item()}")
            meta_info.append(f"std: {obj.std().item()}")
    elif isinstance(obj, np.ndarray):
        meta_info += [
            f"shape: {list(obj.shape)}",
            f"dtype: {obj.dtype}",
            f"has_nan: {np.isnan(obj).any()}",
            f"has_inf: {np.isinf(obj).any()}",
            f"min,max: {obj.min(), obj.max()}",
        ]
        if np.issubdtype(obj.dtype, np.floating):
            meta_info.append(f"mean: {obj.mean()}")
            meta_info.append(f"std: {obj.std()}")
    elif isinstance(obj, dict):
        meta_info.append(f"keys: {list(obj.keys())}")
    else:
        if "__dict__" in dir(obj):
            keys = list(obj.__dict__.keys())
            nonhidden_keys = [k for k in keys if not k.startswith("_")]
            meta_info.append(f"keys: {nonhidden_keys}")

        if "__len__" in dir(obj):
            meta_info.append(f"len: {len(obj)}")

        if "__getitem__" in dir(obj) and "__len__" in dir(obj) and len(obj) > 0:
            meta_info.append(f"item0: {obj[0]}")

    return " | ".join(meta_info)


icecream_debugger = IceCreamDebugger(argToStringFunction=custom_args_to_string)


def ic(*obj: Any, vb=False) -> str:
    call_frame = inspect.currentframe().f_back
    output_str = icecream_debugger._format(call_frame, *obj)
    if vb:
        print(output_str)
    else:
        return output_str


def trace(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).trace(message)


def debug(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).debug(message)


def info(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).info(message)


def success(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).success(message)


def warning(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).warning(message)


def error(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).error(message)


def critical(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).critical(message)


def exception(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).exception(message)


def _get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


# Execute at import time.
init_loguru_stdout()

