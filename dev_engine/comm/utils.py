# Copyright (c) Facebook, Inc. and its affiliates.
"""
Mainly modified from detectron2
copied from https://github.com/baaivision/EVA/blob/master/EVA-02/det/detectron2/utils/comm.py
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import numpy as np
import torch
import torch.distributed as torch_dist
import os
from contextlib import contextmanager
from torch.distributed import broadcast

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def init_dist(backend="nccl", init_backend="torch", **kwargs):
    """
    backend (str): name of the backend to use. Options include:
        - "gloo": to use Gloo backend
        - "nccl": to use NCCL backend
        - "mpi": to use MPI backend
        - "tcp": to use TCP backend
    init_backend: "torch" or "deepspeed"
    """
    if init_backend == "torch":
        torch_dist.init_process_group(backend=backend, init_method="env://", **kwargs)
    elif init_backend == "deepspeed":
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=False, **kwargs)
    local_rank = get_local_rank()
    print(f"Initialized rank {get_rank()} with local rank {local_rank}")
    torch.cuda.set_device(local_rank)


def get_dist_info():
    if not torch_dist.is_available():
        return 0, 1
    if not torch_dist.is_initialized():
        return 0, 1
    return torch_dist.get_rank(), torch_dist.get_world_size()


def get_world_size() -> int:
    if not torch_dist.is_available():
        return 1
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()


def get_rank() -> int:
    if not torch_dist.is_available():
        return 0
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not torch_dist.is_available():
        return 0
    if not torch_dist.is_initialized():
        return 0
    # assert _LOCAL_PROCESS_GROUP is not None, "Local process group is not created! Please use launch() to spawn processes!"
    return int(os.getenv("LOCAL_RANK", 0))


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not torch_dist.is_available():
        return 1
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not torch_dist.is_available():
        return
    if not torch_dist.is_initialized():
        return
    world_size = torch_dist.get_world_size()
    if world_size == 1:
        return

    # if torch_dist.get_backend() == torch_dist.Backend.NCCL:
    #     # This argument is needed to avoid warnings.
    #     # It's valid only for NCCL backend.
    #     torch_dist.barrier(device_ids=[torch.cuda.current_device()])
    # else:
    #     torch_dist.barrier()
    torch_dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if torch_dist.get_backend() == "nccl":
        return torch_dist.new_group(backend="gloo")
    else:
        return torch_dist.group.WORLD


def get_backend_device():
    # only supports nccl so far
    return torch.device("cuda", torch.cuda.current_device())


def all_gather(data, group=None, requires_grad=False):
    """
    Return a list of tensors
    """

    if requires_grad:
        _all_gather_func = torch_dist.nn.functional.all_gather(data, group=group)
    else:
        _all_gather_func = torch_dist.all_gather

    if group is None:
        group = _get_global_gloo_group()

    world_size = torch_dist.get_world_size(group)
    if world_size == 1:
        return [data]

    device = get_backend_device()
    output = [torch.empty_like(data, device=device) for _ in range(world_size)]
    _all_gather_func(output, data, group=group)
    return output


def all_gather_variable(data, lengths=None, requires_grad=False, group=None):
    """
    Support variable length tensor (here length defined as the first dimension)
    This implementation supposedly supports grad backprop (not verified yet)
    """
    world_size = get_world_size()
    assert len(lengths) == world_size, f"len(lengths) must match world size: lengths({len(lengths)}) vs world_size({world_size})"
    if lengths is None:
        _length = torch.tensor(data.shape[0], dtype=torch.int)
        lengths = [_.item() for _ in all_gather(_length)]

    max_length = max(lengths)

    pad_length = max_length - data.shape[0]
    if pad_length > 0:
        data = torch.cat([data, torch.zeros(pad_length, *data.shape[1:], dtype=data.dtype, device=data.device)], dim=0)
    padded_list = all_gather(data, requires_grad=requires_grad, group=group)
    ret_list = [padded[:length] for padded, length in zip(padded_list, lengths)]
    return ret_list


def all_gather_object(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    # if group is None:
    #     group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = torch_dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    torch_dist.all_gather_object(output, data, group=group)
    return output


def broadcast_object_list(data, src=0, group=None):
    if get_world_size() == 1:
        return data
    # if group is None:
    #     group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = torch_dist.get_world_size(group)
    if world_size == 1:
        return data

    torch_dist.broadcast_object_list(data, src=src, group=group)
    return data


def gather_object(data, dst=0, group=None):
    if get_world_size() == 1:
        return [data]
    world_size = torch_dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)] if get_rank() == dst else None
    torch_dist.gather_object(data, output, dst=dst, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = torch_dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = torch_dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        torch_dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        torch_dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch_dist.reduce(values, dst=0)
        if torch_dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
