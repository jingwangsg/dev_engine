from dev_engine.logging import ic
from dev_engine.debug import (
    setup_debugpy,
)
import ipdb
import debugpy
from dev_engine import logging as log
import builtins
from loguru import logger
import coredumpy
import os
import pathlib as osp

try:
    import torch
except ImportError:
    is_torch_available = False
    logger.warning("torch not installed")
else:
    is_torch_available = True


def install_debug(breakpoint_type: str = "ipdb"):
    logger.debug(f"Installing debug builtins with breakpoint: {breakpoint_type}")
    from dev_engine.debug.global_ops import (
        set_object,
        del_object,
        save_object,
        load_object,
        get_object,
        object_in_disk,
        object_in_store,
        capture_calls,
    )

    def print_file(obj, path: str = "example.txt"):
        if osp.exists(path):
            with open(path, "a") as f:
                f.write(str(obj) + "\n")
        else:
            with open(path, "w") as f:
                f.write(str(obj) + "\n")

    log.debug("Installing debug builtins")

    builtins.ic = ic
    builtins._set = set_object
    builtins._del = del_object
    builtins._save = save_object
    builtins._load = load_object
    builtins._get = get_object
    builtins._in_disk = object_in_disk
    builtins._in_store = object_in_store
    builtins.capture_calls = capture_calls
    builtins.print_file = print_file

    if breakpoint_type == "debugpy":
        setup_debugpy()
        builtins.breakpoint = debugpy.breakpoint
    elif breakpoint_type == "ipdb":
        builtins.breakpoint = ipdb.set_trace
    else:
        raise ValueError(f"Invalid breakpoint: {breakpoint_type}")

    if os.environ.get("COREDUMPY"):
        logger.debug("register coredumpy")
        coredumpy.patch_except(directory="./dumps")


def install_visualize():
    from dev_engine.debug.visualize import (
        write_image,
        write_video,
        draw_heatmap,
        draw_histogram,
        draw_barplot,
    )

    log.debug("Installing visualize builtins")
    builtins.write_image = write_image
    builtins.write_video = write_video
    builtins.draw_heatmap = draw_heatmap
    builtins.draw_histogram = draw_histogram
    builtins.draw_barplot = draw_barplot


def install_torch_utils():
    if not is_torch_available:
        logger.warning("torch not installed, skipping torch_utils installation")
        return

    from dev_engine.debug.torch_utils import param_info

    builtins.dist_barrier = torch.distributed.barrier
    builtins.dist_rank = torch.distributed.get_rank
    builtins.dist_world_size = torch.distributed.get_world_size

    builtins.param_info = param_info


def install_all():
    install_debug()
    install_torch_utils()
    install_visualize()
