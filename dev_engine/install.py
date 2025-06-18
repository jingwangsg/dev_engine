from dev_engine.logging.logging import ic
from dev_engine.debug import (
    setup_debugpy,
)
import ipdb
import debugpy
from dev_engine.logging import logging as log
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

def omni_read(path: str):
    if path.endswith(".pkl") or path.endswith(".pickle") or path.endswith(".pt"):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            import dill
            print(f"Error loading {path}: {e}, trying dill...")
            data = dill.load(open(path, "rb"))
    elif path.endswith(".json"):
        import json
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        import yaml
        with open(path, "r") as f:
            data = yaml.load(f)
    elif path.endswith(".csv"):
        import polars as pl
        data = pl.read_csv(path)
    elif path.endswith(".parquet"):
        import polars as pl
        data = pl.read_parquet(path)
    elif path.endswith(".h5") or path.endswith(".hdf5"):
        import h5py
        data = h5py.File(path, "r")
    elif path.endswith(".mp4") or path.endswith(".avi") or path.endswith(".mov") or path.endswith(".webm"):
        import decord
        data = decord.VideoReader(path)
        frames = data.get_batch(range(len(data)))
        return frames
    else:
        raise ValueError(f"Unsupported file extension: {path}")



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

    def fprint(obj, path: str = "example.txt"):
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
    builtins.fprint = fprint
    builtins.omr = omni_read

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
    from dev_engine.visualize import (
        write_image,
        write_video,
        draw_heatmap,
        draw_histogram,
        draw_barplot,
        draw_lines,
    )

    log.debug("Installing visualize builtins")
    builtins.write_image = write_image
    builtins.write_video = write_video
    builtins.draw_heatmap = draw_heatmap
    builtins.draw_histogram = draw_histogram
    builtins.draw_barplot = draw_barplot
    builtins.draw_lines = draw_lines


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
