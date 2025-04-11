from dev_engine.logging import ic
from dev_engine.debug import (
    debugpy_breakpoint,
    pdb_breakpoint,
    ipdb_breakpoint,
    setup_debugpy,
)
from dev_engine import logging as log
import builtins
from loguru import logger

try:
    import torch
except ImportError:
    is_torch_available = False
    logger.warning("torch not installed")
else:
    is_torch_available = True


def install_debug(breakpoint: str = "debugpy"):
    from dev_engine.debug.global_ops import (
        set_object,
        del_object,
        save_object,
        load_object,
        get_object,
        object_in_disk,
        object_in_store,
    )

    def write_file(obj, path: str = "example.txt"):
        with open(path, "w") as f:
            f.write(str(obj))

    log.debug("Installing debug builtins")

    builtins.ic = ic

    builtins._set = set_object
    builtins._del = del_object
    builtins._save = save_object
    builtins._load = load_object
    builtins._get = get_object
    builtins._in_disk = object_in_disk
    builtins._in_store = object_in_store

    builtins.write_file = write_file

    if breakpoint == "debugpy":
        setup_debugpy()
        builtins.breakpoint = debugpy_breakpoint
    elif breakpoint == "pdb":
        builtins.breakpoint = pdb_breakpoint
    elif breakpoint == "ipdb":
        builtins.breakpoint = ipdb_breakpoint
    else:
        raise ValueError(f"Invalid breakpoint: {breakpoint}")


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
