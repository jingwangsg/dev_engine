import torch
from dev_engine.logging import ic
from dev_engine.debug import (
    debugpy_breakpoint,
    pdb_breakpoint,
    ipdb_breakpoint,
    setup_debugpy,
)
from dev_engine.debug.global_ops import set_object, del_object, save_object, load_object, get_object
from dev_engine import logging as log
from dev_engine.debug.visualize import write_image, write_video, draw_heatmap, draw_histogram
import builtins

def install_distributed():
    log.debug("Installing distributed builtins")

    builtins.dist_barrier = torch.distributed.barrier
    builtins.dist_rank = torch.distributed.get_rank
    builtins.dist_world_size = torch.distributed.get_world_size


def install_debug():
    log.debug("Installing debug builtins")

    builtins.ic = ic
    builtins.debugpy_breakpoint = debugpy_breakpoint
    builtins.pdb_breakpoint = pdb_breakpoint
    builtins.ipdb_breakpoint = ipdb_breakpoint

    builtins._set = set_object
    builtins._del = del_object
    builtins._save = save_object
    builtins._load = load_object
    builtins._get = get_object

    setup_debugpy()


def install_visualize():
    log.debug("Installing visualize builtins")
    builtins.write_image = write_image
    builtins.write_video = write_video
    builtins.draw_heatmap = draw_heatmap
    builtins.draw_histogram = draw_histogram


def install_all():
    install_debug()
    install_distributed()
    install_visualize()
