import torch
from dev_engine.logging import ic
from dev_engine.debug import (
    debugpy_breakpoint,
    pdb_breakpoint,
    ipdb_breakpoint,
    setup_debugpy,
)
from dev_engine import logging as log
from dev_engine.debug.visualize import write_image, write_video
import builtins


def install_distributed():
    log.debug("Installing distributed builtins")

    builtins.__barrier = torch.distributed.barrier
    builtins.__rank = torch.distributed.get_rank
    builtins.__world_size = torch.distributed.get_world_size


def install_debug():
    log.debug("Installing debug builtins")

    builtins.ic = ic
    builtins.debugpy_breakpoint = debugpy_breakpoint
    builtins.pdb_breakpoint = pdb_breakpoint
    builtins.ipdb_breakpoint = ipdb_breakpoint

    setup_debugpy()


def install_visualize():
    log.debug("Installing visualize builtins")
    builtins.write_image = write_image
    builtins.write_video = write_video


def install_all():
    install_debug()
    install_distributed()
    install_visualize()
