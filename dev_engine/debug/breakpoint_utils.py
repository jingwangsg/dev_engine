import torch.distributed as dist

def debugpy_breakpoint():
    import debugpy
    debugpy.breakpoint()
    if dist.is_initialized():
        dist.barrier()

def pdb_breakpoint():
    import pdb; pdb.set_trace()
    if dist.is_initialized():
        dist.barrier()

def ipdb_breakpoint():
    import ipdb; ipdb.set_trace(context=0)
    if dist.is_initialized():
        dist.barrier()