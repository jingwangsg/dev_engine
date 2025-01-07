import torch.distributed as dist

def debugpy_breakpoint():
    import debugpy
    debugpy.breakpoint()
    dist.barrier()

def pdb_breakpoint():
    import pdb; pdb.set_trace()
    dist.barrier()

def ipdb_breakpoint():
    import ipdb; ipdb.set_trace()
    dist.barrier()