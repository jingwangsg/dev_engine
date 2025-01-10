from dev_engine.logging import objinfo
from dev_engine.debug import (
    debugpy_breakpoint,
    pdb_breakpoint,
    ipdb_breakpoint,
    setup_debugpy,
)


def install():
    import builtins

    builtins.objinfo = objinfo
    builtins.debugpy_breakpoint = debugpy_breakpoint
    builtins.pdb_breakpoint = pdb_breakpoint
    builtins.ipdb_breakpoint = ipdb_breakpoint

    setup_debugpy()
