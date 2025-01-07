import getopt
import importlib.util
import os
import runpy
import sys
import os
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.getcwd()))

import debugpy
from termcolor import colored


def setup_debugpy(endpoint="localhost", port=5678, ranks=[0]):
    if "DEBUGPY" not in os.environ:
        return

    # Overwrite with environment variables
    if "DEBUGPY_RANKS" in os.environ:
        ranks = [int(r) for r in os.environ["DEBUGPY_RANKS"].split(",")]
    port = int(os.getenv("DEBUGPY_PORT", port))
    endpoint = os.getenv("DEBUGPY_ENDPOINT", endpoint)
    rank = int(os.getenv("RANK", 0))
    if rank not in ranks:
        return
    port += rank

    pid = os.getpid()

    try:
        debugpy.listen((endpoint, port))
        print(colored(f"Waiting for debugger attach on {endpoint}:{port} (process {pid})", "red"))
        debugpy.wait_for_client()
    except Exception as e:
        print(
            colored(f"Failed to setup debugpy on {endpoint}:{port} (process {pid}). Error: {e}", "red")
        )


def main():
    opts, args = getopt.getopt(sys.argv[1:], "mhc:", ["help", "command="])

    if not args and not any(opt in ("-c", "--command") for opt, _ in opts):
        print("No script specified")
        sys.exit(2)

    run_as_module = False
    command = None
    for opt, optarg in opts:
        if opt == "-m":
            run_as_module = True
        elif opt in ("-c", "--command"):
            command = optarg
        elif opt in ("-h", "--help"):
            print("Usage: script.py [options] [script_or_module]")
            sys.exit()
    
    if command is not None:
        sys.argv = ["-c"] + args  # Adjust sys.argv for exec
        exec(command, {"__name__": "__main__"})
    elif run_as_module:
        if not args:
            print("No module specified")
            sys.exit(2)
        module_name = args[0]
        sys.argv = args  # Adjust sys.argv appropriately

        # Find the module's file path
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            print(f"Cannot find module {module_name}")
            sys.exit(1)
        module_file = spec.origin
        sys.path[0] = os.path.dirname(module_file)
        runpy.run_path(module_file, run_name="__main__")
    else:
        if not args:
            print("No script specified")
            sys.exit(2)
        mainpyfile = args[0]
        mainpyfile = os.path.realpath(mainpyfile)
        sys.argv = args  # Adjust sys.argv for the script
        sys.path[0] = os.path.dirname(mainpyfile)
        runpy.run_path(mainpyfile, run_name="__main__")


if __name__ == "__main__":
    setup_debugpy()
    main()
