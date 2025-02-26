import subprocess
import os.path as osp
import os

HOMEBREW_BIN = "/mnt/amlfs-01/home/jingwang/homebrew/bin"


def listdir_fd(
    path: str, pattern: str = None, postfix: str = None, num_threads: int = 64
):
    fd_executable = osp.join(HOMEBREW_BIN, "fd")
    path = osp.abspath(path)
    if pattern is not None:
        stdout = run_cmd(
            f"{fd_executable} -j {num_threads} {pattern} --full-path {path}"
        ).stdout
    elif postfix is not None:
        stdout = run_cmd(
            f"{fd_executable} -j {num_threads} . {path} -e {postfix}"
        ).stdout
    else:
        raise ValueError("Either pattern or postfix must be provided")
    return stdout.splitlines()


def run_cmd(
    cmd: str,
    verbose: bool = False,
    async_cmd: bool = False,
    fault_tolerance: bool = False,
) -> subprocess.CompletedProcess:
    # print(cmd)
    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        lines = []
        for line in popen.stdout:
            line = line.rstrip().decode("utf-8")
            print(line)
            lines.append(line)
        popen.wait()
        if popen.returncode != 0 and not fault_tolerance:
            raise RuntimeError(
                f"Failed to run command: {cmd}\nERROR {popen.stderr}\nSTDOUT{popen.stdout}"
            )
        popen.stdout = "\n".join(lines)
        return popen
    else:
        if not async_cmd:
            # decode bug fix: https://stackoverflow.com/questions/73545218/utf-8-encoding-exception-with-subprocess-run
            ret = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, encoding="cp437"
            )
            if ret.returncode != 0 and not fault_tolerance:
                raise RuntimeError(
                    f"Failed to run command: {cmd}\nERROR {ret.stderr}\nSTDOUT{ret.stdout}"
                )
            return ret
        else:
            popen = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return popen
