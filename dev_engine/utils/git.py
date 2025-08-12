import os
import os.path as osp
from dev_engine.system import run_cmd


def save_git_status(output_dir):
    os.makedirs(osp.join(output_dir, "git"), exist_ok=True)
    run_cmd(
        f"git rev-parse HEAD > {osp.join(output_dir, 'git', 'commit.txt')}",
        fault_tolerant=True,
    )
    run_cmd("git reset", fault_tolerant=True)
    run_cmd("git add -N **/*.py", fault_tolerant=True)
    run_cmd("git add -N **/*.sh", fault_tolerant=True)
    run_cmd("git add -N **/*.fish", fault_tolerant=True)
    run_cmd("git add -N configs/**/*.yaml", fault_tolerant=True)
    run_cmd(
        f"git diff > {osp.join(output_dir, 'git', 'patch.diff')}", fault_tolerant=True
    )
