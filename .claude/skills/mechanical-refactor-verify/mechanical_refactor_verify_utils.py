"""Utilities for mechanical refactor verification scripts.

See SKILL.md for usage and transform script template.
"""

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Optional


def exec_command(cmd: str, cwd: Optional[str] = None, check: bool = True) -> str:
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_add_and_commit(message: str, cwd: str) -> None:
    exec_command(f"git add -A && git commit -m '{message}'", cwd=cwd)


def verify_mechanical_refactor(
    base_commit: str,
    target_commit: str,
    transform: "Callable[[Path], None]",
) -> None:
    repo_root = exec_command("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{base_commit[:8]}"

    try:
        print(f"[1/3] Creating worktree at {base_commit[:8]}...")
        exec_command(
            f"git worktree add -b {branch_name} {worktree_dir} {base_commit}",
            cwd=repo_root,
        )

        print("[2/3] Running transformation...")
        transform(Path(worktree_dir))

        print(f"[3/3] Diffing against {target_commit[:8]}...")
        diff = exec_command(
            f"git diff {target_commit} -- .",
            cwd=worktree_dir,
            check=False,
        )

        if diff:
            print(f"\nFAIL: diff is non-empty:\n{diff}")
            sys.exit(1)
        else:
            print("\nPASS: transform reproduces the commit exactly.")

    finally:
        print(f"\nWorktree left at: {worktree_dir}")
        print(f"Branch: {branch_name}")
        print("To clean up manually:")
        print(f"  git worktree remove {worktree_dir} && git branch -D {branch_name}")
