"""Utilities for mechanical refactor verification scripts.

Usage in transform.py (gist):

    import sys
    sys.path.append(".claude/skills/mechanical-refactor-verify")
    from utils import MechanicalVerifier

    verifier = MechanicalVerifier(
        base_commit="<sha>",
        target_commit="<sha>",
    )
    verifier.run(transform)

    def transform(root: Path, run: RunFn) -> None:
        # ... your transformation logic here ...
"""

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

type RunFn = Callable[..., str]


def _run(cmd: str, cwd: str | None = None, check: bool = True) -> str:
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


class MechanicalVerifier:
    def __init__(self, base_commit: str, target_commit: str) -> None:
        self.base_commit = base_commit
        self.target_commit = target_commit

    def run(self, transform: Callable[[Path, RunFn], None]) -> None:
        repo_root = _run("git rev-parse --show-toplevel")
        worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
        branch_name = f"verify-mechanical-{self.base_commit[:8]}"

        try:
            print(f"[1/3] Creating worktree at {self.base_commit[:8]}...")
            _run(
                f"git worktree add -b {branch_name} {worktree_dir} {self.base_commit}",
                cwd=repo_root,
            )

            print("[2/3] Running transformation...")
            transform(Path(worktree_dir), _run)

            print(f"[3/3] Diffing against {self.target_commit[:8]}...")
            diff = _run(
                f"git diff {self.target_commit} -- .",
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
