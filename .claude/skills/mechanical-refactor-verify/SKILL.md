---
name: mechanical-refactor-verify
description: Verify mechanical refactoring commits by requiring a reproducible transform script (gist) in the PR description. Use when doing or reviewing file splits, function moves, or module extractions.
user_invocable: true
argument: "[verify <pr_url_or_commit>] — verify an existing PR, or omit to see the workflow guide"
---

# Mechanical Refactor — Reproducible Verification

## Core Principle

The deliverable of a mechanical move (file split, function move, module extraction) is NOT the diff — it is **the script that produces the diff**.
A script is auditable; a diff is not.

## Workflow

Regardless of who did the move (human or agent) and when (before or after committing), the workflow is the same:

### Step 1: Write the transform script

Write a **fully self-contained** Python script that:

1. Creates a git worktree at the base commit on a temp branch
2. Performs the mechanical transformation (may produce multiple commits, e.g. "move files", "fix imports", "format")
3. Diffs the final result against the PR's target commit
4. Reports pass/fail
5. Leaves the worktree for inspection (prints cleanup command)

If the move is already done, reverse-engineer the script by reading the before state (`git show <base>:<file>`) and the after state, then encode the transformation.

Requirements:

- **No external dependencies**: stdlib + git CLI only
- **Idempotent**: safe to run multiple times
- **Self-verifying**: the script itself checks the diff and reports the result

Script template:

```python
#!/usr/bin/env python3
"""Reproducible transform script for: <describe the mechanical move>

Verifies that the mechanical refactoring in this PR is reproducible.
Run from anywhere inside the repo.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_COMMIT = "<base_sha>"
TARGET_COMMIT = "<pr_mechanical_move_final_sha>"

def run(cmd: str, cwd: str | None = None, check: bool = True) -> str:
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()

def main() -> None:
    repo_root = run("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{BASE_COMMIT[:8]}"

    try:
        print(f"[1/4] Creating worktree at {BASE_COMMIT[:8]}...")
        run(f"git worktree add -b {branch_name} {worktree_dir} {BASE_COMMIT}",
            cwd=repo_root)

        print("[2/4] Running transformation...")
        transform(worktree_dir)

        print(f"[3/4] Diffing against {TARGET_COMMIT[:8]}...")
        diff = run(
            f"git diff {TARGET_COMMIT} -- .",
            cwd=worktree_dir, check=False,
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

def transform(worktree: str) -> None:
    """Perform the mechanical transformation and commit each step."""
    root = Path(worktree)

    # --- Step 1: Split source file into target files ---
    source = root / "path/to/source.py"
    content = source.read_text()
    lines = content.splitlines(keepends=True)

    splits = [
        ("path/to/pkg/target_a.py", 1, 50),
        ("path/to/pkg/target_b.py", 51, 120),
    ]
    for target_path, start, end in splits:
        target = root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[start - 1 : end]))

    source.unlink()
    (root / "path/to/pkg/__init__.py").touch()

    run("git add -A && git commit -m 'mechanical: split source.py into pkg/'",
        cwd=worktree)

    # --- Step 2: Fix imports ---
    # <make import changes here>
    # run("git add -A && git commit -m 'fix imports'", cwd=worktree)

    # --- Step 3: Format ---
    # run("ruff format .", cwd=worktree)
    # run("git add -A && git commit -m 'fmt'", cwd=worktree)

if __name__ == "__main__":
    main()
```

### Step 2: Run locally to verify

```bash
python3 transform.py
# Expected output: "PASS: transform reproduces the commit exactly."
```

### Step 3: Upload gist and add to PR description

Always upload the script as a gist (never inline in PR body).

```bash
gh gist create --public -d "Mechanical refactor transform: <description>" transform.py
```

PR description format:

````markdown
## Mechanical Move

Transform script: <gist_url>

### One-click verification

```bash
python3 <(curl -sL <gist_raw_url>)
```
````

Anyone verifying the PR just copy-pastes the one-liner.

### Step 4: Non-mechanical changes go in subsequent commits

After the mechanical move commits, separate commits handle semantic changes (rename symbols, restructure APIs, etc.). These are standard code review — no script verification needed.

## Verifying an existing PR (`/mechanical-refactor-verify verify`)

1. Find the gist URL and one-click command in the PR description
2. Run the one-click command in the repo
3. Report: PASS or show the diff

## Relationship to existing refactor workflows

- `simpledev-refactor-execute` (global skill): uses rope to execute moves, produces two commits
- This skill: **verification layer** — regardless of how the move was done (rope, manual, sed), require a reproducible script as a gist
- They compose: rope does the move → wrap the rope commands in a transform script → upload as gist
