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

The scaffold (worktree creation, diff check, cleanup) lives in `utils.py` next to this skill. The transform script only needs to define the `transform()` function and call `MechanicalVerifier`.

Script template:

```python
#!/usr/bin/env python3
"""Reproducible transform for: <describe the mechanical move>

Run from the repo root:  python3 transform.py
"""
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import verify_mechanical_refactor, exec_command, git_add_and_commit

BASE_COMMIT = "<base_sha>"
TARGET_COMMIT = "<pr_mechanical_move_final_sha>"


def transform(dir_root: Path) -> None:
    """Perform the mechanical transformation and commit each step.

    Args:
        dir_root: Path to the worktree (checked out at BASE_COMMIT).
    """
    # --- Step 1: Split source file ---
    source = dir_root / "path/to/source.py"
    content = source.read_text()
    lines = content.splitlines(keepends=True)

    splits = [
        ("path/to/pkg/target_a.py", 1, 50),
        ("path/to/pkg/target_b.py", 51, 120),
    ]
    for target_path, start, end in splits:
        target = dir_root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[start - 1 : end]))

    source.unlink()
    (dir_root / "path/to/pkg/__init__.py").touch()

    git_add_and_commit("mechanical: split source.py", cwd=str(dir_root))

    # --- Step 2: Fix imports ---
    # <edit files>
    # git_add_and_commit("fix imports", cwd=str(dir_root))

    # Note: formatting (ruff format) is handled automatically by verify_mechanical_refactor


if __name__ == "__main__":
    verify_mechanical_refactor(
        base_commit=BASE_COMMIT,
        target_commit=TARGET_COMMIT,
        transform=transform,
    )
```

The `transform()` function:
- Receives `dir_root` (worktree path, checked out at base commit)
- Makes file changes and commits each logical step
- The verifier handles worktree creation, diffing against target, and reporting pass/fail

### Step 2: Run locally to verify

```bash
python3 transform.py
# Expected: "PASS: transform reproduces the commit exactly."
```

### Step 3: Upload gist and add to PR description

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
