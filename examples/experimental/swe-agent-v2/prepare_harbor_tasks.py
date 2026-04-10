"""
Convert training data to Harbor task directories (generic fallback).

Reads a Miles JSONL (produced by ``download_and_process_data.py``) and
creates one Harbor task directory per instance.  Each task directory is
self-contained — Harbor treats all tasks identically regardless of
their origin (SWE-bench, Terminal-Bench, custom, etc.).

For standard benchmarks, prefer using Harbor's official adapters or
``harbor run -d <dataset>`` to generate task directories — they
produce the exact grading harness used upstream.  This script is a
generic fallback for custom datasets.

Usage:

    python prepare_harbor_tasks.py \\
        --input /root/custom_train.jsonl \\
        --output /root/harbor_tasks/ \\
        --docker-network swe-net

Required metadata fields per record:
    - instance_id:  unique task identifier (becomes directory name)

Optional metadata fields (read if present):
    - problem_statement / instruction / prompt:  task text -> instruction.md
    - docker_image:   base Docker image (default: ubuntu:24.04)
    - setup_commands: extra Dockerfile RUN commands (str or list)
    - test_script:    content of tests/test.sh
    - timeout:        verifier timeout in seconds (default: 1800)
    - repo, version:  included in task.toml if present
    - patch:          oracle solution -> solution/solve.sh
"""

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)


# ── SWE-bench auto-detection ────────────────────────────────────────────
def _is_swebench_instance(metadata: dict) -> bool:
    """Return True if metadata looks like a SWE-bench instance."""
    return all(metadata.get(k) for k in ("repo", "version", "base_commit", "test_patch"))


def _swebench_docker_image(instance_id: str) -> str:
    """Derive the pre-built SWE-bench Docker image from instance_id.

    Image naming convention (xingyaoww registry):
        instance_id:  getmoto__moto-7365
        image:        xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest

    The ``__`` in the instance_id maps to ``_s_`` in the image name.
    """
    slug = instance_id.replace("__", "_s_")
    return f"xingyaoww/sweb.eval.x86_64.{slug}:latest"


def _get_instruction(metadata: dict) -> str:
    for key in ("problem_statement", "instruction", "prompt"):
        val = metadata.get(key, "")
        if val:
            return val
    return ""


def _create_task_dir(
    instance_id: str,
    metadata: dict,
    output_dir: Path,
    docker_network: str | None = None,
) -> Path:
    """Create a Harbor task directory from metadata."""
    task_dir = output_dir / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(_get_instruction(metadata))

    repo = metadata.get("repo", "")
    version = metadata.get("version", "")
    timeout = metadata.get("timeout", 1800)

    toml_lines = [
        "[task]",
        f'name = "{instance_id}"',
    ]
    if repo:
        toml_lines.append(f'repo = "{repo}"')
    if version:
        toml_lines.append(f'version = "{version}"')
    toml_lines += [
        "",
        "[limits]",
        f"timeout = {timeout}",
    ]
    (task_dir / "task.toml").write_text("\n".join(toml_lines) + "\n")

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)

    # Auto-detect SWE-bench instances and derive the correct Docker image
    is_swebench = not metadata.get("docker_image") and _is_swebench_instance(metadata)
    if is_swebench:
        docker_image = _swebench_docker_image(instance_id)
        logger.debug(f"SWE-bench auto-detected: {instance_id} -> {docker_image}")
        extra_lines = "WORKDIR /testbed\nRUN mkdir -p /logs\n"
    else:
        docker_image = metadata.get("docker_image", "ubuntu:24.04")
        extra_lines = ""
    setup_cmds = metadata.get("setup_commands", "")
    if isinstance(setup_cmds, list):
        setup_cmds = " && ".join(setup_cmds)
    setup_block = f"RUN {setup_cmds}\n" if setup_cmds else ""

    (env_dir / "Dockerfile").write_text(f"FROM {docker_image}\n{extra_lines}{setup_block}")

    if docker_network:
        compose_yaml = textwrap.dedent(
            f"""\
            services:
              main:
                networks:
                  - {docker_network}
            networks:
              {docker_network}:
                external: true
        """
        )
        (env_dir / "docker-compose.yaml").write_text(compose_yaml)

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    test_script = metadata.get("test_script", "")
    if test_script:
        test_sh = f"#!/bin/bash\n{test_script}\n"
    else:
        test_sh = textwrap.dedent(
            """\
            #!/bin/bash
            echo 0 > /logs/verifier/reward.txt
        """
        )

    (tests_dir / "test.sh").write_text(test_sh)
    os.chmod(tests_dir / "test.sh", 0o755)

    patch = metadata.get("patch", "")
    if patch:
        sol_dir = task_dir / "solution"
        sol_dir.mkdir(exist_ok=True)
        (sol_dir / "fix.patch").write_text(patch)
        solve_sh = textwrap.dedent(
            """\
            #!/bin/bash
            git apply "$(dirname "$0")/fix.patch"
        """
        )
        (sol_dir / "solve.sh").write_text(solve_sh)
        os.chmod(sol_dir / "solve.sh", 0o755)

    return task_dir


def convert(
    input_path: str,
    output_dir: str,
    docker_network: str | None = None,
) -> int:
    """Convert all instances from JSONL to Harbor task directories.

    Returns the number of tasks created.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: {e}")
                continue

            metadata = data.get("metadata", data)
            instance_id = metadata.get("instance_id", "")
            if not instance_id:
                logger.warning(f"Skipping line {line_num}: no instance_id")
                continue

            records.append(metadata)

    if not records:
        logger.warning("No valid records found")
        return 0

    count = 0
    for metadata in records:
        instance_id = metadata["instance_id"]
        _create_task_dir(
            instance_id,
            metadata,
            output_path,
            docker_network=docker_network,
        )
        count += 1
        if count % 100 == 0:
            logger.info(f"Created {count} task directories...")

    logger.info(f"Created {count} task directories in {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert training JSONL to Harbor task directories",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Harbor tasks",
    )
    parser.add_argument(
        "--docker-network",
        default=None,
        help="External Docker network for containers to join " "(e.g. swe-net)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    convert(args.input, args.output, docker_network=args.docker_network)


if __name__ == "__main__":
    main()
