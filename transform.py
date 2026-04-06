#!/usr/bin/env python3
"""Reproducible transform for: split miles/ray/rollout.py into miles/ray/rollout/ package

Run from the repo root:  python3 transform.py
"""
import sys
import textwrap
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import exec_command, git_add_and_commit

BASE_COMMIT = "4dd7770ed8caf59e45f387c5af7061e5c7e2cc41"
TARGET_COMMIT = "118423b7a"

DIFF_PATHS = [
    "miles/ray/rollout.py",
    "miles/ray/rollout/",
    ".gitignore",
]


def _lines(L: list[str], start: int, end: int) -> str:
    """Extract lines start..end (1-indexed, inclusive) from L."""
    return "".join(L[start - 1 : end])


def _dedent4(text: str) -> str:
    """Remove exactly 4 leading spaces from each line."""
    out = []
    for line in text.splitlines(keepends=True):
        if line.startswith("    "):
            out.append(line[4:])
        elif line.strip() == "":
            out.append(line)
        else:
            out.append(line)
    return "".join(out)


def transform(dir_root: Path) -> None:
    source = dir_root / "miles/ray/rollout.py"
    content = source.read_text()
    L = content.splitlines(keepends=True)

    pkg = dir_root / "miles/ray/rollout"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()

    # === server_group.py ===
    # ServerGroup class: lines 61-208
    body = _lines(L, 61, 208)
    body = body.replace("_allocate_rollout_engine_addr_and_ports_external", "allocate_rollout_engine_addr_and_ports_external")
    body = body.replace("_allocate_rollout_engine_addr_and_ports_normal", "allocate_rollout_engine_addr_and_ports_normal")
    (pkg / "server_group.py").write_text(
        "import dataclasses\n"
        "import os\n"
        "from typing import Any\n"
        "\n"
        "import ray\n"
        "from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy\n"
        "\n"
        "from miles.backends.sglang_utils.sglang_engine import SGLangEngine\n"
        "from miles.ray.rollout.addr_allocator import (\n"
        "    allocate_rollout_engine_addr_and_ports_external,\n"
        "    allocate_rollout_engine_addr_and_ports_normal,\n"
        ")\n"
        "from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST\n"
        "from miles.utils import dumper_utils\n"
        "\n"
        "\n"
        + body
    )

    # === addr_allocator.py ===
    # allocate_rollout_engine_addr_and_ports_normal: lines 810-897
    # allocate_rollout_engine_addr_and_ports_external: lines 796-808 (includes trailing blank line)
    normal_body = _lines(L, 810, 897)
    normal_body = normal_body.replace("def _allocate_rollout_engine_addr_and_ports_normal", "def allocate_rollout_engine_addr_and_ports_normal")
    ext_body = _lines(L, 796, 807)
    ext_body = ext_body.replace("def _allocate_rollout_engine_addr_and_ports_external", "def allocate_rollout_engine_addr_and_ports_external")
    (pkg / "addr_allocator.py").write_text(
        "import logging\n"
        "\n"
        "import ray\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        + normal_body + "\n\n"
        + ext_body
    )

    # === router_manager.py ===
    # start_router: lines 905-964
    # start_session_server: lines 1099-1133
    router_body = _lines(L, 905, 964)
    router_body = router_body.replace("def _start_router(", "def start_router(")
    session_body = _lines(L, 1099, 1133)
    session_body = session_body.replace("def _start_session_server(", "def start_session_server(")
    (pkg / "router_manager.py").write_text(
        "import logging\n"
        "import multiprocessing\n"
        "import random\n"
        "\n"
        "\n"
        "from miles.utils.http_utils import (\n"
        "    _wrap_ipv6,\n"
        "    find_available_port,\n"
        "    get_host_info,\n"
        "    is_port_available,\n"
        "    wait_for_server_ready,\n"
        ")\n"
        "\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        + router_body + "\n\n"
        + session_body
    )

    # === metrics.py ===
    log_eval = _lines(L, 1136, 1166)
    log_eval = log_eval.replace("def _log_eval_rollout_data(", "def log_eval_rollout_data(")
    log_eval = log_eval.replace("compute_metrics_from_samples(", "_compute_metrics_from_samples(")

    log_rollout = _lines(L, 1169, 1184)
    log_rollout = log_rollout.replace("def _log_rollout_data(", "def log_rollout_data(")
    log_rollout = log_rollout.replace("compute_metrics_from_samples(", "_compute_metrics_from_samples(")
    log_rollout = log_rollout.replace("compute_perf_metrics_from_samples(", "_compute_perf_metrics_from_samples(")

    compute_metrics = _lines(L, 1187, 1218)
    compute_metrics = compute_metrics.replace("def compute_metrics_from_samples(", "def _compute_metrics_from_samples(")

    perf_metrics = _lines(L, 1221, 1251)
    perf_metrics = perf_metrics.replace("def compute_perf_metrics_from_samples(", "def _compute_perf_metrics_from_samples(")

    zero_std = _lines(L, 1254, 1268)
    spec = _lines(L, 1271, 1278)
    prefix_cache = _lines(L, 1281, 1289)
    reward_cat = _lines(L, 1292, 1299)

    (pkg / "metrics.py").write_text(
        "import logging\n"
        "from typing import Any\n"
        "\n"
        "import numpy as np\n"
        "\n"
        "from miles.utils import tracking_utils\n"
        "from miles.utils.iter_utils import group_by\n"
        "from miles.utils.metric_utils import (\n"
        "    compute_pass_rate,\n"
        "    compute_rollout_step,\n"
        "    compute_statistics,\n"
        "    dict_add_prefix,\n"
        "    has_repetition,\n"
        ")\n"
        "from miles.utils.misc import load_function\n"
        "from miles.utils.types import Sample\n"
        "\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        + log_eval + "\n\n"
        + log_rollout + "\n\n"
        + compute_metrics + "\n\n"
        + perf_metrics + "\n\n"
        + zero_std + "\n\n"
        + spec + "\n\n"
        + prefix_cache + "\n\n"
        + reward_cat
    )

    # === debug_data.py ===
    # _save_debug_rollout_data body: lines 621-637 (method body, needs dedent)
    debug_body = _dedent4(_lines(L, 621, 637))
    (pkg / "debug_data.py").write_text(
        "import logging\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "# TODO extract `load_debug_rollout_data`\n"
        "\n"
        "\n"
        "# TODO: remove `self`\n"
        "def save_debug_rollout_data(self, data, rollout_id, evaluation: bool):\n"
        + debug_body
    )

    # === train_data_conversion.py ===
    # These are extracted class methods -> standalone functions, need dedent
    # convert_samples_to_train_data body: lines 667-734
    convert_body = _dedent4(_lines(L, 667, 734))
    convert_body = convert_body.replace("self._post_process_rewards(", "_post_process_rewards(self, ")

    # _post_process_rewards body: lines 640-664
    post_process_body = _dedent4(_lines(L, 640, 664))

    # split_train_data_by_dp body: lines 740-788
    split_body = _dedent4(_lines(L, 740, 788))

    (pkg / "train_data_conversion.py").write_text(
        "import ray\n"
        "import torch\n"
        "\n"
        "from miles.utils.ray_utils import Box\n"
        "from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions\n"
        "from miles.utils.types import Sample\n"
        "\n"
        "\n"
        "# TODO: remove `self`\n"
        "def convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):\n"
        + convert_body
        + "\n\n"
        "# TODO: remove `self`\n"
        "def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):\n"
        + post_process_body
        + "\n\n"
        "# TODO: remove `self`\n"
        "def split_train_data_by_dp(self, data, dp_size):\n"
        + split_body
    )

    # === rollout_server.py ===
    # start_rollout_servers: lines 991-1069
    start_servers = _lines(L, 991, 1069)
    start_servers = start_servers.replace("_start_router(", "start_router(")
    # Forward reference: only in the return type annotation, not in local variables
    start_servers = start_servers.replace(") -> dict[str, RolloutServer]:", ') -> dict[str, "RolloutServer"]:')

    # _resolve_sglang_config: lines 1072-1091
    resolve_config = _lines(L, 1072, 1091)

    # _compute_rollout_offset: lines 967-976
    compute_offset = _lines(L, 967, 976)

    # _compute_megatron_num_gpus: lines 979-988
    compute_megatron = _lines(L, 979, 988)

    # RolloutServer class: lines 211-325
    rollout_server_class = _lines(L, 211, 325)

    (pkg / "rollout_server.py").write_text(
        "import dataclasses\n"
        "import logging\n"
        "\n"
        "import ray\n"
        "from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS\n"
        "\n"
        "from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig\n"
        "from miles.ray.rollout.router_manager import start_router\n"
        "from miles.ray.rollout.server_group import ServerGroup\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        + start_servers + "\n\n"
        + resolve_config + "\n\n"
        + compute_offset + "\n\n"
        + compute_megatron + "\n\n"
        + rollout_server_class
    )

    # === rollout_manager.py ===
    # RolloutManager class: lines 333-618 (through _compute_dynamic_global_batch_size)
    # + set_train_parallel_config: lines 736-737
    # Excludes: _save_debug_rollout_data (620-637), _post_process_rewards (639-664),
    #           _convert_samples_to_train_data (666-734), _split_train_data_by_dp (739-788)
    manager_body = _lines(L, 333, 618)
    set_train = _lines(L, 736, 737)
    manager_body += "\n" + set_train

    manager_body = manager_body.replace("self._save_debug_rollout_data(", "save_debug_rollout_data(self, ")
    manager_body = manager_body.replace("_log_rollout_data(", "log_rollout_data(")
    manager_body = manager_body.replace("_log_eval_rollout_data(", "log_eval_rollout_data(")
    manager_body = manager_body.replace("self._convert_samples_to_train_data(", "convert_samples_to_train_data(self, ")
    manager_body = manager_body.replace("self._split_train_data_by_dp(", "split_train_data_by_dp(self, ")
    manager_body = manager_body.replace("_start_session_server(", "start_session_server(")
    manager_body = manager_body.replace(
        "    def _try_ci_fault_injection(self):",
        "    # TODO will be replaced by full ft\n    def _try_ci_fault_injection(self):",
    )
    # Add TODO comment before load_debug_rollout_data
    manager_body = manager_body.replace(
        "        if self.args.load_debug_rollout_data:\n            data = torch.load(",
        "        if self.args.load_debug_rollout_data:\n            # TODO extract to `load_debug_rollout_data`\n            data = torch.load(",
    )

    (pkg / "rollout_manager.py").write_text(
        "import itertools\n"
        "import logging\n"
        "import time\n"
        "\n"
        "import ray\n"
        "import torch\n"
        "\n"
        "from miles.ray.rollout.debug_data import save_debug_rollout_data\n"
        "from miles.ray.rollout.metrics import log_eval_rollout_data, log_rollout_data\n"
        "from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers\n"
        "from miles.ray.rollout.router_manager import start_session_server\n"
        "from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp\n"
        "from miles.ray.utils import Lock\n"
        "from miles.rollout.base_types import (\n"
        "    RolloutFnConstructorInput,\n"
        "    RolloutFnEvalInput,\n"
        "    RolloutFnTrainInput,\n"
        "    call_rollout_fn,\n"
        ")\n"
        "from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function\n"
        "from miles.utils.environ import enable_experimental_rollout_refactor\n"
        "from miles.utils.health_monitor import RolloutHealthMonitor\n"
        "from miles.utils.http_utils import init_http_client\n"
        "from miles.utils.logging_utils import configure_logger\n"
        "from miles.utils.metric_checker import MetricChecker\n"
        "from miles.utils.misc import load_function\n"
        "from miles.utils.tracking_utils import init_tracking\n"
        "from miles.utils.types import Sample\n"
        "\n"
        "logging.getLogger(\"httpx\").setLevel(logging.WARNING)\n"
        "logging.getLogger(\"httpcore\").setLevel(logging.WARNING)\n"
        "\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        + manager_body
    )

    # Remove the original file
    source.unlink()

    # Fix .gitignore: remove `.claude/` line
    gitignore = dir_root / ".gitignore"
    gi = gitignore.read_text()
    gi = gi.replace(".claude/\n", "")
    gitignore.write_text(gi)

    git_add_and_commit("split rollout.py into rollout/ package", cwd=str(dir_root))


def verify() -> None:
    """Custom verification that only diffs the relevant paths."""
    import tempfile

    repo_root = exec_command("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{BASE_COMMIT[:8]}"

    try:
        print(f"[1/3] Creating worktree at {BASE_COMMIT[:8]}...")
        exec_command(
            f"git worktree add -b {branch_name} {worktree_dir} {BASE_COMMIT}",
            cwd=repo_root,
        )

        print("[2/3] Running transformation...")
        transform(Path(worktree_dir))

        print(f"[3/3] Diffing against {TARGET_COMMIT[:8]}...")
        diff_paths = " ".join(DIFF_PATHS)
        diff = exec_command(
            f"git diff {TARGET_COMMIT} -- {diff_paths}",
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


if __name__ == "__main__":
    verify()
