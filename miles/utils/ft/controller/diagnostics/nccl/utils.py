"""Backward-compatibility re-export. Implementation moved to agents.diagnostics.nccl.utils."""
from miles.utils.ft.agents.diagnostics.nccl.utils import (
    build_nccl_test_cmd,
    parse_avg_bus_bandwidth,
    run_nccl_test,
)

__all__ = ["build_nccl_test_cmd", "parse_avg_bus_bandwidth", "run_nccl_test"]
