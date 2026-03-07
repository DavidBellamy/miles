"""Backward-compatibility re-export. Implementation moved to agents.diagnostics.gpu_check_script."""
from miles.utils.ft.agents.diagnostics.gpu_check_script import (
    GpuCheckResult,
    _check_nvml,
    _check_single_gpu,
    _generate_matmul_reference,
    _check_matmul,
    main,
)

__all__ = [
    "GpuCheckResult",
    "_check_nvml",
    "_check_single_gpu",
    "_generate_matmul_reference",
    "_check_matmul",
    "main",
]

if __name__ == "__main__":
    main()
