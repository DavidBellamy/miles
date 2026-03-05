from __future__ import annotations

import os

from miles.utils.ft.controller.diagnostics._nccl_utils import run_nccl_test
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models import DiagnosticResult


class InterMachineCommDiagnostic(BaseDiagnostic):
    """Two-node inter-machine communication diagnostic.

    Runs ``all_gather_perf`` on one side of a 2-node pair and compares
    the measured bus bandwidth against an expected baseline.  Peer
    coordination is handled via MASTER_ADDR / MASTER_PORT environment
    variables (NCCL TCPStore rendezvous).
    """

    diagnostic_type = "inter_machine"

    def __init__(
        self,
        expected_bandwidth_gbps: float = 40.0,
        num_gpus: int = 8,
        master_addr: str = "",
        master_port: int = 29500,
        nccl_test_binary: str = "all_gather_perf",
    ) -> None:
        self._expected_bandwidth_gbps = expected_bandwidth_gbps
        self._num_gpus = num_gpus
        self._master_addr = master_addr
        self._master_port = master_port
        self._nccl_test_binary = nccl_test_binary

    async def run(
        self, node_id: str, timeout_seconds: int = 180,
    ) -> DiagnosticResult:
        cmd = [
            self._nccl_test_binary,
            "-b", "1M", "-e", "1G", "-f", "2",
            "-g", str(self._num_gpus),
        ]

        env = {**os.environ}
        if self._master_addr:
            env["MASTER_ADDR"] = self._master_addr
        env["MASTER_PORT"] = str(self._master_port)

        return await run_nccl_test(
            cmd=cmd,
            node_id=node_id,
            diagnostic_type=self.diagnostic_type,
            expected_bandwidth_gbps=self._expected_bandwidth_gbps,
            timeout_seconds=timeout_seconds,
            log_prefix="inter_machine",
            env=env,
        )
