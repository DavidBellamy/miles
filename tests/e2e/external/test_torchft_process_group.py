"""Explore torchft ProcessGroup behavior under various failure modes.

Run inside a Ray cluster with GPUs. Uses Ray actors to simulate multi-rank
NCCL/Gloo groups and tests what happens when one rank dies in different ways.

Usage:
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode os-exit
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode ray-kill
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode shutdown
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode sigterm
    python tests/e2e/external/test_torchft_process_group.py run-all
"""

import logging
import os
import signal
import time
from datetime import timedelta
from enum import Enum
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    OS_EXIT = "os-exit"
    RAY_KILL = "ray-kill"
    SHUTDOWN = "shutdown"
    SIGTERM = "sigterm"
    EXCEPTION = "exception"
    NONE = "none"


@ray.remote(num_gpus=1)
class _PGWorker:
    """A Ray actor that holds a torchft ProcessGroup and runs collectives."""

    def init(
        self,
        *,
        store_addr: str,
        rank: int,
        world_size: int,
        backend: str,
        timeout_s: float,
    ) -> dict:
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._rank = rank
        self._backend = backend
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        pg_cls = ProcessGroupNCCL if backend == "nccl" else ProcessGroupGloo
        self._pg = pg_cls(timeout=timedelta(seconds=timeout_s))
        self._pg.configure(
            store_addr=store_addr,
            replica_id=str(rank),
            rank=rank,
            world_size=world_size,
            quorum_id=0,
        )

        return {"rank": rank, "backend": backend, "device": str(self._device)}

    def run_allreduce(self) -> dict:
        """Run a blocking allreduce and return timing + result."""
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()
        errored_before = self._pg.errored()

        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)
            success = work.wait()
            elapsed = time.monotonic() - start
            errored_after = self._pg.errored()
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
                "errored_before": str(errored_before),
                "errored_after": str(errored_after),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() itself failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "errored_before": str(errored_before),
                "errored_after": str(errored_after),
            }

    def run_allreduce_poll(self, timeout_s: float = 30.0, poll_interval_s: float = 0.05) -> dict:
        """Run allreduce with is_completed() polling instead of blocking wait."""
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)

            inner_work = getattr(work, "_work", work)
            deadline = time.monotonic() + timeout_s
            while inner_work is not None and not inner_work.is_completed():
                if time.monotonic() > deadline:
                    elapsed = time.monotonic() - start
                    return {
                        "rank": self._rank,
                        "status": "poll_timeout",
                        "elapsed_s": round(elapsed, 2),
                    }
                time.sleep(poll_interval_s)

            success = work.wait()
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
            }

    def die_os_exit(self) -> None:
        os._exit(1)

    def die_shutdown(self) -> None:
        self._pg.shutdown()

    def die_sigterm(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def die_exception(self) -> None:
        raise RuntimeError("simulated crash")

    def get_status(self) -> dict:
        errored = None
        try:
            errored = self._pg.errored()
        except Exception as e:
            errored = f"errored() failed: {e}"
        return {"rank": self._rank, "errored": str(errored)}


def _get_free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_test(
    *,
    failure_mode: FailureMode,
    backend: str,
    timeout_s: float,
    use_poll: bool,
    poll_timeout_s: float,
) -> None:
    port = _get_free_port()
    store_addr = f"localhost:{port}/test"
    world_size = 2

    print(f"\n{'='*60}")
    print(f"  backend={backend}  failure={failure_mode.value}  timeout={timeout_s}s  poll={use_poll}")
    print(f"{'='*60}\n")

    workers = [_PGWorker.remote() for _ in range(world_size)]

    init_results = ray.get([
        w.init.remote(
            store_addr=store_addr,
            rank=i,
            world_size=world_size,
            backend=backend,
            timeout_s=timeout_s,
        )
        for i, w in enumerate(workers)
    ])
    for r in init_results:
        print(f"  init: {r}")

    # Step 1: sanity — normal allreduce should work
    print("\n--- Step 1: Normal allreduce ---")
    results = ray.get([w.run_allreduce.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    # Step 2: kill rank 0
    print(f"\n--- Step 2: Kill rank 0 ({failure_mode.value}) ---")
    victim = workers[0]
    survivor = workers[1]

    if failure_mode == FailureMode.OS_EXIT:
        try:
            ray.get(victim.die_os_exit.remote(), timeout=5)
        except Exception as e:
            print(f"  rank 0 died: {type(e).__name__}")
    elif failure_mode == FailureMode.RAY_KILL:
        ray.kill(victim, no_restart=True)
        print("  rank 0 killed via ray.kill")
    elif failure_mode == FailureMode.SHUTDOWN:
        ray.get(victim.die_shutdown.remote())
        print("  rank 0 shutdown PG")
    elif failure_mode == FailureMode.SIGTERM:
        try:
            ray.get(victim.die_sigterm.remote(), timeout=5)
        except Exception as e:
            print(f"  rank 0 sigterm: {type(e).__name__}")
    elif failure_mode == FailureMode.EXCEPTION:
        try:
            ray.get(victim.die_exception.remote())
        except Exception as e:
            print(f"  rank 0 exception: {type(e).__name__}")
    elif failure_mode == FailureMode.NONE:
        print("  (no kill)")

    time.sleep(1)

    # Step 3: survivor tries allreduce
    print(f"\n--- Step 3: Survivor allreduce ({'poll' if use_poll else 'blocking'}) ---")
    start = time.monotonic()

    if use_poll:
        ref = survivor.run_allreduce_poll.remote(timeout_s=poll_timeout_s)
    else:
        ref = survivor.run_allreduce.remote()

    try:
        result = ray.get(ref, timeout=timeout_s + 60)
        print(f"  {result}")
    except ray.exceptions.GetTimeoutError:
        elapsed = time.monotonic() - start
        print(f"  TIMEOUT: ray.get timed out after {elapsed:.0f}s — actor likely dead or hung")
    except ray.exceptions.RayActorError as e:
        elapsed = time.monotonic() - start
        print(f"  ACTOR DIED after {elapsed:.0f}s: {e}")

    # Step 4: check survivor status
    print("\n--- Step 4: Survivor status ---")
    try:
        status = ray.get(survivor.get_status.remote(), timeout=10)
        print(f"  {status}")
    except Exception as e:
        print(f"  Cannot reach survivor: {type(e).__name__}: {e}")

    print()


app = typer.Typer()


@app.command()
def run(
    failure_mode: Annotated[FailureMode, typer.Option(help="How to kill rank 0")] = FailureMode.OS_EXIT,
    backend: Annotated[str, typer.Option(help="nccl or gloo")] = "nccl",
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    use_poll: Annotated[bool, typer.Option(help="Use is_completed() polling instead of blocking wait")] = False,
    poll_timeout_s: Annotated[float, typer.Option(help="Poll timeout in seconds (only with --use-poll)")] = 15.0,
) -> None:
    """Run a single failure-mode test."""
    ray.init(ignore_reinit_error=True)
    _run_test(
        failure_mode=failure_mode,
        backend=backend,
        timeout_s=timeout_s,
        use_poll=use_poll,
        poll_timeout_s=poll_timeout_s,
    )


@app.command()
def run_all(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
) -> None:
    """Run all failure modes × backends × blocking/poll."""
    ray.init(ignore_reinit_error=True)

    for backend in ["gloo", "nccl"]:
        for failure_mode in [FailureMode.SHUTDOWN, FailureMode.OS_EXIT, FailureMode.RAY_KILL]:
            for use_poll in [False, True]:
                try:
                    _run_test(
                        failure_mode=failure_mode,
                        backend=backend,
                        timeout_s=timeout_s,
                        use_poll=use_poll,
                        poll_timeout_s=min(15.0, timeout_s),
                    )
                except Exception as e:
                    print(f"  TEST FAILED: {e}\n")


if __name__ == "__main__":
    app()
