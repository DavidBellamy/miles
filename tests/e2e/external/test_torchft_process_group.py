"""Explore torchft ProcessGroup behavior under various failure modes.

Mirrors how torchft Manager uses ProcessGroupNCCL/Gloo internally:
  - pg.allreduce([tensor], opts) → work
  - work.get_future() → fut.then(callback)
  - pg.errored() checked before/after operations
  - pg.configure() for setup

Run inside a Ray cluster with GPUs.

Usage:
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode os-exit
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


@ray.remote(num_gpus=1)
class _PGWorker:
    """Ray actor that holds a torchft PG and runs collectives the same way Manager does."""

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

    def run_allreduce_manager_style(self) -> dict:
        """Run allreduce exactly like torchft Manager.allreduce() does.

        Ref: torchft/manager.py Manager.allreduce() lines ~435-493
        Pattern:
          1. if self.errored(): return noop
          2. opts = AllreduceOptions(); work = pg.allreduce([tensor], opts)
          3. managed_work wraps work, calls work.get_future().then(callback)
          4. on exception: report_error, return noop
        """
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        # Step 1: check errored (Manager line 435-436)
        if (e := self._pg.errored()) is not None:
            return {
                "rank": self._rank,
                "status": "skipped_errored",
                "error": str(e),
                "elapsed_s": 0,
            }

        # Step 2: allreduce + get_future (Manager lines 466-484)
        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)

            # Manager wraps in _ManagedWork and calls get_future().then(callback)
            fut = work.get_future()
            # Block on the future (simulates what happens when optimizer.step waits)
            fut.wait()

            elapsed = time.monotonic() - start
            errored_after = self._pg.errored()
            return {
                "rank": self._rank,
                "status": "ok",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }
        except Exception as e:
            # Manager lines 487-493: report_error, return DummyWork
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }

    def run_allreduce_blocking_wait(self) -> dict:
        """Run allreduce with blocking work.wait() — how miles indep_dp uses it."""
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)
            success = work.wait()
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(self._pg.errored()),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }

    def run_allreduce_poll(self, timeout_s: float = 15.0) -> dict:
        """Run allreduce with is_completed() polling — experimental approach."""
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
                    return {
                        "rank": self._rank,
                        "status": "poll_timeout",
                        "elapsed_s": round(time.monotonic() - start, 2),
                        "errored_after": str(self._pg.errored()),
                    }
                time.sleep(0.05)

            success = work.wait()
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
            }
        except Exception as e:
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.monotonic() - start, 2),
            }

    def run_allreduce_then_die(self, *, tensor_size: int, die_after_s: float) -> None:
        """Start a large allreduce, then os._exit after a delay.

        This simulates a crash DURING an in-flight allreduce — the key scenario
        that causes ncclCommAbort to hang in production (NVLink DMA residuals).
        """
        import threading

        import torch
        import torch.distributed as dist

        def _delayed_exit() -> None:
            time.sleep(die_after_s)
            logger.warning("rank %d: os._exit after %.1fs (mid-allreduce kill)", self._rank, die_after_s)
            os._exit(1)

        threading.Thread(target=_delayed_exit, daemon=True).start()

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        while True:
            work = self._pg.allreduce([tensor], opts)
            work.wait()

    def die_os_exit(self) -> None:
        os._exit(1)

    def die_shutdown(self) -> None:
        self._pg.shutdown()

    def die_sigterm(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def get_status(self) -> dict:
        errored = None
        try:
            errored = self._pg.errored()
        except Exception as e:
            errored = f"errored() failed: {e}"
        return {"rank": self._rank, "alive": True, "errored": str(errored)}


class _WaitStyle(str, Enum):
    MANAGER = "manager"
    BLOCKING = "blocking"
    POLL = "poll"


def _run_test(
    *,
    failure_mode: FailureMode,
    backend: str,
    timeout_s: float,
    wait_style: _WaitStyle,
    poll_timeout_s: float,
    world_size: int,
) -> None:
    from torch.distributed import TCPStore

    store = TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    store_addr = f"localhost:{store.port}/test"

    print(f"\n{'='*70}")
    print(f"  backend={backend}  failure={failure_mode.value}  timeout={timeout_s}s"
          f"  wait={wait_style.value}  world_size={world_size}")
    print(f"{'='*70}\n")

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

    # Step 1: sanity check — normal allreduce
    print("\n--- Step 1: Normal allreduce (sanity) ---")
    results = ray.get([w.run_allreduce_manager_style.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    # Step 2: kill rank 0
    victim = workers[0]
    survivors = workers[1:]
    print(f"\n--- Step 2: Kill rank 0 ({failure_mode.value}) ---")

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
        print("  rank 0 shutdown PG gracefully")
    elif failure_mode == FailureMode.SIGTERM:
        try:
            ray.get(victim.die_sigterm.remote(), timeout=5)
        except Exception as e:
            print(f"  rank 0 sigterm: {type(e).__name__}")

    time.sleep(1)

    # Step 3: survivors try allreduce
    print(f"\n--- Step 3: Survivor allreduce (wait_style={wait_style.value}) ---")
    start = time.monotonic()

    if wait_style == _WaitStyle.MANAGER:
        refs = [s.run_allreduce_manager_style.remote() for s in survivors]
    elif wait_style == _WaitStyle.BLOCKING:
        refs = [s.run_allreduce_blocking_wait.remote() for s in survivors]
    else:
        refs = [s.run_allreduce_poll.remote(timeout_s=poll_timeout_s) for s in survivors]

    for ref in refs:
        try:
            result = ray.get(ref, timeout=timeout_s + 90)
            print(f"  {result}")
        except ray.exceptions.GetTimeoutError:
            elapsed = time.monotonic() - start
            print(f"  TIMEOUT: ray.get timed out after {elapsed:.0f}s — actor likely dead or hung")
        except ray.exceptions.RayActorError as e:
            elapsed = time.monotonic() - start
            print(f"  ACTOR DIED after {elapsed:.0f}s: {e}")

    # Step 4: check survivor status
    print("\n--- Step 4: Survivor status ---")
    for s in survivors:
        try:
            status = ray.get(s.get_status.remote(), timeout=10)
            print(f"  {status}")
        except Exception as e:
            print(f"  Cannot reach survivor: {type(e).__name__}: {e}")

    # Cleanup: kill all remaining actors to free GPU memory for next test
    for w in survivors:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store

    print()


app = typer.Typer()


@app.command()
def run(
    failure_mode: Annotated[FailureMode, typer.Option(help="How to kill rank 0")] = FailureMode.OS_EXIT,
    backend: Annotated[str, typer.Option(help="nccl or gloo")] = "nccl",
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    wait_style: Annotated[_WaitStyle, typer.Option(help="manager / blocking / poll")] = _WaitStyle.MANAGER,
    poll_timeout_s: Annotated[float, typer.Option(help="Poll timeout (only with --wait-style poll)")] = 15.0,
    world_size: Annotated[int, typer.Option(help="Number of ranks")] = 2,
) -> None:
    """Run a single failure-mode test."""
    ray.init(ignore_reinit_error=True)
    _run_test(
        failure_mode=failure_mode,
        backend=backend,
        timeout_s=timeout_s,
        wait_style=wait_style,
        poll_timeout_s=poll_timeout_s,
        world_size=world_size,
    )


@app.command()
def run_all(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    world_size: Annotated[int, typer.Option(help="Number of ranks")] = 2,
) -> None:
    """Run all failure modes × backends × wait styles."""
    ray.init(ignore_reinit_error=True)

    for backend in ["gloo", "nccl"]:
        for failure_mode in [FailureMode.SHUTDOWN, FailureMode.OS_EXIT, FailureMode.RAY_KILL]:
            for wait_style in [_WaitStyle.MANAGER, _WaitStyle.BLOCKING, _WaitStyle.POLL]:
                try:
                    _run_test(
                        failure_mode=failure_mode,
                        backend=backend,
                        timeout_s=timeout_s,
                        wait_style=wait_style,
                        poll_timeout_s=min(15.0, timeout_s),
                        world_size=world_size,
                    )
                except Exception as e:
                    print(f"  TEST FAILED: {e}\n")


@app.command()
def run_inflight(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    tensor_size: Annotated[int, typer.Option(help="Tensor size for allreduce (larger = longer)")] = 100_000_000,
    die_after_s: Annotated[float, typer.Option(help="Kill rank 0 after this many seconds")] = 2.0,
    wait_style: Annotated[_WaitStyle, typer.Option(help="How survivor waits")] = _WaitStyle.BLOCKING,
) -> None:
    """Test in-flight allreduce crash — simulates the miles production scenario.

    Both ranks start a continuous allreduce loop with a large tensor.
    Rank 0 os._exit's after die_after_s while allreduce is in progress.
    This tests whether ncclCommAbort hangs when NCCL kernels are in-flight on NVLink.
    """
    from torch.distributed import TCPStore

    ray.init(ignore_reinit_error=True)

    store = TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    store_addr = f"localhost:{store.port}/inflight"

    print(f"\n{'='*70}")
    print(f"  IN-FLIGHT CRASH TEST: tensor_size={tensor_size}  die_after={die_after_s}s"
          f"  timeout={timeout_s}s  wait={wait_style.value}")
    print(f"{'='*70}\n")

    workers = [_PGWorker.remote() for _ in range(2)]
    init_results = ray.get([
        w.init.remote(
            store_addr=store_addr, rank=i, world_size=2,
            backend="nccl", timeout_s=timeout_s,
        )
        for i, w in enumerate(workers)
    ])
    for r in init_results:
        print(f"  init: {r}")

    # Step 1: sanity allreduce
    print("\n--- Step 1: Sanity allreduce ---")
    results = ray.get([w.run_allreduce_blocking_wait.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    # Step 2: start continuous allreduce on both, rank 0 will die mid-flight
    print(f"\n--- Step 2: Start continuous allreduce, rank 0 dies after {die_after_s}s ---")

    victim_ref = workers[0].run_allreduce_then_die.remote(
        tensor_size=tensor_size, die_after_s=die_after_s,
    )
    # Give rank 0 time to start the allreduce loop
    time.sleep(0.5)

    # Survivor also starts allreduce (will block when peer dies)
    print("  Survivor starting allreduce...")
    start = time.monotonic()

    if wait_style == _WaitStyle.BLOCKING:
        survivor_ref = workers[1].run_allreduce_blocking_wait.remote()
    elif wait_style == _WaitStyle.MANAGER:
        survivor_ref = workers[1].run_allreduce_manager_style.remote()
    else:
        survivor_ref = workers[1].run_allreduce_poll.remote(timeout_s=min(15.0, timeout_s))

    # Wait for victim to confirm death
    try:
        ray.get(victim_ref, timeout=die_after_s + 10)
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  Rank 0 confirmed dead after {elapsed:.1f}s: {type(e).__name__}")

    # Step 3: wait for survivor result
    print(f"\n--- Step 3: Waiting for survivor (timeout={timeout_s + 90}s) ---")
    try:
        result = ray.get(survivor_ref, timeout=timeout_s + 90)
        elapsed = time.monotonic() - start
        print(f"  Survivor result after {elapsed:.1f}s: {result}")
    except ray.exceptions.GetTimeoutError:
        elapsed = time.monotonic() - start
        print(f"  TIMEOUT: survivor hung for {elapsed:.0f}s — ABORT HANG CONFIRMED")
    except ray.exceptions.RayActorError as e:
        elapsed = time.monotonic() - start
        print(f"  ACTOR DIED after {elapsed:.0f}s: {e}")

    # Step 4: check survivor
    print("\n--- Step 4: Survivor status ---")
    try:
        status = ray.get(workers[1].get_status.remote(), timeout=10)
        print(f"  {status}")
    except Exception as e:
        print(f"  Cannot reach survivor: {type(e).__name__}: {e}")

    # Cleanup
    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print()


if __name__ == "__main__":
    app()
