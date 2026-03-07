"""Fixtures for live Prometheus integration tests.

Downloads the Prometheus binary at test time (session scope), starts a
local Prometheus server (module scope), and exposes test metrics via a
local HTTP endpoint using prometheus_client (module scope).
"""

from __future__ import annotations

import logging
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest
from prometheus_client import CollectorRegistry, Gauge, start_http_server

logger = logging.getLogger(__name__)

_PROMETHEUS_VERSION = "3.5.0"
_PROMETHEUS_TARBALL = f"prometheus-{_PROMETHEUS_VERSION}.linux-amd64.tar.gz"
_PROMETHEUS_URL = (
    f"https://github.com/prometheus/prometheus/releases/download/"
    f"v{_PROMETHEUS_VERSION}/{_PROMETHEUS_TARBALL}"
)
_BINARY_DIR = Path("/tmp/prometheus_test_binary")
_BINARY_PATH = _BINARY_DIR / "prometheus"

_STARTUP_TIMEOUT_SECONDS = 15
_DOWNLOAD_TIMEOUT_SECONDS = 120


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _download_prometheus() -> Path:
    if _BINARY_PATH.exists():
        logger.info("prometheus_binary already exists at %s", _BINARY_PATH)
        return _BINARY_PATH

    _BINARY_DIR.mkdir(parents=True, exist_ok=True)
    strip_prefix = f"prometheus-{_PROMETHEUS_VERSION}.linux-amd64/prometheus"
    cmd = (
        f"curl -sL --max-time {_DOWNLOAD_TIMEOUT_SECONDS} {_PROMETHEUS_URL} "
        f"| tar xz --strip-components=1 -C {_BINARY_DIR} {strip_prefix}"
    )
    logger.info("downloading prometheus: %s", cmd)
    subprocess.run(cmd, shell=True, check=True)

    if not _BINARY_PATH.exists():
        raise FileNotFoundError(f"Prometheus binary not found after download: {_BINARY_PATH}")

    _BINARY_PATH.chmod(0o755)
    return _BINARY_PATH


def _wait_for_prometheus(url: str, timeout: float = _STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/api/v1/status/config", timeout=2.0)
            if resp.status_code == 200:
                logger.info("prometheus_ready at %s", url)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Prometheus did not become ready at {url} within {timeout}s")


@pytest.fixture(scope="session")
def prometheus_binary() -> Path:
    if platform.system() != "Linux":
        pytest.skip("Live Prometheus tests require Linux")

    return _download_prometheus()


@dataclass
class ExporterInfo:
    registry: CollectorRegistry
    port: int
    test_gauge: Gauge
    labeled_gauge: Gauge


@pytest.fixture(scope="module")
def local_exporter() -> ExporterInfo:
    registry = CollectorRegistry()
    test_gauge = Gauge(
        "test_gauge",
        "A test gauge for integration tests",
        registry=registry,
    )
    test_gauge.set(42.0)

    labeled_gauge = Gauge(
        "test_labeled_gauge",
        "A test gauge with labels",
        labelnames=["node_id", "device"],
        registry=registry,
    )
    labeled_gauge.labels(node_id="node-0", device="ib0").set(1.0)
    labeled_gauge.labels(node_id="node-0", device="ib1").set(0.0)

    port = _find_free_port()
    httpd, thread = start_http_server(port=port, registry=registry)

    yield ExporterInfo(
        registry=registry,
        port=port,
        test_gauge=test_gauge,
        labeled_gauge=labeled_gauge,
    )

    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def prometheus_server(
    prometheus_binary: Path,
    local_exporter: ExporterInfo,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    prom_port = _find_free_port()
    data_dir = tmp_path_factory.mktemp("prom_data")

    config_path = data_dir / "prometheus.yml"
    config_path.write_text(
        f"""\
global:
  scrape_interval: 1s
  evaluation_interval: 1s

scrape_configs:
  - job_name: test_exporter
    static_configs:
      - targets: ["127.0.0.1:{local_exporter.port}"]
"""
    )

    proc = subprocess.Popen(
        [
            str(prometheus_binary),
            f"--config.file={config_path}",
            f"--storage.tsdb.path={data_dir / 'tsdb'}",
            f"--web.listen-address=127.0.0.1:{prom_port}",
            "--storage.tsdb.retention.time=5m",
            "--log.level=warn",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    prom_url = f"http://127.0.0.1:{prom_port}"
    try:
        _wait_for_prometheus(prom_url)
    except TimeoutError:
        proc.terminate()
        proc.wait(timeout=5)
        raise

    yield prom_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
