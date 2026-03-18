import json
import subprocess
from dataclasses import asdict
from unittest.mock import patch

import pytest

from miles.utils.env_report import (
    ENV_REPORT_PREFIX,
    EditablePackageInfo,
    GitRepoInfo,
    NodeEnvReport,
    _collect_git_info,
    _collect_pip_info,
    _is_editable,
    _parse_pip_entry,
    collect_and_print_node_env_report,
)

_SAMPLE_PIP_INSPECT = {
    "version": "1",
    "pip_version": "24.0",
    "installed": [
        {
            "metadata": {"name": "miles", "version": "0.2.1"},
            "direct_url": {
                "url": "file:///workspace/miles",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "sglang", "version": "0.4.0"},
            "direct_url": {
                "url": "file:///workspace/sglang",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "torch", "version": "2.5.0"},
        },
        {
            "metadata": {"name": "numpy", "version": "1.26.0"},
            "direct_url": {
                "url": "https://files.pythonhosted.org/numpy-1.26.0.tar.gz",
                "archive_info": {},
            },
        },
    ],
}


class TestParsePipEntry:
    def test_normal_package(self) -> None:
        entry = _parse_pip_entry({"metadata": {"name": "torch", "version": "2.5.0"}})
        assert entry == {"name": "torch", "version": "2.5.0"}

    def test_missing_metadata(self) -> None:
        entry = _parse_pip_entry({})
        assert entry == {"name": "", "version": ""}


class TestIsEditable:
    def test_editable_package(self) -> None:
        pkg = {"direct_url": {"url": "file:///workspace/miles", "dir_info": {"editable": True}}}
        assert _is_editable(pkg) is True

    def test_non_editable_package(self) -> None:
        assert _is_editable({"metadata": {"name": "torch"}}) is False

    def test_archive_url_not_editable(self) -> None:
        pkg = {"direct_url": {"url": "https://example.com/foo.tar.gz", "archive_info": {}}}
        assert _is_editable(pkg) is False


class TestCollectPipInfo:
    def test_parses_pip_inspect_output(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info()

        assert len(full_list) == 4
        assert full_list[0] == {"name": "miles", "version": "0.2.1"}
        assert full_list[2] == {"name": "torch", "version": "2.5.0"}

        assert len(editable) == 2
        assert editable[0] == EditablePackageInfo(
            name="miles", version="0.2.1", location="/workspace/miles",
        )
        assert editable[1] == EditablePackageInfo(
            name="sglang", version="0.4.0", location="/workspace/sglang",
        )

    def test_pip_inspect_failure_returns_empty(self) -> None:
        mock_result = subprocess.CompletedResult = subprocess.CompletedProcess(
            args=["pip", "inspect"], returncode=1, stdout="", stderr="error",
        )
        with patch("miles.utils.env_report.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info()
        assert editable == []
        assert full_list == []

    def test_pip_inspect_exception_returns_empty(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", side_effect=OSError("no pip")):
            editable, full_list = _collect_pip_info()
        assert editable == []
        assert full_list == []


class TestCollectGitInfo:
    def test_collects_commit_and_diff(self, tmp_path) -> None:
        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "init", "--allow-empty"],
            capture_output=True,
            env={"GIT_AUTHOR_NAME": "test", "GIT_COMMITTER_NAME": "test",
                 "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t",
                 "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )

        info = _collect_git_info(package_name="test_pkg", location=str(tmp_path))
        assert info is not None
        assert len(info.commit) == 40
        assert info.package_name == "test_pkg"

    def test_missing_directory_returns_none(self) -> None:
        assert _collect_git_info(package_name="x", location="/nonexistent") is None

    def test_empty_location_returns_none(self) -> None:
        assert _collect_git_info(package_name="x", location="") is None

    def test_not_a_git_repo_returns_none(self, tmp_path) -> None:
        assert _collect_git_info(package_name="x", location=str(tmp_path)) is None


class TestCollectAndPrintNodeEnvReport:
    def _mock_pip_inspect(self) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(_SAMPLE_PIP_INSPECT),
            stderr="",
        )

    def test_returns_structured_report(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training", rank=0, partial_env_report='{"flavor": "test"}',
            )

        assert isinstance(report, NodeEnvReport)
        assert report.role == "training"
        assert report.rank == 0
        assert report.launcher_env_report == {"flavor": "test"}
        assert len(report.editable_packages) == 2
        assert len(report.full_pip_list) == 4

    def test_prints_single_line_json(self, capsys) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="rollout", rank=3, partial_env_report="",
            )

        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if l.startswith(ENV_REPORT_PREFIX)]
        assert len(lines) == 1
        json_str = lines[0].removeprefix(ENV_REPORT_PREFIX)
        parsed = json.loads(json_str)
        assert parsed["role"] == "rollout"
        assert parsed["rank"] == 3

    def test_empty_partial_env_report(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training", rank=0, partial_env_report="",
            )
        assert report.launcher_env_report is None

    def test_invalid_json_partial_env_report(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training", rank=0, partial_env_report="not json",
            )
        assert report.launcher_env_report is None

    def test_report_serializable(self) -> None:
        with patch("miles.utils.env_report.subprocess.run", return_value=self._mock_pip_inspect()):
            report = collect_and_print_node_env_report(
                role="training", rank=0, partial_env_report='{"x": 1}',
            )
        report_dict = asdict(report)
        json_str = json.dumps(report_dict, default=str)
        parsed = json.loads(json_str)
        assert parsed["editable_packages"][0]["name"] == "miles"
