"""Tests for miles.utils.ft.cli.diagnostics.output."""

from __future__ import annotations

import json

import pytest
import typer

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.cli.diagnostics.output import exit_with_results, print_results, validate_check_names


def _pass_result(diagnostic_type: str = "check_a") -> DiagnosticResult:
    return DiagnosticResult(
        diagnostic_type=diagnostic_type,
        node_id="node-0",
        passed=True,
        details="all good",
    )


def _fail_result(diagnostic_type: str = "check_b") -> DiagnosticResult:
    return DiagnosticResult(
        diagnostic_type=diagnostic_type,
        node_id="node-0",
        passed=False,
        details="something wrong",
    )


class TestPrintResults:
    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [_pass_result(), _fail_result()]

        print_results(results, json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2
        assert data[0]["passed"] is True
        assert data[1]["passed"] is False

    def test_human_readable_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [_pass_result(), _fail_result()]

        print_results(results, json_output=False, node_id="test-node")

        captured = capsys.readouterr()
        assert "test-node" in captured.out
        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "1 FAIL" in captured.out
        assert "1 PASS" in captured.out

    def test_all_pass_human_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [_pass_result(), _pass_result("check_c")]

        print_results(results, json_output=False)

        captured = capsys.readouterr()
        assert "0 FAIL" in captured.out
        assert "2 PASS" in captured.out


class TestExitWithResults:
    def test_has_failure_raises_system_exit(self) -> None:
        results = [_pass_result(), _fail_result()]

        with pytest.raises(SystemExit) as exc_info:
            exit_with_results(results)

        assert exc_info.value.code == 1

    def test_all_pass_no_exception(self) -> None:
        results = [_pass_result(), _pass_result("check_c")]

        exit_with_results(results)

    def test_empty_results_no_exception(self) -> None:
        exit_with_results([])


class TestValidateCheckNames:
    def test_unknown_names_raises_typer_exit(self) -> None:
        with pytest.raises(typer.Exit):
            validate_check_names(
                selected=["known", "unknown_x"],
                available=["known", "other"],
            )

    def test_all_known_no_exception(self) -> None:
        validate_check_names(
            selected=["check_a", "check_b"],
            available=["check_a", "check_b", "check_c"],
        )

    def test_empty_selection_no_exception(self) -> None:
        validate_check_names(selected=[], available=["check_a"])
