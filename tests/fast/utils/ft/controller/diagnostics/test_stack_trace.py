"""Tests for StackTraceAggregator."""

from __future__ import annotations

from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceAggregator
from tests.fast.utils.ft.helpers import (
    SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
    SAMPLE_PYSPY_OUTPUT_NORMAL,
    SAMPLE_PYSPY_OUTPUT_STUCK,
)


class TestStackTraceAggregatorBasic:
    def test_empty_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        assert agg.aggregate(traces={}) == []

    def test_single_node_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        result = agg.aggregate(traces={"node-0": SAMPLE_PYSPY_OUTPUT_NORMAL})
        assert result == []

    def test_all_same_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_OUTPUT_NORMAL,
            "node-1": SAMPLE_PYSPY_OUTPUT_NORMAL,
            "node-2": SAMPLE_PYSPY_OUTPUT_NORMAL,
        }
        result = agg.aggregate(traces=traces)
        assert result == []


class TestStackTraceAggregatorSuspectDetection:
    def test_one_different_node_is_suspect(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-1": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-2": SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == ["node-2"]

    def test_two_nodes_all_different_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-1": SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == []

    def test_multiple_minority_groups(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_OUTPUT_NORMAL,
            "node-1": SAMPLE_PYSPY_OUTPUT_NORMAL,
            "node-2": SAMPLE_PYSPY_OUTPUT_NORMAL,
            "node-3": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-4": SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert "node-3" in result
        assert "node-4" in result
        assert "node-0" not in result


class TestStackTraceAggregatorFingerprint:
    def test_fingerprint_ignores_line_numbers(self) -> None:
        agg = StackTraceAggregator()
        trace_a = """\
Thread 0x1 (active): "MainThread"
    func_a (file.py:10)
    func_b (file.py:20)
"""
        trace_b = """\
Thread 0x1 (active): "MainThread"
    func_a (file.py:99)
    func_b (file.py:88)
"""
        fp_a = agg._extract_fingerprint(trace_a)
        fp_b = agg._extract_fingerprint(trace_b)
        assert fp_a == fp_b

    def test_fingerprint_differs_on_function_name(self) -> None:
        agg = StackTraceAggregator()
        trace_a = """\
Thread 0x1 (active): "MainThread"
    func_a (file.py:10)
"""
        trace_b = """\
Thread 0x1 (active): "MainThread"
    func_DIFFERENT (file.py:10)
"""
        fp_a = agg._extract_fingerprint(trace_a)
        fp_b = agg._extract_fingerprint(trace_b)
        assert fp_a != fp_b

    def test_real_pyspy_output_parsing(self) -> None:
        agg = StackTraceAggregator()
        fp = agg._extract_fingerprint(SAMPLE_PYSPY_OUTPUT_NORMAL)
        assert fp != ""
        assert "(" in fp

    def test_fingerprint_captures_innermost_frame(self) -> None:
        agg = StackTraceAggregator()
        trace = """\
Thread 0x1 (active): "MainThread"
    innermost_func (inner.py:1)
    middle_func (mid.py:2)
    outermost_func (outer.py:3)
"""
        fp = agg._extract_fingerprint(trace)
        assert "innermost_func" in fp
        assert "outermost_func" not in fp

    def test_tied_groups_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-1": SAMPLE_PYSPY_OUTPUT_STUCK,
            "node-2": SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
            "node-3": SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == []
