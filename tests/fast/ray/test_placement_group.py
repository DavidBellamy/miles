from __future__ import annotations

from miles.ray.placement_group import _parse_excluded_node_ids


class TestParseExcludedNodeIds:
    def test_empty_string_returns_none(self) -> None:
        assert _parse_excluded_node_ids("") is None

    def test_blank_string_returns_none(self) -> None:
        assert _parse_excluded_node_ids("  ") is None

    def test_single_id(self) -> None:
        assert _parse_excluded_node_ids("abc123") == {"abc123"}

    def test_multiple_ids(self) -> None:
        result = _parse_excluded_node_ids("abc123,def456,ghi789")
        assert result == {"abc123", "def456", "ghi789"}

    def test_strips_whitespace(self) -> None:
        result = _parse_excluded_node_ids(" abc123 , def456 ")
        assert result == {"abc123", "def456"}

    def test_ignores_empty_segments(self) -> None:
        result = _parse_excluded_node_ids("abc123,,def456,")
        assert result == {"abc123", "def456"}
