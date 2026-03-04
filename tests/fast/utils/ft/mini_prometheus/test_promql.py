from datetime import timedelta

from miles.utils.ft.controller.mini_prometheus.promql import (
    CompareExpr,
    CompareOp,
    LabelMatchOp,
    MetricSelector,
    RangeFunction,
    RangeFunctionCompare,
    parse_promql,
)


class TestParsePromQL:
    def test_simple_metric_name(self) -> None:
        expr = parse_promql("ft_node_gpu_available")
        assert isinstance(expr, MetricSelector)
        assert expr.name == "ft_node_gpu_available"
        assert expr.matchers == []

    def test_metric_with_label_filter(self) -> None:
        expr = parse_promql('ft_node_xid_code_recent{xid="48"}')
        assert isinstance(expr, MetricSelector)
        assert expr.name == "ft_node_xid_code_recent"
        assert len(expr.matchers) == 1
        assert expr.matchers[0].label == "xid"
        assert expr.matchers[0].op == LabelMatchOp.EQ
        assert expr.matchers[0].value == "48"

    def test_metric_with_neq_label(self) -> None:
        expr = parse_promql('gpu_available{node_id!="node-0"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.NEQ

    def test_metric_with_regex_label(self) -> None:
        expr = parse_promql('gpu_available{node_id=~"node-.*"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.RE
        assert expr.matchers[0].value == "node-.*"

    def test_compare_eq(self) -> None:
        expr = parse_promql("ft_node_gpu_available == 0")
        assert isinstance(expr, CompareExpr)
        assert expr.selector.name == "ft_node_gpu_available"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_compare_gt(self) -> None:
        expr = parse_promql("gpu_temperature_celsius > 90")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.GT
        assert expr.threshold == 90.0

    def test_compare_lte(self) -> None:
        expr = parse_promql("disk_available_bytes <= 1000000")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.LTE

    def test_range_function_count_over_time(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "count_over_time"
        assert expr.selector.name == "nic_alert"
        assert expr.duration == timedelta(minutes=5)

    def test_range_function_changes(self) -> None:
        expr = parse_promql("changes(training_iteration[10m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "changes"
        assert expr.duration == timedelta(minutes=10)

    def test_range_function_with_compare(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m]) >= 2")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "count_over_time"
        assert expr.op == CompareOp.GTE
        assert expr.threshold == 2.0

    def test_changes_with_compare(self) -> None:
        expr = parse_promql("changes(training_iteration[10m]) == 0")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "changes"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_range_function_with_labels(self) -> None:
        expr = parse_promql('count_over_time(xid_code_recent{xid="48"}[5m])')
        assert isinstance(expr, RangeFunction)
        assert expr.selector.name == "xid_code_recent"
        assert len(expr.selector.matchers) == 1
        assert expr.selector.matchers[0].value == "48"

    def test_duration_seconds(self) -> None:
        expr = parse_promql("count_over_time(metric[30s])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(seconds=30)

    def test_duration_hours(self) -> None:
        expr = parse_promql("count_over_time(metric[1h])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(hours=1)
