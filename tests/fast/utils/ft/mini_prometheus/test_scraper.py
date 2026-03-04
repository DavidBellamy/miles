from miles.utils.ft.controller.mini_prometheus.scraper import parse_prometheus_text


class TestParsePrometheusText:
    def test_simple_metric(self) -> None:
        text = "gpu_temperature_celsius 75.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temperature_celsius"
        assert samples[0].value == 75.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = 'gpu_temperature_celsius{gpu="0",node="n1"} 82.5\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].labels == {"gpu": "0", "node": "n1"}
        assert samples[0].value == 82.5

    def test_skips_comments_and_help(self) -> None:
        text = (
            "# HELP gpu_temp GPU temperature\n"
            "# TYPE gpu_temp gauge\n"
            "gpu_temp 42.0\n"
        )
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temp"

    def test_multiple_metrics(self) -> None:
        text = (
            "metric_a 1.0\n"
            "metric_b 2.0\n"
            "metric_c{label=\"x\"} 3.0\n"
        )
        samples = parse_prometheus_text(text)
        assert len(samples) == 3

    def test_metric_with_timestamp(self) -> None:
        text = "http_requests_total 1000 1700000000000\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == 1000.0
