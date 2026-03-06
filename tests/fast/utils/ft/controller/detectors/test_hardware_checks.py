"""Tests for hardware_checks edge cases and boundary conditions."""
from miles.utils.ft.controller.detectors.hardware_checks import (
    check_critical_xid,
    check_disk_fault,
    check_majority_nic_down,
)
from miles.utils.ft.metric_names import NODE_FILESYSTEM_AVAIL_BYTES, NODE_NETWORK_UP, XID_CODE_RECENT
from miles.utils.ft.models import MetricSample
from tests.fast.utils.ft.conftest import make_fake_metric_store


class TestCheckCriticalXid:
    def test_non_critical_xid_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "999"}, value=1.0),
        ])

        result = check_critical_xid(store)

        assert result == []

    def test_critical_xid_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "48"}, value=1.0),
        ])

        result = check_critical_xid(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "48" in result[0].reason

    def test_no_xid_data_returns_empty(self) -> None:
        store = make_fake_metric_store()

        result = check_critical_xid(store)

        assert result == []

    def test_custom_critical_xid_codes(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=XID_CODE_RECENT, labels={"xid": "999"}, value=1.0),
        ])

        result = check_critical_xid(store, critical_xid_codes=frozenset({999}))
        assert len(result) == 1
        assert result[0].node_id == "node-0"



class TestCheckMajorityNicDown:
    def test_exactly_half_nics_down_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
        ])

        result = check_majority_nic_down(store)
        assert result == []

    def test_majority_nics_down_triggers(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=0.0),
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth2"}, value=1.0),
        ])

        result = check_majority_nic_down(store)
        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_all_nics_up_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
            MetricSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=1.0),
        ])

        result = check_majority_nic_down(store)
        assert result == []

    def test_empty_metric_store_returns_empty(self) -> None:
        store = make_fake_metric_store()
        result = check_majority_nic_down(store)
        assert result == []


class TestCheckDiskFault:
    def test_below_threshold_returns_fault(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=500e6),
        ])

        result = check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "disk space low" in result[0].reason

    def test_above_threshold_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=100e9),
        ])

        assert check_disk_fault(store) == []

    def test_empty_store_returns_empty(self) -> None:
        store = make_fake_metric_store()

        assert check_disk_fault(store) == []

    def test_multiple_nodes_only_low_ones_flagged(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=200e6),
        ])
        store.ingest_samples(target_id="node-1", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=50e9),
        ])

        result = check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_custom_threshold(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=5e9),
        ])

        result_default = check_disk_fault(store)
        assert result_default == []

        result_high = check_disk_fault(store, disk_available_threshold_bytes=10e9)
        assert len(result_high) == 1
        assert result_high[0].node_id == "node-0"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(target_id="node-0", samples=[
            MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/"}, value=1e9),
        ])

        result = check_disk_fault(store, disk_available_threshold_bytes=1e9)
        assert result == []
