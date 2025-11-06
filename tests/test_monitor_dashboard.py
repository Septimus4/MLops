import importlib
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from monitor import data_access


def _build_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    events = []

    class _Column:
        def metric(self, name, value, delta=None):
            events.append(("metric", name, value, delta))

    def columns(count: int):
        events.append(("columns", count))
        return [_Column() for _ in range(count)]

    def info(message: str):
        events.append(("info", message))

    def line_chart(data):
        events.append(("line_chart", list(getattr(data, "columns", []))))

    def bar_chart(data):
        events.append(("bar_chart", list(getattr(data, "columns", []))))

    def write(message: str):
        events.append(("write", message))

    def title(message: str):
        events.append(("title", message))

    def set_page_config(**kwargs):
        events.append(("set_page_config", kwargs))

    def button(label: str) -> bool:
        events.append(("button", label))
        state = module._button_state
        module._button_state = False
        return state

    def warning(message: str, icon: str | None = None):
        events.append(("warning", message, icon))

    def error(message: str):
        events.append(("error", message))

    module.columns = columns
    module.info = info
    module.line_chart = line_chart
    module.bar_chart = bar_chart
    module.write = write
    module.title = title
    module.set_page_config = set_page_config
    module.button = button
    module.warning = warning
    module.error = error
    module.components = types.SimpleNamespace(
        v1=types.SimpleNamespace(
            html=lambda html, height=None, scrolling=None: events.append(
                ("html", len(html))
            )
        )
    )
    module._events = events
    module._button_state = False
    return module


@pytest.fixture
def stub_streamlit():
    module = _build_streamlit_stub()
    sys.modules["streamlit"] = module
    try:
        yield module
    finally:
        sys.modules.pop("streamlit", None)


@pytest.fixture
def stub_evidently():
    metric_module = types.ModuleType("evidently.metric_preset")

    class FakePreset:
        def __init__(self, columns):
            self.columns = list(columns)

    metric_module.DataDriftPreset = FakePreset

    report_module = types.ModuleType("evidently.report")

    class FakeReport:
        last_metrics = None
        last_run_reference = None
        last_run_current = None
        last_saved_path = None
        last_summary = None

        def __init__(self, metrics):
            FakeReport.last_metrics = metrics
            self._columns = []

        def run(self, reference_data, current_data):
            FakeReport.last_run_reference = reference_data.copy()
            FakeReport.last_run_current = current_data.copy()
            self._columns = list(current_data.columns)

        def save_html(self, path: str):
            destination = Path(path)
            destination.write_text("<html>report</html>", encoding="utf-8")
            FakeReport.last_saved_path = str(destination)

        def as_dict(self):
            column_metrics = {
                column: {"drift_detected": True if idx % 2 == 0 else False}
                for idx, column in enumerate(self._columns)
            }
            summary = {"metrics": [{"result": {"column_metrics": column_metrics}}]}
            FakeReport.last_summary = summary
            return summary

    report_module.Report = FakeReport

    sys.modules["evidently.metric_preset"] = metric_module
    sys.modules["evidently.report"] = report_module
    try:
        yield FakeReport
    finally:
        sys.modules.pop("evidently.metric_preset", None)
        sys.modules.pop("evidently.report", None)


def test_load_logs_df_parses_recent_records(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    now = datetime.now(timezone.utc)
    recent_record = {
        "timestamp": now.isoformat(),
        "request_id": "req-1",
        "status": "ok",
        "latency_ms": 120,
        "score": 0.7,
        "binary_decision": 1,
        "model_version": "v1",
        "monitor_features": {
            "AMT_INCOME_TOTAL": 150000,
            "CODE_GENDER": "F",
        },
    }
    stale_record = {
        "timestamp": (now - timedelta(days=10)).isoformat(),
        "request_id": "req-stale",
        "status": "ok",
        "latency_ms": 90,
        "score": 0.3,
        "binary_decision": 0,
        "model_version": "v1",
        "monitor_features": {
            "AMT_INCOME_TOTAL": 90000,
            "CODE_GENDER": "M",
        },
    }

    log_file = log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(recent_record) + "\n")
        handle.write(json.dumps(stale_record) + "\n")
        handle.write("not json\n")

    df = data_access.load_logs_df(log_dir=log_dir, max_days=7)

    assert len(df) == 1
    assert df.iloc[0]["request_id"] == "req-1"
    assert "AMT_INCOME_TOTAL" in df.columns
    assert df.iloc[0]["timestamp"].tzinfo is not None


def test_load_reference_sample_filters_columns(tmp_path, monkeypatch):
    path = tmp_path / "reference.parquet"
    path.write_text("")

    sample_df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100000, 120000],
            "CODE_GENDER": ["F", "M"],
            "UNUSED_COLUMN": [1, 2],
        }
    )

    def fake_read_parquet(target):
        assert Path(target) == path
        return sample_df

    monkeypatch.setattr(data_access.pd, "read_parquet", fake_read_parquet)

    result = data_access.load_reference_sample(
        path=path, columns=["AMT_INCOME_TOTAL", "CODE_GENDER", "MISSING"]
    )

    assert list(result.columns) == ["AMT_INCOME_TOTAL", "CODE_GENDER"]
    assert len(result) == 2


def test_load_reference_stats_reads_parquet(tmp_path, monkeypatch):
    path = tmp_path / "feature_stats.parquet"
    path.write_text("")

    stats_df = pd.DataFrame({"feature": ["score"], "mean": [0.5]})

    def fake_read_parquet(target):
        assert Path(target) == path
        return stats_df

    monkeypatch.setattr(data_access.pd, "read_parquet", fake_read_parquet)

    result = data_access.load_reference_stats(path=path)

    assert result.equals(stats_df)


def test_make_drift_report_creates_html(tmp_path, monkeypatch, stub_evidently):
    from api import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "reference"))

    sys.modules.pop("monitor.drift_report", None)
    drift_report = importlib.import_module("monitor.drift_report")

    current = pd.DataFrame(
        {"AMT_INCOME_TOTAL": [1000.0, 1200.0], "EXT_SOURCE_1": [0.1, 0.2]}
    )
    reference = pd.DataFrame(
        {"AMT_INCOME_TOTAL": [950.0, 1100.0], "EXT_SOURCE_1": [0.15, 0.18]}
    )

    output = drift_report.make_drift_report(current, reference, min_rows=1)

    assert output.exists()
    assert output.read_text(encoding="utf-8") == "<html>report</html>"

    fake_report = stub_evidently
    assert fake_report.last_saved_path == str(output)
    assert fake_report.last_run_reference.equals(
        reference[["AMT_INCOME_TOTAL", "EXT_SOURCE_1"]]
    )
    assert fake_report.last_run_current.equals(
        current[["AMT_INCOME_TOTAL", "EXT_SOURCE_1"]]
    )
    assert fake_report.last_metrics[0].columns == ["AMT_INCOME_TOTAL", "EXT_SOURCE_1"]
    summary_path = output.parent / "latest_drift_report.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["drifted_columns"] == 1
    assert summary["dropped_features"] == []
    assert summary["current_row_count"] == 2
    assert summary["reference_row_count"] == 2
    assert summary["features"] == ["AMT_INCOME_TOTAL", "EXT_SOURCE_1"]


def test_make_drift_report_requires_coverage(tmp_path, monkeypatch, stub_evidently):
    from api import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "reference"))

    sys.modules.pop("monitor.drift_report", None)
    drift_report = importlib.import_module("monitor.drift_report")

    current = pd.DataFrame(
        {"AMT_INCOME_TOTAL": [1000.0, None, None], "EXT_SOURCE_1": [0.1, None, None]}
    )
    reference = current.copy()

    with pytest.raises(ValueError, match="coverage"):
        drift_report.make_drift_report(
            current,
            reference,
            min_rows=3,
            min_feature_coverage=0.75,
        )


def test_monitor_dashboard_main_renders(
    tmp_path, monkeypatch, stub_streamlit, stub_evidently
):
    from api import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "reference"))

    sys.modules.pop("monitor.drift_report", None)
    sys.modules.pop("monitor.app", None)

    monitor_app = importlib.import_module("monitor.app")
    object.__setattr__(
        monitor_app.settings, "monitor_features", ["AMT_INCOME_TOTAL", "CODE_GENDER"]
    )
    features = list(monitor_app.settings.monitor_features)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    logs_data = {
        "timestamp": timestamps,
        "status": ["ok", "error", "ok", "ok"],
        "latency_ms": [100.0, 200.0, 150.0, 120.0],
        "score": [0.2, 0.9, 0.6, 0.4],
        "binary_decision": [0, 1, 1, 0],
        "model_version": ["v1"] * 4,
    }

    for feature in features:
        if feature not in logs_data:
            if feature == "CODE_GENDER":
                logs_data[feature] = ["F", "M", "F", "M"]
            elif feature == "NAME_CONTRACT_TYPE":
                logs_data[feature] = ["Cash loans"] * 4
            else:
                logs_data[feature] = [1.0, 2.0, 3.0, 4.0]

    logs_df = pd.DataFrame(logs_data)
    reference_df = pd.DataFrame(
        {feature: [logs_df[feature].iloc[0]] for feature in features}
    )

    monkeypatch.setattr(
        monitor_app.data_access,
        "load_logs_df",
        lambda log_dir=None, max_days=7: logs_df,
    )
    monkeypatch.setattr(
        monitor_app.data_access,
        "load_reference_sample",
        lambda path=None, columns=None: reference_df,
    )

    stub_streamlit._button_state = True

    monitor_app.main()

    events = stub_streamlit._events
    assert ("title", "Home Credit Scoring – Monitoring Dashboard") in events
    assert any(event[0] == "line_chart" for event in events)
    assert any(event[0] == "bar_chart" for event in events)
    assert any(event[0] == "html" for event in events)
