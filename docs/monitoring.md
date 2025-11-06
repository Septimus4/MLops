# Monitoring & Drift

The monitoring stack combines structured request logs, lightweight metrics aggregation, and Evidently-based drift detection to surface issues quickly.

## Log Collection

- Every inference request produces a JSON record (`api/logging_utils.log_prediction`) under `data/logs/YYYY-MM-DD.jsonl`.
- Each record captures timestamps, latency, score, binary decision, model version, and the subset of features configured via `Settings.monitor_features`.
- Logs can be compacted to Parquet with `python -m scripts.compaction --log-dir data/logs --output-dir data/metrics/daily`.

## Metrics API

`GET /metrics` aggregates log data over a configurable window (default 1440 minutes) and returns:

- `request_count` / `error_count`
- `avg_latency_ms` / `p95_latency_ms`
- `mean_score`
- `drift_alerts` (count of features currently flagged by Evidently)

The API reads the latest drift summary from `data/metrics/drift/latest_drift_report.json`, so dashboards and alerting remain consistent.

## Streamlit Dashboard

Launch the dashboard locally:

```zsh
streamlit run monitor/app.py
```

Features:

- KPI tiles (requests, errors, latency, approval rate).
- Latency chart resampled over 15-minute buckets.
- Score distribution histogram.
- “Run drift report” button that triggers Evidently analysis and embeds the resulting HTML.

When deployed (for example, as a container pulled from GHCR), the dashboard consumes the same log directory mounted as a volume or synced via object storage.

## Drift Detection Pipeline

`monitor/drift_report.make_drift_report`:

1. Loads the current window of log-derived features and the reference dataset.
2. Runs Evidently’s `DataDriftPreset` on overlapping columns.
3. Saves an HTML report to `data/metrics/drift/latest_drift_report.html`.
4. Writes a JSON summary containing the number of drifted columns plus raw metrics (`latest_drift_report.json`).

Use `python -m monitor.drift_report` (or call the helper function) to regenerate reports on a schedule. The Streamlit UI provides a manual trigger for ad-hoc analysis.

## Reference Data Management

`scripts/make_reference_snapshot.py` captures baseline statistics and samples into `data/reference/`. Refresh the snapshot whenever the training distribution changes (e.g. new cohort, retrained model).

## Alerting Hooks

For production environments, consider:

- Shipping `log_prediction` output to a log sink (CloudWatch, ELK) in addition to JSONL.
- Emitting metrics to Prometheus via `/metrics` scraping or by wiring a push gateway.
- Automating drift report generation and notifying on `drift_alerts > 0`.
