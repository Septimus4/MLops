"""Streamlit dashboard for monitoring inference behaviour."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
import sys

import pandas as pd

try:  # pragma: no cover - optional dependency
    st = import_module("streamlit")  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - fail fast when missing
    raise RuntimeError(
        "Streamlit must be installed to launch the monitoring dashboard"
    ) from exc

try:
    from api.config import get_settings
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if (
        str(PROJECT_ROOT) not in sys.path
    ):  # ensure sibling packages (e.g. api) are resolvable
        sys.path.insert(0, str(PROJECT_ROOT))
    from api.config import get_settings

from monitor import data_access
from monitor.drift_report import make_drift_report
from monitor.metrics import latency_series, score_distribution, summarise_kpis

settings = get_settings()


def _render_kpis(df: pd.DataFrame) -> None:
    kpis = summarise_kpis(df)
    cols = st.columns(3)
    cols[0].metric("Requests", kpis["request_count"], delta=None)
    cols[0].metric("Errors", kpis["error_count"], delta=None)
    cols[1].metric("Approval rate", f"{kpis['approval_rate'] * 100:.1f}%")
    cols[1].metric("Avg latency", f"{kpis['avg_latency_ms']:.1f} ms")
    cols[2].metric("p95 latency", f"{kpis['p95_latency_ms']:.1f} ms")
    cols[2].metric("Mean score", f"{kpis['mean_score']:.3f}")


def _render_latency(df: pd.DataFrame) -> None:
    series = latency_series(df)
    if series.empty:
        st.info("No latency data available yet.")
    else:
        st.line_chart(series.set_index("timestamp"))


def _render_scores(df: pd.DataFrame) -> None:
    distribution = score_distribution(df)
    if distribution.empty:
        st.info("No score data available yet.")
    else:
        st.bar_chart(distribution.set_index("bucket"))


def _render_drift(df: pd.DataFrame, reference: pd.DataFrame) -> None:
    st.write("## Drift analysis")
    if df.empty or reference.empty:
        st.info("Need both live traffic and reference data to compute drift.")
        return

    features = [
        col
        for col in settings.monitor_features
        if col in df.columns and col in reference.columns
    ]
    if not features:
        st.info("No overlapping monitor features found in logs and reference data.")
        return

    if st.button("Run drift report"):
        try:
            required_rows = max(getattr(settings, "monitor_min_sample_size", 25), 1)
            current_rows = len(df)
            reference_rows = len(reference)
            if current_rows < required_rows or reference_rows < required_rows:
                st.warning(
                    f"Reference/current windows have {reference_rows}/{current_rows} rows; "
                    f"drift thresholds expect {required_rows}+ rows. Proceeding with available data.",
                    icon="⚠️",
                )
            min_rows = max(1, min(required_rows, current_rows, reference_rows))

            path = make_drift_report(
                df,
                reference,
                features=features,
                min_rows=min_rows,
            )
            with path.open("r", encoding="utf-8") as fh:
                html = fh.read()
            st.components.v1.html(html, height=800, scrolling=True)
            summary_path = path.parent / "latest_drift_report.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                dropped = summary.get("dropped_features") or []
                if dropped:
                    st.warning(
                        f"Skipped {len(dropped)} low-coverage feature(s): {', '.join(sorted(dropped))}",
                        icon="⚠️",
                    )
        except Exception as exc:
            st.error(f"Drift computation failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Home Credit Monitoring", layout="wide")
    st.title("Home Credit Scoring – Monitoring Dashboard")

    logs_df = data_access.load_logs_df(max_days=7)
    reference_df = data_access.load_reference_sample(columns=settings.monitor_features)

    _render_kpis(logs_df)

    st.write("## Latency (average per 15 minutes)")
    _render_latency(logs_df)

    st.write("## Score distribution")
    _render_scores(logs_df)

    _render_drift(logs_df, reference_df)


if __name__ == "__main__":
    main()
