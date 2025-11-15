"""
Streamlit dashboard for drift monitoring.
"""

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# API URL (configurable via environment variable)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def get_drift_metrics(window_hours: int):
    """Fetch drift metrics from API."""
    try:
        response = requests.get(
            f"{API_URL}/drift", params={"window_hours": window_hours}, timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching drift metrics: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Please ensure the API service is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_health():
    """Fetch health status from API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


# Page configuration
st.set_page_config(page_title="Drift Monitoring Dashboard", page_icon="📊", layout="wide")

# Title
st.title("📊 Feature Drift Monitoring Dashboard")
st.markdown("Monitor feature drift in real-time using z-score analysis")

# Sidebar controls
st.sidebar.header("Settings")

# Window hours selector
window_hours = st.sidebar.selectbox(
    "Time Window",
    options=[1, 6, 12, 24, 48, 72, 168],
    index=3,  # Default to 24 hours
    format_func=lambda x: f"{x} hours" if x < 24 else f"{x//24} days",
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)

# Refresh button
if st.sidebar.button("🔄 Refresh Now") or auto_refresh:
    st.rerun()

# Display API status
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
health = get_health()
if health:
    st.sidebar.success("✅ Connected")
    st.sidebar.text(f"Model: {health.get('model_version', 'unknown')}")
else:
    st.sidebar.error("❌ Disconnected")

# Main content
col1, col2, col3 = st.columns(3)

# Fetch drift metrics
drift_data = get_drift_metrics(window_hours)

if drift_data:
    metrics_list = drift_data.get("metrics", [])
    num_samples = drift_data.get("num_samples", 0)

    # Display summary metrics
    with col1:
        st.metric("Time Window", f"{window_hours}h")

    with col2:
        st.metric("Predictions", num_samples)

    with col3:
        if metrics_list:
            max_drift = max(m["z_score"] for m in metrics_list)
            st.metric("Max Drift (z-score)", f"{max_drift:.2f}")
        else:
            st.metric("Max Drift (z-score)", "N/A")

    # Check if we have data
    if not metrics_list:
        st.warning(
            f"No predictions found in the last {window_hours} hours. Make some predictions to see drift metrics."
        )
    else:
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)

        # Sort by z_score descending
        df = df.sort_values("z_score", ascending=False)

        # Drift interpretation
        st.markdown("---")
        st.subheader("📈 Drift Analysis")

        # Color coding function
        def color_z_score(val):
            """Color code z-scores."""
            if val < 1.0:
                return "background-color: #d4edda; color: #155724"  # Green
            elif val < 2.0:
                return "background-color: #fff3cd; color: #856404"  # Yellow
            else:
                return "background-color: #f8d7da; color: #721c24"  # Red

        # Display table with color coding
        st.markdown("### Feature Drift Metrics")

        styled_df = df.style.applymap(color_z_score, subset=["z_score"]).format(
            {"mean_train": "{:.4f}", "mean_live": "{:.4f}", "z_score": "{:.3f}"}
        )

        st.dataframe(styled_df, use_container_width=True, height=400)

        # Interpretation guide
        st.markdown(
            """
        **Z-Score Interpretation:**
        - 🟢 **< 1.0**: No significant drift
        - 🟡 **1.0 - 2.0**: Moderate drift - monitor
        - 🔴 **> 2.0**: Significant drift - investigate
        """
        )

        # Bar chart
        st.markdown("---")
        st.subheader("📊 Top 10 Features by Drift")

        top_features = df.head(10).copy()

        # Create bar chart
        st.bar_chart(
            top_features.set_index("feature_name")["z_score"], use_container_width=True, height=400
        )

        # Detailed metrics for top drifting features
        st.markdown("---")
        st.subheader("🔍 Top 5 Drifting Features (Detailed)")

        for idx, row in df.head(5).iterrows():
            with st.expander(f"{row['feature_name']} (z-score: {row['z_score']:.3f})"):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Training Mean", f"{row['mean_train']:.4f}")

                with col_b:
                    st.metric("Live Mean", f"{row['mean_live']:.4f}")

                with col_c:
                    diff_pct = (
                        ((row["mean_live"] - row["mean_train"]) / abs(row["mean_train"]) * 100)
                        if row["mean_train"] != 0
                        else 0
                    )
                    st.metric("Change", f"{diff_pct:+.2f}%")

        # Timestamp
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.error("Failed to fetch drift metrics. Please check the API connection.")

# Footer
st.markdown("---")
st.markdown(
    """
### About
This dashboard monitors feature drift by comparing live prediction data against training baseline statistics.
Drift is measured using z-scores, which indicate how many standard deviations the live mean differs from the training mean.
"""
)

# Auto-refresh implementation
if auto_refresh:
    import time

    time.sleep(refresh_interval)
    st.rerun()
