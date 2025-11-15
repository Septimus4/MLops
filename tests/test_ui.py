"""UI module tests for Gradio and Streamlit helpers.

These tests exercise pure functions and API interaction wrappers with mocking,
giving coverage without launching full servers.
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.ui import gradio_app
from src.ui import streamlit_drift


def test_plot_gauge_returns_image():
    img = gradio_app.plot_gauge(0.42, threshold=0.5, uncertain_width=0.1)
    assert hasattr(img, "save")  # PIL Image like


@patch("src.ui.gradio_app.requests.post")
def test_predict_risk_success(mock_post):
    mock_post.return_value = SimpleNamespace(
        status_code=200,
        json=lambda: {"risk_score": 0.73, "predicted_class": 1, "model_version": "vTest"},
        text="OK",
    )
    output_md, gauge_img = gradio_app.predict_risk(
        0.6, 0.5, 0.7, 600000.0, 30000.0, 200000.0, 550000.0, 38, 5, 3, 2, 0.02, 12, 10.0
    )
    assert "Prediction Result" in output_md
    assert "HIGH RISK" in output_md
    assert gauge_img is not None


@patch("src.ui.gradio_app.requests.post")
def test_predict_risk_api_error(mock_post):
    mock_post.return_value = SimpleNamespace(status_code=500, text="Server Error")
    output_md, gauge_img = gradio_app.predict_risk(
        0.5, 0.5, 0.5, 600000.0, 27000.0, 150000.0, 500000.0, 38, 5, 3, 2, 0.02, 12, 10.0
    )
    assert "Error" in output_md
    assert gauge_img is None


@patch("src.ui.streamlit_drift.requests.get")
def test_streamlit_get_drift_metrics_success(mock_get):
    mock_get.return_value = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "window_hours": 1,
            "metrics": [
                {
                    "feature_name": "EXT_SOURCE_1",
                    "mean_train": 0.5,
                    "mean_live": 0.6,
                    "z_score": 1.0,
                }
            ],
            "num_samples": 5,
        },
    )
    data = streamlit_drift.get_drift_metrics(1)
    assert data is not None
    assert data["metrics"][0]["feature_name"] == "EXT_SOURCE_1"


@patch("src.ui.streamlit_drift.requests.get")
def test_streamlit_get_drift_metrics_error(mock_get):
    mock_get.return_value = SimpleNamespace(status_code=404)
    data = streamlit_drift.get_drift_metrics(1)
    assert data is None


@patch("src.ui.streamlit_drift.requests.get")
def test_streamlit_get_health_success(mock_get):
    mock_get.return_value = SimpleNamespace(status_code=200, json=lambda: {"status": "ok"})
    health = streamlit_drift.get_health()
    assert health == {"status": "ok"}


@patch("src.ui.streamlit_drift.requests.get")
def test_streamlit_get_health_failure(mock_get):
    mock_get.return_value = SimpleNamespace(status_code=500)
    health = streamlit_drift.get_health()
    assert health is None
