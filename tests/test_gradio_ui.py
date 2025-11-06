from __future__ import annotations

import pytest
from api import gradio_ui

gradio = pytest.importorskip("gradio")  # noqa: F401


def test_make_gauge_returns_plotly_figure() -> None:
    figure = gradio_ui._make_gauge(score=0.42, threshold=0.65)
    assert figure.data
    indicator = figure.data[0]
    assert indicator["value"] == 0.42
    assert indicator["gauge"]["threshold"]["value"] == 0.65


def test_sample_population_round_trips_defaults() -> None:
    positives, negatives = gradio_ui._sample_choices()
    choices = positives or negatives
    if not choices:
        pytest.skip("Training dataset not available to build sample applicants")
    first = choices[0]
    (
        income,
        credit,
        annuity,
        goods_price,
        children,
        age_years,
        years_employed,
        years_registered,
        contract_type,
        gender,
        own_car,
        own_realty,
        info,
        expected,
    ) = gradio_ui._populate_from_sample(first)

    assert isinstance(income, (int, float))
    assert isinstance(info, str) and info
    assert expected in (0, 1)

    gauge, result, expectation, details_update = gradio_ui._auto_predict_from_sample(
        first
    )
    assert gauge is not None
    assert "Decision" in result
    assert "TARGET" in expectation
    details_candidate = getattr(details_update, "value", details_update)
    if isinstance(details_candidate, dict) and "__type__" in details_candidate:
        payload = details_candidate.get("value", {})
    else:
        payload = details_candidate
    assert "score" in payload
