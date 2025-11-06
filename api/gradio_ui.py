"""Gradio-powered manual testing surface for the scoring API."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

try:  # pragma: no cover - convenience import for optional dependency
    gr = importlib.import_module("gradio")  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - surfaced at runtime when tool missing
    raise RuntimeError("Gradio must be installed to launch the manual tester") from exc

from .artifacts import get_categorical_mappings, get_feature_defaults
from .inference import FeatureValidationError, predict_one
from .model_loader import get_model
from .schemas import InputPayload

DEFAULTS = get_feature_defaults()
CATEGORICAL_MAPPINGS = get_categorical_mappings()
_, MODEL_METADATA = get_model()

TRAIN_DATA_PATH = Path("home-credit-default-risk-DATA/application_train.csv")
SAMPLE_COUNT = 75
UNCERTAINTY_BAND = 0.05
DEFAULT_SAMPLE_INFO = "Select a training applicant to prefill the form."


@dataclass(frozen=True)
class SampleApplicant:
    label: str
    income: float
    credit: float
    annuity: float
    goods_price: float
    children: int
    age_years: float
    years_employed: float
    years_registered: float
    contract_type: str
    gender: str
    own_car: str
    own_realty: str
    target: int

    @property
    def expected_decision(self) -> int:
        # TARGET == 0 -> good client -> approve
        return 1 if self.target == 0 else 0


def _safe_years_from_days(days: float | int | None, fallback: float) -> float:
    """Convert negative day offsets into positive years, guarding against sentinels."""

    if days is None or (isinstance(days, float) and math.isnan(days)):
        return fallback
    if abs(float(days)) >= 365243:  # DATA placeholder for missing employment
        return fallback
    return float(abs(float(days)) / 365.25)


def _load_training_samples(limit: int = SAMPLE_COUNT) -> Dict[str, SampleApplicant]:
    """Load a stratified subset of training applicants for manual validation."""

    if not TRAIN_DATA_PATH.exists():
        return {}

    usecols = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
    ]
    try:
        frame = pd.read_csv(TRAIN_DATA_PATH, usecols=usecols)
    except Exception:
        return {}

    frame = frame.dropna(subset=["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"])
    if frame.empty:
        return {}

    sample = (
        frame.sample(n=min(limit, len(frame)), random_state=42)
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )

    defaults = DEFAULTS
    applicants: Dict[str, SampleApplicant] = {}
    for _, row in sample.iterrows():
        goods_price = row.get("AMT_GOODS_PRICE")
        if pd.isna(goods_price):
            goods_price = float(defaults.get("AMT_GOODS_PRICE", 0.0))

        years_reg_default = float(defaults.get("YEARS_REGISTRATION", 5.0))
        years_emp_default = float(defaults.get("YEARS_EMPLOYED", 3.0))

        applicant = SampleApplicant(
            label=f"{int(row['SK_ID_CURR'])} (TARGET={int(row['TARGET'])})",
            income=float(row["AMT_INCOME_TOTAL"]),
            credit=float(row["AMT_CREDIT"]),
            annuity=float(row["AMT_ANNUITY"]),
            goods_price=float(goods_price),
            children=int(row.get("CNT_CHILDREN") or 0),
            age_years=_safe_years_from_days(
                row.get("DAYS_BIRTH"), float(defaults.get("AGE", 40.0))
            ),
            years_employed=_safe_years_from_days(
                row.get("DAYS_EMPLOYED"), years_emp_default
            ),
            years_registered=_safe_years_from_days(
                row.get("DAYS_REGISTRATION"), years_reg_default
            ),
            contract_type=str(row.get("NAME_CONTRACT_TYPE", "Cash loans")),
            gender=str(row.get("CODE_GENDER", "F")),
            own_car=str(row.get("FLAG_OWN_CAR", "N")),
            own_realty=str(row.get("FLAG_OWN_REALTY", "Y")),
            target=int(row.get("TARGET", 1)),
        )
        applicants[applicant.label] = applicant
    return applicants


@lru_cache(maxsize=1)
def _get_sample_applicants() -> Dict[str, SampleApplicant]:
    """Return cached sample applicants drawn from the training dataset."""

    return _load_training_samples()


def _days_from_years(years: float) -> int:
    """Convert a positive duration in years back to negative day offsets."""

    return -int(round(abs(years) * 365.25))


def _default_age_years() -> float:
    """Return the representative applicant age in years for UI defaults."""

    raw_age = float(DEFAULTS.get("AGE", 0.0))
    if raw_age > 0:
        return raw_age
    days = DEFAULTS.get("DAYS_BIRTH")
    if days:
        return float(abs(days) / 365.25)
    return 40.0


def _default_years_employed() -> float:
    """Return the representative employment history in years for UI defaults."""

    raw = float(DEFAULTS.get("YEARS_EMPLOYED", 0.0))
    if raw > 0:
        return raw
    days = DEFAULTS.get("DAYS_EMPLOYED")
    if days:
        return float(abs(days) / 365.25)
    return 5.0


def _default_years_registered() -> float:
    """Return the representative registration duration in years for UI defaults."""

    raw = float(DEFAULTS.get("YEARS_REGISTRATION", 0.0))
    if raw > 0:
        return raw
    days = DEFAULTS.get("DAYS_REGISTRATION")
    if days:
        return float(abs(days) / 365.25)
    return 5.0


def _build_payload(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
    children: int,
    age_years: float,
    years_employed: float,
    years_registered: float,
    contract_type: str,
    gender: str,
    own_car: str,
    own_realty: str,
) -> Dict[str, float | int | str]:
    days_birth = _days_from_years(age_years)
    days_employed = _days_from_years(years_employed)
    days_registration = _days_from_years(years_registered)

    features: Dict[str, float | int | str] = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "CNT_CHILDREN": children,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_REGISTRATION": days_registration,
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
    }
    return features


def _make_gauge(score: float, threshold: float) -> go.Figure:
    """Render a gauge visual showing probability vs approval threshold."""

    band = min(max(UNCERTAINTY_BAND, threshold * 0.1), 0.2)
    lower = max(0.0, threshold - band / 2)
    upper = min(1.0, threshold + band / 2)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": "Default Probability"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#2c3e50"},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [0, lower], "color": "#27ae60"},  # approve
                    {"range": [lower, upper], "color": "#f39c12"},  # uncertainty band
                    {"range": [upper, 1], "color": "#e74c3c"},  # decline
                ],
                "threshold": {
                    "line": {"color": "#34495e", "width": 4},
                    "value": threshold,
                },
            },
        )
    )
    figure.update_layout(margin={"l": 20, "r": 20, "t": 60, "b": 20}, height=320)
    return figure


def _format_result(score: float, threshold: float, decision_binary: int) -> str:
    """Summarise model outputs in Markdown."""

    decision = "Approve" if decision_binary else "Decline"
    margin = score - threshold
    descriptor = "below" if margin <= 0 else "above"
    return (
        f"**Decision:** {decision}\n\n"
        f"- Score: `{score:.4f}`\n"
        f"- Threshold: `{threshold:.2f}`\n"
        f"- Margin: `{abs(margin):.4f}` {descriptor} the threshold"
    )


def _format_expectation(decision_binary: int, expected_label: int | None) -> str:
    """Explain how the live decision compares to the historical training label."""

    if expected_label is None:
        return "Select a training applicant to compare the prediction with the original TARGET label."
    expected_decision = 1 if expected_label == 0 else 0
    expected_text = "Approve" if expected_decision else "Decline"
    matches = decision_binary == expected_decision
    status = "✅" if matches else "⚠️"
    explanation = (
        "matches the training label."
        if matches
        else "differs from the training label—investigate this applicant."
    )
    return (
        f"{status} Expected decision from training label (TARGET={expected_label}): **{expected_text}**.\n\n"
        f"The model's current decision {explanation}"
    )


def _run_prediction(
    payload_features: Dict[str, float | int | str], expected_label: int | None
) -> Tuple[Any, str, str, Any]:
    """Score the payload and return UI artefacts."""

    booster, metadata = get_model()
    payload = InputPayload(features=payload_features)

    try:
        processed, score, monitor = predict_one(payload, booster)
    except FeatureValidationError as exc:
        message = f"⚠️ Validation error: **{exc}**"
        return (
            None,
            message,
            "Unable to compare against training labels until the payload is corrected.",
            gr.update(value={"error": str(exc)}, visible=True),
        )

    threshold = metadata.threshold or 0.5
    decision_binary = int(score <= threshold)
    gauge = _make_gauge(score, threshold)
    result_text = _format_result(score, threshold, decision_binary)
    expectation_text = _format_expectation(decision_binary, expected_label)

    details_payload = {
        "score": score,
        "threshold": threshold,
        "binary_decision": decision_binary,
        "expected_decision": None
        if expected_label is None
        else (1 if expected_label == 0 else 0),
        "processed_features": processed,
        "monitor_features": monitor,
    }
    details = gr.update(value=details_payload, visible=True)
    return gauge, result_text, expectation_text, details


def _predict(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
    children: int,
    age_years: float,
    years_employed: float,
    years_registered: float,
    contract_type: str,
    gender: str,
    own_car: str,
    own_realty: str,
    expected_label: int | None,
) -> Tuple[Any, str, str, Any]:
    """Bridge Gradio inputs to the generic prediction helper."""

    payload = _build_payload(
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
    )
    return _run_prediction(payload, expected_label)


def _sample_choices() -> Tuple[List[str], List[str]]:
    """Return sorted applicant labels, split by TARGET."""

    applicants = _get_sample_applicants()
    positive = []
    negative = []
    for label, sample in applicants.items():
        if sample.target == 0:
            positive.append(label)
        else:
            negative.append(label)
    return sorted(positive), sorted(negative)


def _populate_from_sample(sample_label: str) -> Tuple[Any, ...]:
    """Produce component updates when a training applicant is selected."""

    applicants = _get_sample_applicants()
    sample = applicants.get(sample_label)
    if sample is None:
        empty_updates = tuple(gr.update() for _ in range(12))
        return (*empty_updates, DEFAULT_SAMPLE_INFO, None)

    info = (
        f"Loaded training applicant **{sample.label}**. "
        f"TARGET={sample.target} → expected decision: "
        f"{'Approve' if sample.expected_decision else 'Decline'}."
    )
    return (
        sample.income,
        sample.credit,
        sample.annuity,
        sample.goods_price,
        sample.children,
        round(sample.age_years, 1),
        round(sample.years_employed, 1),
        round(sample.years_registered, 1),
        sample.contract_type,
        sample.gender,
        sample.own_car,
        sample.own_realty,
        info,
        sample.target,
    )


def _auto_predict_from_sample(sample_label: str) -> Tuple[Any, str, str, Any]:
    """Immediately score the selected training applicant."""

    applicants = _get_sample_applicants()
    sample = applicants.get(sample_label)
    if sample is None:
        return (
            gr.update(value=None),
            "Awaiting input.",
            "Select an applicant to compare against training labels.",
            gr.update(value={}, visible=False),
        )

    payload = _build_payload(
        sample.income,
        sample.credit,
        sample.annuity,
        sample.goods_price,
        sample.children,
        sample.age_years,
        sample.years_employed,
        sample.years_registered,
        sample.contract_type,
        sample.gender,
        sample.own_car,
        sample.own_realty,
    )
    return _run_prediction(payload, sample.target)


def build_interface() -> Any:
    contract_options = sorted(CATEGORICAL_MAPPINGS["NAME_CONTRACT_TYPE"].keys())
    gender_options = sorted(CATEGORICAL_MAPPINGS["CODE_GENDER"].keys())
    own_car_options = sorted(CATEGORICAL_MAPPINGS["FLAG_OWN_CAR"].keys())
    own_realty_options = sorted(CATEGORICAL_MAPPINGS["FLAG_OWN_REALTY"].keys())

    with gr.Blocks(title="Home Credit Scoring – Manual Tester") as demo:
        gr.Markdown(
            """
            # Home Credit Scoring
            Adjust the inputs and run a manual inference using the production model.
            Select a training applicant to prefill values and verify the model stays aligned with historical labels.
            """
        )
        gr.Markdown(
            "Days-based model features are derived automatically: ages and durations are converted to the negative day counts expected by the model."
        )

        expected_state = gr.State(value=None)

        with gr.Row():
            good_choices, bad_choices = _sample_choices()
            positive_dropdown = gr.Dropdown(
                label="Prefill (TARGET = 0 → approved in training)",
                choices=good_choices,
                value=None,
                allow_custom_value=False,
                info=(
                    "Pick an applicant that stayed current during training."
                    if good_choices
                    else "Training dataset not available or no approved applicants sampled."
                ),
            )
            negative_dropdown = gr.Dropdown(
                label="Prefill (TARGET = 1 → defaulted in training)",
                choices=bad_choices,
                value=None,
                allow_custom_value=False,
                info=(
                    "Pick an applicant that defaulted during training."
                    if bad_choices
                    else "Training dataset not available or no defaulted applicants sampled."
                ),
            )

        sample_info = gr.Markdown(DEFAULT_SAMPLE_INFO)

        with gr.Row():
            with gr.Column():
                income = gr.Number(
                    label="Annual Income",
                    value=float(DEFAULTS.get("AMT_INCOME_TOTAL", 135000.0)),
                    precision=2,
                )
                credit = gr.Number(
                    label="Requested Credit",
                    value=float(DEFAULTS.get("AMT_CREDIT", 135000.0)),
                    precision=2,
                )
                annuity = gr.Number(
                    label="Annuity",
                    value=float(DEFAULTS.get("AMT_ANNUITY", 9000.0)),
                    precision=2,
                )
                goods_price = gr.Number(
                    label="Goods Price",
                    value=float(DEFAULTS.get("AMT_GOODS_PRICE", 135000.0)),
                    precision=2,
                )
                children = gr.Slider(
                    label="Children",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=int(DEFAULTS.get("CNT_CHILDREN", 0)),
                )
                age_years = gr.Number(
                    label="Age (years)", value=_default_age_years(), precision=1
                )
            with gr.Column():
                years_employed = gr.Number(
                    label="Years Employed",
                    value=_default_years_employed(),
                    precision=1,
                )
                years_registered = gr.Number(
                    label="Years With Current Documents",
                    value=_default_years_registered(),
                    precision=1,
                )
                contract_type = gr.Dropdown(
                    label="Contract Type",
                    choices=contract_options,
                    value=contract_options[0],
                )
                gender = gr.Dropdown(
                    label="Gender", choices=gender_options, value=gender_options[0]
                )
                own_car = gr.Dropdown(
                    label="Own Car", choices=own_car_options, value=own_car_options[0]
                )
                own_realty = gr.Dropdown(
                    label="Own Realty",
                    choices=own_realty_options,
                    value=own_realty_options[0],
                )

        run_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Row():
            gauge_plot = gr.Plot(label="Risk Gauge")
            with gr.Column():
                result_md = gr.Markdown(
                    "Run a prediction to view the decision details."
                )
                expectation_md = gr.Markdown("")
        details_json = gr.JSON(label="Prediction Details", value={}, visible=False)

        inputs: List[gr.components.Component] = [
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
            expected_state,
        ]

        outputs: List[gr.components.Component] = [
            gauge_plot,
            result_md,
            expectation_md,
            details_json,
        ]

        run_btn.click(_predict, inputs=inputs, outputs=outputs)

        prefill_outputs = [
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
            sample_info,
            expected_state,
        ]

        for dropdown in (positive_dropdown, negative_dropdown):
            dropdown.change(
                _populate_from_sample,
                inputs=dropdown,
                outputs=prefill_outputs,
            )
            dropdown.change(
                _auto_predict_from_sample,
                inputs=dropdown,
                outputs=outputs,
            )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
