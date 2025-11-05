"""Gradio-powered manual testing surface for the scoring API."""

from __future__ import annotations

from typing import Any, Dict

import importlib

try:  # pragma: no cover - convenience import for optional dependency
    gr = importlib.import_module("gradio")  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - surfaced at runtime when tool missing
    raise RuntimeError("Gradio must be installed to launch the manual tester") from exc

from .artifacts import get_categorical_mappings, get_feature_defaults
from .inference import FeatureValidationError, predict_one
from .model_loader import get_model
from .schemas import InputPayload

# Load shared resources once at module import.
DEFAULTS = get_feature_defaults()
CATEGORICAL_MAPPINGS = get_categorical_mappings()
_, MODEL_METADATA = get_model()


def _build_payload(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
    children: int,
    days_birth: int,
    days_employed: int,
    days_registration: int,
    contract_type: str,
    gender: str,
    own_car: str,
    own_realty: str,
) -> Dict[str, float | int | str]:
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


def _predict(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
    children: int,
    days_birth: int,
    days_employed: int,
    days_registration: int,
    contract_type: str,
    gender: str,
    own_car: str,
    own_realty: str,
) -> Dict[str, str]:
    booster, metadata = get_model()
    payload = InputPayload(features=_build_payload(
        income,
        credit,
        annuity,
        goods_price,
        children,
        days_birth,
        days_employed,
        days_registration,
        contract_type,
        gender,
        own_car,
        own_realty,
    ))

    try:
        _, score, _ = predict_one(payload, booster)
    except FeatureValidationError as exc:
        return {"Error": str(exc)}

    threshold = metadata.threshold or 0.5
    decision = "Approve" if score >= threshold else "Decline"
    return {
        "Score": f"{score:.4f}",
        "Decision": decision,
        "Threshold": f"{threshold:.2f}",
    }


def build_interface() -> Any:
    contract_options = sorted(CATEGORICAL_MAPPINGS["NAME_CONTRACT_TYPE"].keys())
    gender_options = sorted(CATEGORICAL_MAPPINGS["CODE_GENDER"].keys())
    own_car_options = sorted(CATEGORICAL_MAPPINGS["FLAG_OWN_CAR"].keys())
    own_realty_options = sorted(CATEGORICAL_MAPPINGS["FLAG_OWN_REALTY"].keys())

    with gr.Blocks(title="Home Credit Scoring – Manual Tester") as demo:
        gr.Markdown("""
        # Home Credit Scoring
        Adjust the inputs and run a manual inference using the production model. Defaults come from the training median.
        """)

        with gr.Row():
            with gr.Column():
                income = gr.Number(label="Annual Income", value=float(DEFAULTS.get("AMT_INCOME_TOTAL", 135000.0)))
                credit = gr.Number(label="Requested Credit", value=float(DEFAULTS.get("AMT_CREDIT", 135000.0)))
                annuity = gr.Number(label="Annuity", value=float(DEFAULTS.get("AMT_ANNUITY", 9000.0)))
                goods_price = gr.Number(label="Goods Price", value=float(DEFAULTS.get("AMT_GOODS_PRICE", 135000.0)))
                children = gr.Slider(label="Children", minimum=0, maximum=10, step=1, value=int(DEFAULTS.get("CNT_CHILDREN", 0)))
                days_birth = gr.Number(label="Days Since Birth (negative)", value=int(DEFAULTS.get("DAYS_BIRTH", -12000)))
                days_employed = gr.Number(label="Days Employed (negative)", value=int(DEFAULTS.get("DAYS_EMPLOYED", -2000)))
                days_registration = gr.Number(label="Days Since Registration (negative)", value=int(DEFAULTS.get("DAYS_REGISTRATION", -1500)))
            with gr.Column():
                contract_type = gr.Dropdown(label="Contract Type", choices=contract_options, value=contract_options[0])
                gender = gr.Dropdown(label="Gender", choices=gender_options, value=gender_options[0])
                own_car = gr.Dropdown(label="Own Car", choices=own_car_options, value=own_car_options[0])
                own_realty = gr.Dropdown(label="Own Realty", choices=own_realty_options, value=own_realty_options[0])

        run_btn = gr.Button("Run Prediction")
        output = gr.JSON(label="Prediction Result")

        run_btn.click(
            _predict,
            inputs=[
                income,
                credit,
                annuity,
                goods_price,
                children,
                days_birth,
                days_employed,
                days_registration,
                contract_type,
                gender,
                own_car,
                own_realty,
            ],
            outputs=[output],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
