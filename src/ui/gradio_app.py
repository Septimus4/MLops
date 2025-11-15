import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from matplotlib.patches import Wedge, Circle

import os

import gradio as gr
import requests


def plot_gauge(risk_score: float, threshold: float = 0.5, uncertain_width: float = 0.2):
    """Render a semicircular gauge.
    Green: 0 -> (threshold - uncertain_width/2)
    Orange (uncertain): (threshold - uncertain_width/2) -> (threshold + uncertain_width/2)
    Red: (threshold + uncertain_width/2) -> 1
    """
    # Clamp values
    risk_score = max(0.0, min(1.0, risk_score))
    threshold = max(0.05, min(0.95, threshold))
    uncertain_width = max(0.05, min(0.4, uncertain_width))

    low_end = 0.0
    uncertain_start = max(low_end, threshold - uncertain_width / 2)
    uncertain_end = min(1.0, threshold + uncertain_width / 2)
    high_end = 1.0

    def val_to_angle(val: float) -> float:
        # Map 0 -> 180 degrees, 1 -> 0 degrees
        return 180 - (val * 180)

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"aspect": "equal"})
    ax.axis("off")
    fig.patch.set_facecolor("white")

    radius = 1.0
    # Zones as wedges
    wedges = [
        (low_end, uncertain_start, "#2ecc71"),  # Green
        (uncertain_start, uncertain_end, "#f39c12"),  # Orange
        (uncertain_end, high_end, "#e74c3c"),  # Red
    ]
    for start, end, color in wedges:
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius,
                theta1=val_to_angle(end),
                theta2=val_to_angle(start),
                facecolor=color,
                edgecolor="white",
                lw=2,
                alpha=0.85,
            )
        )

    # Inner white circle to create gauge arc effect
    ax.add_patch(Circle((0, 0), radius * 0.65, facecolor="white", edgecolor="white"))

    # Ticks
    for t in np.linspace(0, 1, 11):
        ang = np.deg2rad(val_to_angle(t))
        x_outer = np.cos(ang) * radius
        y_outer = np.sin(ang) * radius
        x_inner = np.cos(ang) * radius * 0.72
        y_inner = np.sin(ang) * radius * 0.72
        ax.plot([x_inner, x_outer], [y_inner, y_outer], color="#333", lw=2)
        if t in {0, 0.25, 0.5, 0.75, 1.0}:
            ax.text(
                np.cos(ang) * radius * 0.5,
                np.sin(ang) * radius * 0.5,
                f"{int(t*100)}%",
                ha="center",
                va="center",
                fontsize=10,
                color="#333",
            )

    # Needle
    needle_ang = np.deg2rad(val_to_angle(risk_score))
    nx = np.cos(needle_ang) * radius * 0.9
    ny = np.sin(needle_ang) * radius * 0.9
    ax.plot([0, nx], [0, ny], color="#222", lw=4, solid_capstyle="round")
    ax.add_patch(Circle((0, 0), 0.06, facecolor="#222", edgecolor="#222"))

    # Status label
    if risk_score < uncertain_start:
        zone_label = "LOW"
        zone_color = "#2ecc71"
    elif risk_score <= uncertain_end:
        zone_label = "UNCERTAIN"
        zone_color = "#f39c12"
    else:
        zone_label = "HIGH"
        zone_color = "#e74c3c"

    ax.text(
        0,
        -0.25,
        f"Risk {risk_score*100:.1f}%",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=zone_color,
    )
    ax.text(0, -0.45, f"{zone_label} RISK", ha="center", va="center", fontsize=12, color=zone_color)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


"""
Gradio UI for Home Credit Risk prediction.
"""

# API URL (configurable via environment variable)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def predict_risk(
    ext_source_1: float,
    ext_source_2: float,
    ext_source_3: float,
    amt_credit: float,
    amt_annuity: float,
    amt_income_total: float,
    amt_goods_price: float,
    age_years: float,
    employed_years: float,
    registration_years: float,
    id_publish_years: float,
    region_population_relative: float,
    hour_appr_process_start: int,
    own_car_age: float,
):
    """
    Make a prediction using the API.

    Returns formatted prediction result.
    """
    # Build features dictionary
    # Transform friendly year inputs to negative day counts (dataset convention)
    features = {
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_GOODS_PRICE": amt_goods_price,
        "DAYS_BIRTH": -int(age_years * 365),
        "DAYS_EMPLOYED": -int(employed_years * 365),
        "DAYS_REGISTRATION": -int(registration_years * 365),
        "DAYS_ID_PUBLISH": -int(id_publish_years * 365),
        "REGION_POPULATION_RELATIVE": region_population_relative,
        "HOUR_APPR_PROCESS_START": hour_appr_process_start,
        "OWN_CAR_AGE": own_car_age,
    }

    try:
        # Call API
        response = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)

        if response.status_code == 200:
            result = response.json()
            risk_score = result["risk_score"]
            predicted_class = result["predicted_class"]
            model_version = result["model_version"]

            # Format output
            risk_pct = risk_score * 100
            status = "HIGH RISK" if predicted_class == 1 else "LOW RISK"

            output = f"""## Prediction Result

**Status:** {status}

**Default Risk Score:** {risk_pct:.2f}%

**Predicted Class:** {'Default (1)' if predicted_class == 1 else 'No Default (0)'}

**Model Version:** {model_version}

---
### Input Summary (Friendly)
* Age: {age_years:.1f} years
* Employed: {employed_years:.1f} years
* Registration: {registration_years:.1f} years ago
* ID Issued: {id_publish_years:.1f} years ago
* External Scores: [{ext_source_1:.2f}, {ext_source_2:.2f}, {ext_source_3:.2f}]

### Risk Interpretation
* **0-30%**: Low risk - likely to repay
* **30-50%**: Medium risk - monitor closely
* **50-70%**: High risk - requires careful evaluation
* **70-100%**: Very high risk - likely to default
"""
            gauge_img = plot_gauge(risk_score, threshold=0.5, uncertain_width=0.1)
            return output, gauge_img
        else:
            return f"❌ Error: {response.status_code} - {response.text}", None

    except requests.exceptions.ConnectionError:
        return (
            f"❌ Error: Cannot connect to API at {API_URL}. Please ensure the API service is running.",
            None,
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# Create Gradio interface
def load_profile(profile: str):
    """Return preset friendly inputs for selected profile."""
    presets = {
        # ext1, ext2, ext3, credit, annuity, income, goods_price, age, employed, registration, id_publish, region_pop, hour, car_age
        "Low Risk": [0.85, 0.80, 0.83, 200000, 10000, 250000, 180000, 45, 10, 8, 5, 0.02, 12, 2],
        # Tuned to yield ~0.50 risk score
        "Uncertain (~50%)": [
            0.13,
            0.14,
            0.14,
            800000,
            50000,
            50000,
            750000,
            20,
            0.0,
            0.05,
            0.02,
            0.095,
            3,
            18,
        ],
        "High Risk": [
            0.03,
            0.04,
            0.05,
            800000,
            50000,
            40000,
            750000,
            20,
            0.0,
            0.05,
            0.02,
            0.095,
            3,
            18,
        ],
    }
    return presets.get(profile, presets["Low Risk"])  # default


with gr.Blocks(title="Home Credit Risk Predictor") as demo:
    gr.Markdown(
        """
    # 🏦 Home Credit Default Risk Predictor
    
    Enter loan application details to predict default risk.
    
    **Note:** DAYS_* features are negative (e.g., -14000 days = ~38 years old)
    """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### External Scores")
            ext_source_1 = gr.Slider(
                0,
                1,
                value=0.5,
                label="EXT_SOURCE_1",
                info="External risk score 1 (0-1, higher is better)",
            )
            ext_source_2 = gr.Slider(
                0,
                1,
                value=0.5,
                label="EXT_SOURCE_2",
                info="External risk score 2 (0-1, higher is better)",
            )
            ext_source_3 = gr.Slider(
                0,
                1,
                value=0.5,
                label="EXT_SOURCE_3",
                info="External risk score 3 (0-1, higher is better)",
            )

            gr.Markdown("### Loan Details")
            amt_credit = gr.Number(value=600000, label="Loan Amount (AMT_CREDIT)")
            amt_annuity = gr.Number(value=27000, label="Annuity (AMT_ANNUITY)")
            amt_goods_price = gr.Number(value=500000, label="Goods Price (AMT_GOODS_PRICE)")

        with gr.Column():
            gr.Markdown("### Applicant & Timeline (Years)")
            amt_income_total = gr.Number(value=150000, label="Annual Income (AMT_INCOME_TOTAL)")
            age_years = gr.Number(value=38, label="Age (years)")
            employed_years = gr.Number(value=5, label="Years Employed")
            registration_years = gr.Number(value=3, label="Years Since Registration")
            id_publish_years = gr.Number(value=2, label="Years Since ID Issued")

            gr.Markdown("### Other Features")
            region_population_relative = gr.Slider(
                0, 0.1, value=0.02, label="Region Population Relative"
            )
            hour_appr_process_start = gr.Slider(
                0, 23, value=12, step=1, label="Application Hour (0-23)"
            )
            own_car_age = gr.Number(value=10, label="Car Age (years)")

    gr.Markdown("### Demo Profiles")
    profile_choice = gr.Radio(["Low Risk", "Uncertain (~50%)", "High Risk"], label="Select Profile")
    load_profile_btn = gr.Button("Load Profile Values")
    load_profile_btn.click(
        fn=load_profile,
        inputs=profile_choice,
        outputs=[
            ext_source_1,
            ext_source_2,
            ext_source_3,
            amt_credit,
            amt_annuity,
            amt_income_total,
            amt_goods_price,
            age_years,
            employed_years,
            registration_years,
            id_publish_years,
            region_population_relative,
            hour_appr_process_start,
            own_car_age,
        ],
    )

    predict_btn = gr.Button("🔮 Predict Default Risk", variant="primary", size="lg")

    output = gr.Markdown()
    gauge = gr.Image(label="Risk Gauge", type="pil")

    predict_btn.click(
        fn=predict_risk,
        inputs=[
            ext_source_1,
            ext_source_2,
            ext_source_3,
            amt_credit,
            amt_annuity,
            amt_income_total,
            amt_goods_price,
            age_years,
            employed_years,
            registration_years,
            id_publish_years,
            region_population_relative,
            hour_appr_process_start,
            own_car_age,
        ],
        outputs=[output, gauge],
    )

    gr.Markdown(
        """
    ---
    ### About
    This interface uses a LightGBM model trained on the Home Credit Default Risk dataset.
    All predictions are logged for drift monitoring.
    """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
