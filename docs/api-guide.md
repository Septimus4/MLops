# API & UX Guide

## FastAPI Service

The scoring API is defined in `api/app.py` and exposes three primary endpoints:

| Method | Route      | Description                                           |
| ------ | ---------- | ----------------------------------------------------- |
| GET    | `/health`  | Returns model metadata, feature count, and threshold. |
| POST   | `/predict` | Scores a single applicant payload.                    |
| GET    | `/metrics` | Aggregates recent latency, error, and drift signals.  |

### Request Schema

`POST /predict` expects a JSON body shaped as:

```json
{
  "request_id": "optional-idempotency-token",
  "applicant_id": 12345,
  "features": {
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 428000,
    "AMT_ANNUITY": 17500,
    "DAYS_BIRTH": -15000,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "...": "..."
  }
}
```

- `request_id` is optional; if omitted, the API assigns a UUID.
- `features` must align with `artifacts/feature_list.json`; categorical values are validated with `categorical_mappings.json`.

### Response Schema

Successful response (`200 OK`):

```json
{
  "request_id": "same-as-input",
  "applicant_id": 12345,
  "model_name": "home_credit_model",
  "model_version": "v_20250922_192201",
  "score": 0.63,
  "binary_decision": 0,
  "threshold": 0.69,
  "inference_ms": 7,
  "processed_at": "2025-11-05T17:56:15.469061Z"
}
```

Validation errors return a structured payload:

```json
{
  "request_id": "generated-id",
  "message": "Unknown category 'Invalid' for feature 'CODE_GENDER'"
}
```

Unexpected failures mirror the same envelope with `message: "Unexpected inference failure"` and `500` status.

### Authentication & CORS

The reference implementation does not include authentication. CORS is open (`allow_origins=["*"]`) to simplify demos; harden this for production deployments.

## Gradio Manual Tester

`python -m api.gradio_ui` launches a Gradio Blocks interface bound to the same inference path as `/predict`. Use it to:

- Experiment with feature tweaks and observe score movements.
- Share a user-friendly interface with non-engineering stakeholders.
- Validate that categorical constraints behave as expected (errors bubble up if violated).

By default Gradio runs on `http://127.0.0.1:7860`. You can package it as a standalone container (for example, pushing a dedicated image to GHCR) if you prefer to isolate it from the API runtime.

## OpenAPI Specification

Generate the OpenAPI definition with:

```zsh
python -m scripts.generate_openapi --output docs/openapi/openapi.json
```

The resulting document is linked in the navigation sidebar and can feed tooling such as Stoplight, Swagger UI, or client SDK generators.

## Sample Workflow

1. Start the API: `uvicorn api.app:app --host 127.0.0.1 --port 8080`
2. Send a request: `python -m scripts.smoke_predict --url http://127.0.0.1:8080/predict`
3. Inspect metrics: `curl http://127.0.0.1:8080/metrics`
4. Explore interactively: run `python -m api.gradio_ui`

Every prediction appends a JSON record to `data/logs/YYYY-MM-DD.jsonl`, which powers the monitoring dashboards and drift analysis described in the next section.
