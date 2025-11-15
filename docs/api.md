# API Documentation

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Endpoints

### GET /health

Health check endpoint to verify service status.

**Response**: `HealthResponse`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0.0"
}
```

**Status Codes**:
- `200 OK` - Service is healthy

**Example**:

```bash
curl http://localhost:8000/health
```

### POST /predict

Make a loan default risk prediction.

**Request Body**: `PredictionRequest`

```json
{
  "features": {
    "EXT_SOURCE_1": 0.6,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.7,
    "AMT_CREDIT": 600000.0,
    "AMT_ANNUITY": 30000.0,
    "AMT_INCOME_TOTAL": 200000.0,
    "AMT_GOODS_PRICE": 550000.0,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -2500,
    "DAYS_REGISTRATION": -5000,
    "DAYS_ID_PUBLISH": -3500,
    "REGION_POPULATION_RELATIVE": 0.025,
    "HOUR_APPR_PROCESS_START": 14,
    "OWN_CAR_AGE": 8.0
  }
}
```

**Note**: Missing features will be filled with default values.

**Response**: `PredictionResponse`

```json
{
  "risk_score": 0.3456,
  "predicted_class": 0,
  "model_version": "v1.0.0",
  "feature_values": {
    "EXT_SOURCE_1": 0.6,
    "EXT_SOURCE_2": 0.5,
    "...": "..."
  }
}
```

**Response Fields**:
- `risk_score` (float): Probability of default (0-1)
- `predicted_class` (int): 0=no default, 1=default (threshold 0.5)
- `model_version` (string): Model version used
- `feature_values` (object): Processed feature values (including defaults)

**Status Codes**:
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input features
- `503 Service Unavailable` - Model not loaded

**Example**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "EXT_SOURCE_1": 0.6,
      "AMT_CREDIT": 600000.0,
      "DAYS_BIRTH": -15000
    }
  }'
```

**Python Example**:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
            "EXT_SOURCE_1": 0.6,
            "EXT_SOURCE_2": 0.5,
            "AMT_CREDIT": 600000.0
        }
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Predicted Class: {result['predicted_class']}")
else:
    print(f"Error: {response.status_code}")
```

### GET /drift

Get feature drift metrics for a specified time window.

**Query Parameters**:
- `window_hours` (int, optional): Time window in hours (default: 24, min: 1, max: 168)

**Response**: `DriftResponse`

```json
{
  "window_hours": 24,
  "num_samples": 150,
  "metrics": [
    {
      "feature_name": "EXT_SOURCE_1",
      "mean_train": 0.5123,
      "mean_live": 0.5876,
      "z_score": 2.145
    },
    {
      "feature_name": "AMT_CREDIT",
      "mean_train": 599025.45,
      "mean_live": 612340.22,
      "z_score": 0.891
    }
  ]
}
```

**Response Fields**:
- `window_hours` (int): Time window used
- `num_samples` (int): Number of predictions in window
- `metrics` (array): Drift metrics for each feature, sorted by z_score descending
  - `feature_name` (string): Feature name
  - `mean_train` (float): Mean value in training data
  - `mean_live` (float): Mean value in live predictions
  - `z_score` (float): Drift magnitude (|mean_live - mean_train| / std_train)

**Z-Score Interpretation**:
- < 1.0: No significant drift
- 1.0 - 2.0: Moderate drift, monitor
- \> 2.0: Significant drift, investigate

**Status Codes**:
- `200 OK` - Drift metrics computed
- `400 Bad Request` - Invalid window_hours
- `500 Internal Server Error` - Computation failed

**Example**:

```bash
curl "http://localhost:8000/drift?window_hours=48"
```

**Python Example**:

```python
import requests

response = requests.get(
    "http://localhost:8000/drift",
    params={"window_hours": 24}
)

if response.status_code == 200:
    result = response.json()
    print(f"Analyzed {result['num_samples']} predictions")
    
    # Show top drifting features
    for metric in result['metrics'][:5]:
        print(f"{metric['feature_name']}: z={metric['z_score']:.2f}")
```

### GET /

Root endpoint providing API information.

**Response**:

```json
{
  "name": "Home Credit Risk API",
  "version": "1.0.0",
  "description": "API for predicting loan default risk and monitoring model drift",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "drift": "/drift",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

## Data Models

### PredictionRequest

```python
{
  "features": {
    str: float | int | str
  }
}
```

Features can be:
- Fully specified (all model features)
- Partially specified (missing features use defaults)
- Empty (all features use defaults)

### PredictionResponse

```python
{
  "risk_score": float,        # 0.0 to 1.0
  "predicted_class": int,     # 0 or 1
  "model_version": str,
  "feature_values": {
    str: float
  }
}
```

### DriftMetric

```python
{
  "feature_name": str,
  "mean_train": float,
  "mean_live": float,
  "z_score": float
}
```

### DriftResponse

```python
{
  "window_hours": int,
  "num_samples": int,
  "metrics": [DriftMetric]
}
```

### HealthResponse

```python
{
  "status": str,              # "ok" or "degraded"
  "model_loaded": bool,
  "model_version": str
}
```

## Error Responses

All errors return a standard format:

```json
{
  "detail": "Error message description"
}
```

**Common Error Codes**:
- `400 Bad Request` - Invalid input
- `404 Not Found` - Endpoint doesn't exist
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service not ready

## Rate Limiting

Currently no rate limiting is implemented. For production:
- Implement rate limiting middleware
- Consider per-client API keys
- Use reverse proxy (Nginx, Traefik) for rate limiting

## Authentication

Currently no authentication is required. For production:
- Implement OAuth2 with JWT tokens
- Use API keys for service-to-service
- Configure CORS appropriately

## CORS

CORS is currently configured to allow all origins. For production:
- Restrict to specific domains
- Configure allowed methods and headers
- Enable credentials if needed

## Versioning

API version is included in responses but URL versioning is not yet implemented. For future versions, consider:
- URL versioning: `/v1/predict`, `/v2/predict`
- Header versioning: `Accept: application/vnd.api.v1+json`
- Query parameter versioning: `/predict?version=v1`

## Performance

Typical response times (single API instance):
- `/health`: < 10ms
- `/predict`: 50-100ms
- `/drift`: 100-500ms (depends on window size and data volume)

For better performance:
- Use async clients
- Batch predictions if possible
- Cache drift metrics
- Scale horizontally

## SDK / Client Libraries

No official SDKs are currently provided. Example client code is available in:
- `scripts/demo_requests.py` - Python client
- Documentation examples above

Consider creating SDKs for:
- Python (pip package)
- JavaScript/TypeScript (npm package)
- R (CRAN package)
