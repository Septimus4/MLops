"""
FastAPI application for Home Credit Risk prediction service.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from . import db, drift, model_loader
from .feature_processor import build_feature_vector, validate_features
from .schemas import (
    DriftMetric,
    DriftResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting Home Credit Risk API...")

    # Initialize database
    try:
        db.init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")

    # Verify model is loaded
    if model_loader.MODEL is None:
        print("✗ Model not loaded - attempting to load...")
        try:
            model_loader.load_model()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
    else:
        print(f"✓ Model loaded (version: {model_loader.MODEL_VERSION})")

    # Verify baseline stats are loaded
    if drift.BASELINE_STATS is None:
        print("✗ Baseline stats not loaded - attempting to load...")
        try:
            drift.load_baseline_stats()
            print("✓ Baseline stats loaded successfully")
        except Exception as e:
            print(f"✗ Baseline stats loading failed: {e}")
    else:
        print("✓ Baseline stats loaded")

    print("Home Credit Risk API ready!")

    yield

    # Shutdown (if needed)
    print("Shutting down Home Credit Risk API...")


# Create FastAPI app
app = FastAPI(
    title="Home Credit Risk API",
    version="1.0.0",
    description="API for predicting loan default risk and monitoring model drift",
    lifespan=lifespan,
)


# Custom OpenAPI to enumerate allowed feature keys instead of generic object
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    try:
        # Locate PredictionRequest schema and refine features property
        components = openapi_schema.get("components", {}).get("schemas", {})
        if "PredictionRequest" in components:
            pr_schema = components["PredictionRequest"]
            features_prop = pr_schema.get("properties", {}).get("features")
            if features_prop:
                # Determine feature list: prefer model feature names if loaded
                from . import model_loader  # local import to avoid circular timing issues

                try:
                    feature_names = (
                        list(model_loader.FEATURE_NAMES) if model_loader.FEATURE_NAMES else []
                    )
                except Exception:
                    feature_names = []
                if not feature_names:
                    # Fallback: use configured dtypes keys
                    from training.feature_config import FEATURE_DTYPES

                    feature_names = list(FEATURE_DTYPES.keys())

                # Build explicit properties for each feature
                feature_properties = {}
                # Import dtypes for accurate OpenAPI typing
                try:
                    from training.feature_config import FEATURE_DTYPES as _DTYPES
                except Exception:
                    _DTYPES = {}
                for name in feature_names:
                    dtype = _DTYPES.get(name, "float")
                    json_type = "integer" if dtype == "int" else "number"
                    example = 0 if json_type == "integer" else 0.0
                    feature_properties[name] = {
                        "type": json_type,
                        "description": (
                            f"Input feature '{name}' ({json_type}). Optional; server fills default if omitted."
                        ),
                        "example": example,
                    }
                features_prop["type"] = "object"
                features_prop["properties"] = feature_properties
                features_prop["additionalProperties"] = False
                features_prop[
                    "description"
                ] = "Feature values for prediction. Only listed keys are accepted; unspecified features will use defaults."
        app.openapi_schema = openapi_schema
    except Exception as e:
        # In case of failure, keep original schema
        print(f"Warning: custom OpenAPI generation failed: {e}")
        app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the service and model.
    """
    model_loaded = model_loader.MODEL is not None
    model_version = model_loader.MODEL_VERSION if model_loaded else "unknown"

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for loan default risk.

    Args:
        request: PredictionRequest with feature values

    Returns:
        PredictionResponse with risk score and prediction
    """
    # Check if model is loaded
    if model_loader.MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and the service is properly initialized.",
        )

    try:
        # Convert Pydantic model to dict including None values (Pydantic v2: use model_dump)
        raw_feature_input = request.features.model_dump(exclude_none=False)

        # Determine which fields user actually provided (so we don't reject missing optional ones)
        provided_fields = getattr(
            request.features, "model_fields_set", set(raw_feature_input.keys())
        )
        none_keys = [k for k in provided_fields if raw_feature_input.get(k) is None]
        if none_keys:
            raise HTTPException(status_code=400, detail=f"Feature(s) {none_keys} cannot be None")

        # Filter out None values (only truly provided Nones would have triggered above)
        feature_input = {k: v for k, v in raw_feature_input.items() if v is not None}

        # Validate provided features (after removing None)
        if feature_input:
            validate_features(feature_input)

        # Build feature vector
        feature_vector, filled_features = build_feature_vector(
            feature_input, model_loader.FEATURE_NAMES
        )

        # Make prediction
        risk_score = model_loader.predict_proba_row(feature_vector)

        # Apply threshold for classification
        predicted_class = int(risk_score >= 0.5)

        # Log prediction to database
        try:
            db.log_prediction(
                model_version=model_loader.MODEL_VERSION,
                features_dict=filled_features,
                risk_score=risk_score,
                predicted_class=predicted_class,
            )
        except Exception as e:
            print(f"Warning: Failed to log prediction: {e}")

        return PredictionResponse(
            risk_score=risk_score,
            predicted_class=predicted_class,
            model_version=model_loader.MODEL_VERSION,
            feature_values=filled_features,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        # Propagate already formed HTTP errors (e.g., 400 for None features)
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
async def get_drift_metrics(window_hours: int = 24):
    """
    Get feature drift metrics for the specified time window.

    Args:
        window_hours: Time window in hours (default: 24)

    Returns:
        DriftResponse with drift metrics for each feature
    """
    if window_hours < 1:
        raise HTTPException(status_code=400, detail="window_hours must be at least 1")

    if window_hours > 168:  # 1 week
        raise HTTPException(status_code=400, detail="window_hours cannot exceed 168 (1 week)")

    try:
        # Compute drift metrics
        metrics_list = drift.compute_drift_metrics(window_hours=window_hours)

        # Get number of samples
        num_samples = drift.get_num_samples(window_hours)

        # Convert to DriftMetric objects
        metrics = [DriftMetric(**metric) for metric in metrics_list]

        return DriftResponse(
            window_hours=window_hours,
            metrics=metrics,
            num_samples=num_samples,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift computation failed: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Home Credit Risk API",
        "version": "1.0.0",
        "description": "API for predicting loan default risk and monitoring model drift",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "drift": "/drift",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }
