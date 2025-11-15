"""
Pydantic schemas for API request/response models.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ApplicationFeatures(BaseModel):
    """Exhaustive list of accepted input features.

    All fields are optional; omitted values will be filled with server defaults.
    Extra/unknown keys are forbidden.
    """

    EXT_SOURCE_1: Optional[float] = Field(None, description="Normalized external source 1 (float)")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Normalized external source 2 (float)")
    EXT_SOURCE_3: Optional[float] = Field(None, description="Normalized external source 3 (float)")
    AMT_CREDIT: Optional[float] = Field(None, description="Credit amount of the loan")
    AMT_ANNUITY: Optional[float] = Field(None, description="Loan annuity amount")
    AMT_INCOME_TOTAL: Optional[float] = Field(None, description="Total income of client")
    AMT_GOODS_PRICE: Optional[float] = Field(None, description="Goods price for the loan")
    DAYS_BIRTH: Optional[int] = Field(None, description="Client age in days (negative from today)")
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Days employed (negative from today)")
    DAYS_REGISTRATION: Optional[int] = Field(None, description="Days since registration")
    DAYS_ID_PUBLISH: Optional[int] = Field(None, description="Days since ID publish")
    REGION_POPULATION_RELATIVE: Optional[float] = Field(
        None, description="Region population relative"
    )
    HOUR_APPR_PROCESS_START: Optional[int] = Field(
        None, description="Application process start hour"
    )
    OWN_CAR_AGE: Optional[float] = Field(None, description="Age of client's car")
    CNT_CHILDREN: Optional[int] = Field(None, description="Number of children")
    CNT_FAM_MEMBERS: Optional[float] = Field(None, description="Number of family members")
    REGION_RATING_CLIENT: Optional[int] = Field(None, description="Region rating client")
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(
        None, description="Region rating client with city"
    )
    WEEKDAY_APPR_PROCESS_START: Optional[int] = Field(
        None, description="Weekday of application process start"
    )
    REG_REGION_NOT_LIVE_REGION: Optional[int] = Field(
        None, description="Registration region not live region flag"
    )
    REG_REGION_NOT_WORK_REGION: Optional[int] = Field(
        None, description="Registration region not work region flag"
    )
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = Field(
        None, description="Live region not work region flag"
    )
    REG_CITY_NOT_LIVE_CITY: Optional[int] = Field(
        None, description="Registration city not live city flag"
    )
    REG_CITY_NOT_WORK_CITY: Optional[int] = Field(
        None, description="Registration city not work city flag"
    )
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = Field(None, description="Live city not work city flag")
    FLAG_MOBIL: Optional[int] = Field(None, description="Has mobile flag")
    FLAG_EMP_PHONE: Optional[int] = Field(None, description="Has employer phone flag")
    FLAG_WORK_PHONE: Optional[int] = Field(None, description="Has work phone flag")
    FLAG_CONT_MOBILE: Optional[int] = Field(None, description="Continuous mobile flag")
    FLAG_PHONE: Optional[int] = Field(None, description="Has phone flag")
    FLAG_EMAIL: Optional[int] = Field(None, description="Has email flag")

    class Config:
        extra = "forbid"


class PredictionRequest(BaseModel):
    """Wrapped request body for prediction endpoint."""

    features: ApplicationFeatures = Field(
        ...,
        description="Input feature set (all optional; server fills defaults)",
        json_schema_extra={
            "example": {
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "AMT_CREDIT": 600000.0,
                "AMT_ANNUITY": 27000.0,
                "DAYS_BIRTH": -14000,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    risk_score: float = Field(..., description="Probability of default (0-1)", ge=0.0, le=1.0)
    predicted_class: int = Field(
        ..., description="Predicted class (0=no default, 1=default)", ge=0, le=1
    )
    model_version: str = Field(..., description="Version of the model used")
    feature_values: Dict[str, float] = Field(
        ..., description="Processed feature values used for prediction"
    )


class DriftMetric(BaseModel):
    """Drift metric for a single feature."""

    feature_name: str = Field(..., description="Name of the feature")
    mean_train: float = Field(..., description="Mean value in training data")
    mean_live: float = Field(..., description="Mean value in live data")
    z_score: float = Field(..., description="Z-score indicating drift magnitude")


class DriftResponse(BaseModel):
    """Response schema for drift endpoint."""

    window_hours: int = Field(..., description="Time window for drift calculation in hours")
    metrics: List[DriftMetric] = Field(..., description="Drift metrics for each feature")
    num_samples: int = Field(..., description="Number of samples in the window")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Version of loaded model")
