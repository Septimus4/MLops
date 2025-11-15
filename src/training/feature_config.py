"""
Feature configuration for Home Credit Default Risk model.
Defines feature names, types, defaults, and exposed features for UI.
"""

# Column identifiers
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Exposed features for UI (subset of most important features)
EXPOSED_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "REGION_POPULATION_RELATIVE",
    "HOUR_APPR_PROCESS_START",
    "OWN_CAR_AGE",
]

# Feature data types
FEATURE_DTYPES = {
    "EXT_SOURCE_1": "float",
    "EXT_SOURCE_2": "float",
    "EXT_SOURCE_3": "float",
    "AMT_CREDIT": "float",
    "AMT_ANNUITY": "float",
    "AMT_INCOME_TOTAL": "float",
    "AMT_GOODS_PRICE": "float",
    "DAYS_BIRTH": "int",
    "DAYS_EMPLOYED": "int",
    "DAYS_REGISTRATION": "int",
    "DAYS_ID_PUBLISH": "int",
    "REGION_POPULATION_RELATIVE": "float",
    "HOUR_APPR_PROCESS_START": "int",
    "OWN_CAR_AGE": "float",
    "CNT_CHILDREN": "int",
    "CNT_FAM_MEMBERS": "float",
    "REGION_RATING_CLIENT": "int",
    "REGION_RATING_CLIENT_W_CITY": "int",
    "WEEKDAY_APPR_PROCESS_START": "int",
    "REG_REGION_NOT_LIVE_REGION": "int",
    "REG_REGION_NOT_WORK_REGION": "int",
    "LIVE_REGION_NOT_WORK_REGION": "int",
    "REG_CITY_NOT_LIVE_CITY": "int",
    "REG_CITY_NOT_WORK_CITY": "int",
    "LIVE_CITY_NOT_WORK_CITY": "int",
    "FLAG_MOBIL": "int",
    "FLAG_EMP_PHONE": "int",
    "FLAG_WORK_PHONE": "int",
    "FLAG_CONT_MOBILE": "int",
    "FLAG_PHONE": "int",
    "FLAG_EMAIL": "int",
}

# Default values for features (will be computed from training data)
# These are placeholders and should be updated after computing actual statistics
FEATURE_DEFAULTS = {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "AMT_CREDIT": 600000.0,
    "AMT_ANNUITY": 27000.0,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_GOODS_PRICE": 500000.0,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2000,
    "DAYS_REGISTRATION": -4000,
    "DAYS_ID_PUBLISH": -3000,
    "REGION_POPULATION_RELATIVE": 0.02,
    "HOUR_APPR_PROCESS_START": 12,
    "OWN_CAR_AGE": 10.0,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 2.0,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "WEEKDAY_APPR_PROCESS_START": 2,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "FLAG_MOBIL": 1,
    "FLAG_EMP_PHONE": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_CONT_MOBILE": 1,
    "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0,
}


def get_feature_type(feature_name: str) -> str:
    """Get the data type for a feature."""
    return FEATURE_DTYPES.get(feature_name, "float")


def get_feature_default(feature_name: str) -> float | int:
    """Get the default value for a feature."""
    return FEATURE_DEFAULTS.get(feature_name, 0.0)
