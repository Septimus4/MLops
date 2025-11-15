"""
Demo script to test the API endpoints.
"""

import json
import time

import requests

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("=" * 50)
    print("Testing /health endpoint")
    print("=" * 50)

    response = requests.get(f"{API_URL}/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict():
    """Test prediction endpoint."""
    print("=" * 50)
    print("Testing /predict endpoint")
    print("=" * 50)

    # Sample features
    features = {
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
        "OWN_CAR_AGE": 8.0,
    }

    response = requests.post(f"{API_URL}/predict", json={"features": features})

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Results:")
        print(f"  Risk Score: {result['risk_score']:.4f} ({result['risk_score']*100:.2f}%)")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Model Version: {result['model_version']}")
        print(f"  Number of features: {len(result['feature_values'])}")
    else:
        print(f"Error: {response.text}")

    print()


def test_drift():
    """Test drift endpoint."""
    print("=" * 50)
    print("Testing /drift endpoint")
    print("=" * 50)

    response = requests.get(f"{API_URL}/drift", params={"window_hours": 1})

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nDrift Analysis (last {result['window_hours']} hour(s)):")
        print(f"  Number of samples: {result['num_samples']}")
        print(f"  Features analyzed: {len(result['metrics'])}")

        if result["metrics"]:
            print("\n  Top 5 features by drift (z-score):")
            for i, metric in enumerate(result["metrics"][:5], 1):
                print(f"    {i}. {metric['feature_name']}: z={metric['z_score']:.3f}")
                print(
                    f"       Train mean: {metric['mean_train']:.4f}, Live mean: {metric['mean_live']:.4f}"
                )
        else:
            print("  No drift metrics available (need predictions in the time window)")
    else:
        print(f"Error: {response.text}")

    print()


def main():
    """Run demo tests."""
    print("\n")
    print("🚀 Home Credit Risk API Demo")
    print("=" * 50)
    print(f"API URL: {API_URL}")
    print()

    try:
        # Test health
        test_health()

        # Test prediction (make multiple requests to populate data)
        print("Making 3 prediction requests...")
        for i in range(3):
            print(f"\nRequest {i+1}/3:")
            test_predict()
            if i < 2:
                time.sleep(0.5)

        # Give database a moment to write
        print("Waiting 2 seconds for data to be logged...")
        time.sleep(2)

        # Test drift
        test_drift()

        print("=" * 50)
        print("✅ Demo completed successfully!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {API_URL}")
        print("Please ensure the API service is running:")
        print("  uvicorn src.service.main:app --reload")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
