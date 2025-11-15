# Home Credit Risk Service

## Overview

The Home Credit Risk Service is a complete MLOps implementation for predicting loan default risk using the Home Credit Default Risk dataset. It demonstrates best practices for machine learning operations including:

- **Model Training**: LightGBM classifier with automated training pipeline
- **API Service**: FastAPI-based REST API for predictions
- **Drift Monitoring**: Real-time feature drift detection using z-scores
- **User Interfaces**: Gradio for predictions, Streamlit for monitoring
- **Testing**: Comprehensive test suite with pytest
- **Containerization**: Docker-based deployment with docker-compose
- **CI/CD**: Automated testing and deployment with GitHub Actions

## Key Features

### Prediction Service
- REST API for loan default risk prediction
- Support for partial feature input with intelligent defaults
- Automatic prediction logging for drift analysis

### Drift Monitoring
- Real-time feature drift detection
- Configurable time windows for analysis
- Visual dashboard for drift metrics
- Z-score based drift quantification

### Developer-Friendly
- Comprehensive API documentation with OpenAPI/Swagger
- Unit and integration tests
- Type hints and clear code structure
- Detailed logging and error handling

### Production-Ready
- Docker containerization
- Horizontal scalability
- Health checks and monitoring
- CI/CD pipeline with automated testing

## Use Case

Home Credit is a consumer finance provider that serves customers with little or no credit history. This service predicts the likelihood of loan default to help make informed lending decisions.

### Business Value

- **Risk Assessment**: Quantify default risk for each application
- **Operational Monitoring**: Track model performance drift over time
- **Compliance**: Maintain audit trail of all predictions
- **Efficiency**: Automated, scalable prediction service

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**
   
   Download the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk) and place `application_train.csv` in `data/raw/`.

3. **Train Model**
   ```bash
   python -m src.training.train_model
   python -m src.training.compute_baseline_stats
   ```

4. **Run Services**
   ```bash
   docker-compose up
   ```

5. **Access Interfaces**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Gradio UI: http://localhost:7860
   - Streamlit Dashboard: http://localhost:8501

## Documentation Sections

- [Architecture](architecture.md) - System design and components
- [API Documentation](api.md) - Endpoint specifications and examples
- [Drift Monitoring](drift_monitoring.md) - Understanding and using drift detection
- [Deployment](deployment.md) - Deployment guide and operations

## Technologies

- **ML Framework**: LightGBM, scikit-learn
- **API Framework**: FastAPI
- **UI Frameworks**: Gradio, Streamlit
- **Database**: SQLite
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Documentation**: MkDocs with Material theme
- **Testing**: pytest

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/Septimus4/MLOps2).
