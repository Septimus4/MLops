# Architecture

## System Overview

The Home Credit Risk Service is built with a microservices architecture, separating concerns into distinct, independently scalable components.

```
┌─────────────┐
│   Gradio    │ ──┐
│     UI      │   │
└─────────────┘   │
                  │
┌─────────────┐   │    ┌──────────────┐
│  Streamlit  │ ──┼───→│   FastAPI    │
│  Dashboard  │   │    │   Service    │
└─────────────┘   │    └──────────────┘
                  │           │
┌─────────────┐   │           │
│   External  │ ──┘           ↓
│   Clients   │         ┌──────────┐
└─────────────┘         │  SQLite  │
                        │    DB    │
                        └──────────┘
```

## Components

### 1. Training Pipeline

**Location**: `src/training/`

Responsible for:
- Data ingestion and preprocessing
- Model training with LightGBM
- Baseline statistics computation for drift detection
- Model artifact serialization

**Key Files**:
- `train_model.py` - Model training script
- `compute_baseline_stats.py` - Baseline statistics computation
- `feature_config.py` - Feature definitions and defaults

**Outputs**:
- `data/artifacts/home_credit_model.joblib` - Trained model
- `data/artifacts/baseline_stats.json` - Feature statistics

### 2. FastAPI Service

**Location**: `src/service/`

Provides REST API for predictions and monitoring.

**Key Files**:
- `main.py` - FastAPI application with endpoints
- `schemas.py` - Pydantic models for request/response validation
- `model_loader.py` - Model loading and prediction
- `feature_processor.py` - Feature engineering and validation
- `db.py` - Database operations
- `drift.py` - Drift detection logic

**Endpoints**:
- `GET /health` - Service health check
- `POST /predict` - Make predictions
- `GET /drift` - Get drift metrics

### 3. Gradio UI

**Location**: `src/ui/gradio_app.py`

Interactive web interface for making predictions.

**Features**:
- Input forms for loan application features
- Real-time risk score calculation
- User-friendly result presentation
- Responsive design

**Technology**: Gradio Blocks API

### 4. Streamlit Dashboard

**Location**: `src/ui/streamlit_drift.py`

Real-time monitoring dashboard for feature drift.

**Features**:
- Configurable time windows
- Drift metric visualization
- Color-coded alerts
- Auto-refresh capability
- Feature comparison tables

**Technology**: Streamlit

### 5. SQLite Database

**Location**: `data/artifacts/predictions.db`

Stores prediction history for drift analysis.

**Tables**:
- `predictions` - Logged predictions with features and outcomes
- `drift_metrics` - Optional pre-computed drift metrics

## Data Flow

### Prediction Flow

1. Client sends feature data to `/predict`
2. API validates and processes features
3. Missing features filled with defaults
4. Model makes prediction
5. Prediction logged to database
6. Response returned to client

### Drift Monitoring Flow

1. Dashboard requests drift metrics via `/drift?window_hours=24`
2. API queries predictions from specified time window
3. Features extracted and aggregated
4. Z-scores computed against baseline
5. Metrics sorted and returned
6. Dashboard visualizes results

## Design Principles

### Separation of Concerns

Each component has a single, well-defined responsibility:
- Training pipeline: Model creation
- API service: Prediction and monitoring
- UIs: User interaction
- Database: Data persistence

### Stateless API

The API service is stateless, storing no session information. All state is persisted to the database, enabling:
- Horizontal scaling
- Load balancing
- Container restarts without data loss

### Configuration via Environment

Key paths and URLs are configurable via environment variables:
- `MODEL_PATH` - Model artifact location
- `BASELINE_PATH` - Baseline statistics location
- `DB_PATH` - Database location
- `API_URL` - API endpoint for UIs

### Default-First Feature Processing

Missing features are automatically filled with sensible defaults, making the API:
- More forgiving of incomplete data
- Easier to integrate
- Compatible with partial information scenarios

### Container-Native Design

All components are designed to run in containers:
- No host filesystem dependencies
- Volume mounts for persistence
- Environment-based configuration
- Health checks for orchestration

## Technology Choices

### LightGBM

Chosen for:
- Excellent performance on tabular data
- Fast training and inference
- Small model size
- Built-in feature importance

### FastAPI

Chosen for:
- Automatic OpenAPI documentation
- Type validation with Pydantic
- High performance (async support)
- Modern Python features

### SQLite

Chosen for:
- Simple deployment (no separate DB server)
- ACID compliance
- Good performance for moderate load
- Easy backup and migration

### Gradio

Chosen for:
- Rapid UI development
- Clean, professional appearance
- Easy deployment
- Built-in API client

### Streamlit

Chosen for:
- Excellent data visualization
- Reactive programming model
- Quick iteration
- Rich widget library

## Scalability Considerations

### Current Architecture

- Single API instance
- SQLite database
- Suitable for: 10-100 requests/second

### Scaling Options

**Horizontal API Scaling**:
- Add API instances behind load balancer
- Migrate to PostgreSQL or MySQL
- Add Redis cache for hot data

**Vertical Scaling**:
- Increase container resources
- Use faster storage for database
- Enable async database operations

**Advanced Options**:
- Kubernetes deployment
- Message queue for async predictions
- Separate read/write databases
- Caching layer (Redis)

## Security Considerations

### Current Implementation

- Input validation via Pydantic
- SQL injection prevention (parameterized queries)
- CORS configured for cross-origin requests

### Production Enhancements

- Authentication/authorization (OAuth2, JWT)
- HTTPS/TLS encryption
- Rate limiting
- Input sanitization
- Secrets management
- Network segmentation

## Monitoring and Observability

### Built-in Features

- Health check endpoint
- Drift detection
- Prediction logging

### Recommended Additions

- Application metrics (Prometheus)
- Distributed tracing (Jaeger, Zipkin)
- Log aggregation (ELK stack)
- Alerting (PagerDuty, Slack)

## Deployment Architecture

### Development

```
Host Machine
├── API (port 8000)
├── Gradio (port 7860)
└── Streamlit (port 8501)
```

### Docker Compose

```
Docker Network
├── api container
├── gradio container
├── streamlit container
└── shared volume (data/artifacts)
```

### Production (Kubernetes)

```
Kubernetes Cluster
├── API Deployment (3 replicas)
├── Gradio Deployment (1 replica)
├── Streamlit Deployment (1 replica)
├── Ingress (HTTPS termination)
└── PersistentVolume (model & DB)
```

## Future Enhancements

1. **Model Registry**: Track multiple model versions
2. **A/B Testing**: Compare model variants
3. **Feature Store**: Centralized feature management
4. **Model Retraining**: Automated retraining pipeline
5. **Performance Monitoring**: Latency, throughput metrics
6. **Cost Tracking**: Resource usage monitoring
