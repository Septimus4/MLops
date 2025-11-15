# Home Credit Risk MLOps System

A complete MLOps implementation for the Home Credit Default Risk prediction, featuring model training, API service, drift monitoring, and comprehensive UI components.

## 🏗️ Architecture

This system consists of the following components:

- **Training Pipeline**: LightGBM model training with baseline statistics computation
- **FastAPI Backend**: REST API for predictions and drift monitoring
- **Gradio UI**: Interactive prediction interface
- **Streamlit Dashboard**: Real-time drift monitoring visualization
- **SQLite Database**: Prediction logging for drift analysis
- **Docker Deployment**: Containerized services with docker-compose orchestration
- **CI/CD Pipeline**: Automated testing and image publishing to GHCR

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                    # Kaggle CSVs (gitignored)
│   └── artifacts/              # model + baseline stats (gitignored)
├── src/
│   ├── training/               # Model training scripts
│   ├── service/                # FastAPI backend
│   ├── ui/                     # Gradio & Streamlit UIs
│   └── utils/                  # Shared utilities
├── tests/                      # Test suite
├── scripts/                    # Demo and utility scripts
├── docs/                       # MkDocs documentation
├── Dockerfile.api              # API service container
├── Dockerfile.ui               # UI services container
└── docker-compose.yml          # Service orchestration
```

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk) from Kaggle and place `application_train.csv` in `data/raw/`.

### 3. Train Model

```bash
python -m src.training.train_model
python -m src.training.compute_baseline_stats
```

### 4. Run with Docker Compose

```bash
docker-compose up
```

Access the services:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Gradio UI: http://localhost:7860
- Streamlit Dashboard: http://localhost:8501

### 5. Run Locally (Development)

```bash
# Terminal 1: Start API
uvicorn src.service.main:app --reload

# Terminal 2: Start Gradio UI
python -m src.ui.gradio_app

# Terminal 3: Start Streamlit Dashboard
streamlit run src/ui/streamlit_drift.py
```

## 📊 API Endpoints

- `GET /health` - Health check and model status
- `POST /predict` - Get risk prediction for a loan application
- `GET /drift` - Calculate feature drift metrics

Full API documentation available at `/docs` when running the service.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

Complete documentation (MkDocs) is in `docs/`. To develop locally without clashing with the API (which also uses port 8000), run on an alternate port:

```bash
mkdocs serve -a 127.0.0.1:9001
```

Then visit http://localhost:9001 for the live docs.

To build the static site (output in `site/`):

```bash
mkdocs build
```

You can optionally serve the generated `site/` directory via any static file server or mount it under a FastAPI route for a unified domain.

## 🐳 Docker Images

Images are automatically built and published to GitHub Container Registry:

- `ghcr.io/septimus4/mlops2-api:latest`
- `ghcr.io/septimus4/mlops2-gradio:latest`
- `ghcr.io/septimus4/mlops2-streamlit:latest`

## 🔧 Development

See [docs/deployment.md](docs/deployment.md) for detailed deployment instructions and [docs/architecture.md](docs/architecture.md) for system architecture details.

## 📝 License

See [LICENSE](LICENSE) file for details.
