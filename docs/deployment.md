# Deployment Guide

## Overview

The Home Credit Risk Service can be deployed in multiple ways:
- Local development
- Docker Compose (recommended for testing)
- Kubernetes (production)
- Cloud platforms (AWS, GCP, Azure)

## Prerequisites

### All Deployments

1. Trained model artifact (`home_credit_model.joblib`)
2. Baseline statistics (`baseline_stats.json`)
3. Valid `requirements.txt`

### Docker Deployments

- Docker Engine 20.10+
- Docker Compose 2.0+

### Kubernetes Deployments

- kubectl configured
- Kubernetes cluster access
- Helm (optional, recommended)

## Local Development

### Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**:
   - Get Home Credit dataset from Kaggle
   - Place `application_train.csv` in `data/raw/`

4. **Train model**:
   ```bash
   python -m src.training.train_model
   python -m src.training.compute_baseline_stats
   ```

### Running Services

**API Service**:
```bash
uvicorn src.service.main:app --reload --host 0.0.0.0 --port 8000
```

**Gradio UI**:
```bash
python -m src.ui.gradio_app
```

**Streamlit Dashboard**:
```bash
streamlit run src/ui/streamlit_drift.py
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

## Docker Compose Deployment

### Prerequisites

Ensure model artifacts exist:
```bash
ls data/artifacts/
# Should show:
# - home_credit_model.joblib
# - baseline_stats.json
```

### Building Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api
```

### Running Services

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# Start specific service
docker-compose up api

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Accessing Services

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Gradio: http://localhost:7860
- Streamlit: http://localhost:8501

### Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  api:
    environment:
      - MODEL_PATH=/app/data/artifacts/home_credit_model.joblib
      - DB_PATH=/app/data/artifacts/predictions.db
      - BASELINE_PATH=/app/data/artifacts/baseline_stats.json
    ports:
      - "8000:8000"  # Change external port
```

### Troubleshooting

**Container won't start**:
```bash
# Check logs
docker-compose logs api

# Rebuild without cache
docker-compose build --no-cache api

# Check if port is in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows
```

**Model not found**:
```bash
# Verify volume mount
docker-compose exec api ls -l /app/data/artifacts/

# Copy model into running container
docker cp data/artifacts/home_credit_model.joblib \
  $(docker-compose ps -q api):/app/data/artifacts/
```

## GitHub Container Registry (GHCR)

### Publishing Images

Images are automatically published to GHCR via GitHub Actions on push to `main`.

Manual publishing:

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag images
docker tag mlops2-api:latest ghcr.io/septimus4/mlops2-api:latest
docker tag mlops2-gradio:latest ghcr.io/septimus4/mlops2-gradio:latest
docker tag mlops2-streamlit:latest ghcr.io/septimus4/mlops2-streamlit:latest

# Push images
docker push ghcr.io/septimus4/mlops2-api:latest
docker push ghcr.io/septimus4/mlops2-gradio:latest
docker push ghcr.io/septimus4/mlops2-streamlit:latest
```

### Using Published Images

Update `docker-compose.yml`:

```yaml
services:
  api:
    image: ghcr.io/septimus4/mlops2-api:latest
    # Remove build section
```

Pull and run:
```bash
docker-compose pull
docker-compose up -d
```

## Kubernetes Deployment

### Basic Deployment

1. **Create namespace**:
   ```bash
   kubectl create namespace mlops
   ```

2. **Create secrets**:
   ```bash
   # Create model artifact secret
   kubectl create secret generic model-artifacts \
     --from-file=data/artifacts/home_credit_model.joblib \
     --from-file=data/artifacts/baseline_stats.json \
     -n mlops
   ```

3. **Deploy API service**:
   ```yaml
   # k8s/api-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: api
     namespace: mlops
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: api
     template:
       metadata:
         labels:
           app: api
       spec:
         containers:
         - name: api
           image: ghcr.io/septimus4/mlops2-api:latest
           ports:
           - containerPort: 8000
           env:
           - name: MODEL_PATH
             value: /app/data/artifacts/home_credit_model.joblib
           volumeMounts:
           - name: artifacts
             mountPath: /app/data/artifacts
         volumes:
         - name: artifacts
           secret:
             secretName: model-artifacts
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: api
     namespace: mlops
   spec:
     selector:
       app: api
     ports:
     - port: 8000
       targetPort: 8000
     type: LoadBalancer
   ```

4. **Apply configuration**:
   ```bash
   kubectl apply -f k8s/api-deployment.yaml
   ```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-ingress
  namespace: mlops
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: mlops-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 8000
```

### Persistent Storage

For database persistence:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: db-storage
  namespace: mlops
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

Mount in deployment:
```yaml
volumeMounts:
- name: db-storage
  mountPath: /app/data/artifacts
volumes:
- name: db-storage
  persistentVolumeClaim:
    claimName: db-storage
```

### Helm Deployment

Create Helm chart:

```bash
helm create mlops-chart
```

Customize `values.yaml`:
```yaml
api:
  replicaCount: 3
  image:
    repository: ghcr.io/septimus4/mlops2-api
    tag: latest
  service:
    type: LoadBalancer
    port: 8000

gradio:
  replicaCount: 1
  image:
    repository: ghcr.io/septimus4/mlops2-gradio
    tag: latest

streamlit:
  replicaCount: 1
  image:
    repository: ghcr.io/septimus4/mlops2-streamlit
    tag: latest
```

Install:
```bash
helm install mlops ./mlops-chart -n mlops
```

## Cloud Platform Deployments

### AWS ECS

1. Create ECR repositories
2. Push images to ECR
3. Create ECS cluster
4. Define task definitions
5. Create services
6. Configure load balancer

### AWS EKS

1. Create EKS cluster
2. Configure kubectl
3. Follow Kubernetes deployment steps above

### GCP Cloud Run

```bash
# Deploy API
gcloud run deploy api \
  --image ghcr.io/septimus4/mlops2-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Deploy Gradio
gcloud run deploy gradio \
  --image ghcr.io/septimus4/mlops2-gradio:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_URL=https://api-xxx.run.app
```

### Azure Container Instances

```bash
# Create resource group
az group create --name mlops-rg --location eastus

# Deploy API
az container create \
  --resource-group mlops-rg \
  --name api \
  --image ghcr.io/septimus4/mlops2-api:latest \
  --dns-name-label mlops-api \
  --ports 8000
```

## Production Considerations

### Security

1. **Enable HTTPS**:
   - Use reverse proxy (Nginx, Traefik)
   - Configure SSL certificates
   - Redirect HTTP to HTTPS

2. **Authentication**:
   - Implement OAuth2/JWT
   - Use API keys
   - Rate limiting

3. **Network Security**:
   - Use private networks
   - Configure firewalls
   - Enable VPN access

### Monitoring

1. **Application Metrics**:
   - Prometheus for metrics
   - Grafana for dashboards
   - Custom metrics (predictions/sec, latency, drift)

2. **Logging**:
   - Centralized logging (ELK, Loki)
   - Structured logging
   - Log rotation

3. **Tracing**:
   - Distributed tracing (Jaeger, Zipkin)
   - Request tracking
   - Performance profiling

### Backup and Recovery

1. **Database Backups**:
   ```bash
   # Backup SQLite
   sqlite3 predictions.db ".backup predictions_backup.db"
   
   # Restore
   cp predictions_backup.db predictions.db
   ```

2. **Model Versioning**:
   - Store models in S3/GCS
   - Version with timestamps
   - Keep multiple versions

3. **Disaster Recovery**:
   - Document recovery procedures
   - Test recovery regularly
   - Maintain off-site backups

### Scaling

1. **Horizontal Scaling**:
   - Add API replicas
   - Use load balancer
   - Shared database

2. **Database Scaling**:
   - Migrate to PostgreSQL
   - Read replicas
   - Connection pooling

3. **Caching**:
   - Redis for hot data
   - Cache drift metrics
   - Feature caching

### Updates and Rollback

1. **Blue-Green Deployment**:
   ```bash
   # Deploy new version (green)
   kubectl apply -f api-deployment-v2.yaml
   
   # Test green deployment
   # Switch traffic
   kubectl patch service api -p '{"spec":{"selector":{"version":"v2"}}}'
   
   # Rollback if needed
   kubectl patch service api -p '{"spec":{"selector":{"version":"v1"}}}'
   ```

2. **Canary Deployment**:
   - Deploy new version to subset of pods
   - Monitor metrics
   - Gradually increase traffic
   - Rollback if issues detected

## Maintenance

### Regular Tasks

1. **Daily**:
   - Check drift metrics
   - Review error logs
   - Monitor resource usage

2. **Weekly**:
   - Review model performance
   - Analyze prediction patterns
   - Check for anomalies

3. **Monthly**:
   - Evaluate retraining needs
   - Update dependencies
   - Security patches

4. **Quarterly**:
   - Model retraining
   - Architecture review
   - Performance optimization

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Kubernetes health
kubectl get pods -n mlops
kubectl describe pod <pod-name> -n mlops

# Resource usage
docker stats
kubectl top pods -n mlops
```

## Troubleshooting

### Common Issues

1. **503 Service Unavailable**:
   - Check if model is loaded
   - Verify model path
   - Check startup logs

2. **High Latency**:
   - Monitor resource usage
   - Check database performance
   - Enable caching
   - Scale horizontally

3. **Memory Issues**:
   - Increase container limits
   - Check for memory leaks
   - Monitor model size

4. **Database Locked**:
   - Use connection pooling
   - Migrate to PostgreSQL
   - Implement retry logic

## Support

For deployment issues:
- Check logs: `docker-compose logs` or `kubectl logs`
- Review documentation
- Open GitHub issue
- Contact support team
