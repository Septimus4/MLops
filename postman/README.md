# Postman Collection

This folder contains a ready-to-import Postman collection and a local environment to smoke-test the FastAPI service.

## Files

- `MLops-HomeCredit.postman_collection.json` — Requests for:
  - GET `/health`
  - POST `/predict`
  - GET `/drift?window_hours=24`
  - GET `/`
  Each request has lightweight Postman tests to verify status and basic shape.

- `local.postman_environment.json` — Environment with `baseUrl` set to `http://localhost:8000`.

## How to use

1. Start the API locally (e.g., via Docker Compose or `uvicorn src.service.main:app --reload`).
2. In Postman, import both the collection and the environment JSON files from this folder.
3. Select the "Local MLops API" environment.
4. Run requests individually or use the **Collection Runner** to run them all.

## Optional: Run with Newman

If you prefer CLI runs, install Newman and execute the collection:

```zsh
# Install newman (once)
npm install -g newman

# Run with the local environment
newman run postman/MLops-HomeCredit.postman_collection.json \
  -e postman/local.postman_environment.json
```

You can add `--reporters cli,html` to generate an HTML report.
