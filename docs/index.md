# Home Credit MLOps Documentation

Welcome to the Home Credit Default Risk MLOps documentation hub. This site captures how the scoring pipeline, production API, monitoring stack, and CI/CD workflows are put together so you can operate, extend, and deploy the project with confidence.

The repository ships with:

- A production-ready FastAPI service that loads the latest registry model once, validates requests with Pydantic schemas, and persists structured JSONL logs.
- A Gradio manual tester for human-in-the-loop experiments with the same inference path as the API.
- A Streamlit dashboard that visualises operational KPIs, score distributions, and Evidently-generated drift reports.
- Docker images and GitHub Actions workflows that build, test, and publish the API and dashboard to GitHub Container Registry.

Use the navigation on the left to explore the areas that matter most to you:

- **Getting Started** walks through local setup, environment management, and a quick pipeline run.
- **System Architecture** explains how the training code, API, monitoring, and registry cooperate.
- **API & UX Guide** documents endpoints, expected payloads, and how to use the Gradio UI.
- **Monitoring & Drift** covers operational metrics, log aggregation, and drift triage steps.
- **Deployment** highlights Docker usage and the CI/CD pipeline.
- **Data & Artifacts** describes where reference data, model assets, and runtime logs live.
- **Testing & CI** outlines automated checks and recommendations for extending coverage.
- **OpenAPI Spec** links to the generated schema used by clients and tooling.

For quick context on course deliverables, the existing `requirements-mapping.md` page summarises how the repository satisfies the rubric.
