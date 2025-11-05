from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import Iterator

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import config as api_config  # noqa: E402

@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "reference"))
    api_config.get_settings.cache_clear()
    yield
    api_config.get_settings.cache_clear()


@pytest.fixture
def api_client() -> TestClient:
    from api import app as api_app

    reload(api_app)  # ensure settings picked up from fixtures
    return TestClient(api_app.app)
