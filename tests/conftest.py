from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import AsyncIterator, Iterator

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from api import config as api_config  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "reference"))
    api_config.get_settings.cache_clear()
    yield
    api_config.get_settings.cache_clear()


@pytest.fixture
async def api_client() -> AsyncIterator[AsyncClient]:
    from api import app as api_app

    reload(api_app)  # ensure settings picked up from fixtures
    transport = ASGITransport(app=api_app.app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
