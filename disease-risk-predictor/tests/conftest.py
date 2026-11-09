import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from risk.train import train  # noqa: E402


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """Train a throw-away model so tests do not depend on a previous run."""
    out = tmp_path_factory.mktemp("model")
    train(ROOT / "data" / "diabetes.csv", out / "model.joblib", out / "reports", plots=False)
    import os
    os.environ["MODEL_PATH"] = str(out / "model.joblib")
    from app.main import app
    with TestClient(app) as c:
        yield c
