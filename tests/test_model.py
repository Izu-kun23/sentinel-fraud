import numpy as np, pytest
from app.services.model import dataset, splits, RiskService
from app.domain.schema import Transaction


def test_data_reproducible_and_disjoint():
    x, y = dataset()
    x2, y2 = dataset()
    assert x.equals(x2) and y.equals(y2)
    a, b, c, *_ = splits(x, y)
    assert set(a.index).isdisjoint(b.index)
    assert set(a.index).isdisjoint(c.index)
    assert set(b.index).isdisjoint(c.index)
    assert 0 < y.mean() < 0.2


def test_input_validation():
    with pytest.raises(ValueError):
        Transaction(
            amount=-1,
            hour=25,
            distance_km=0,
            transactions_24h=0,
            account_age_days=1,
            international=2,
        )


def test_models_and_shap():
    s = RiskService()
    b = s.ensure()
    assert len(b["models"]) == 3
    assert b["winner"] == max(b["validation_ap"], key=b["validation_ap"].get)
    for name in b["models"]:
        result = s.score(
            {
                "amount": 500,
                "hour": 2,
                "distance_km": 300,
                "transactions_24h": 8,
                "account_age_days": 30,
                "international": 1,
            },
            name,
        )
        assert 0 <= result["probability"] <= 1
        assert len(result["explanation"]["contributions"]) == 6
        assert all(
            np.isfinite(f["value"]) for f in result["explanation"]["contributions"]
        )


def test_api():
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert c.get("/api/overview").json()["transactions"] == 1200
    assert c.post("/api/predict", json={}).status_code == 422


def test_csv_import():
    from app.services.data import load_csv
    from fastapi.testclient import TestClient
    from app.main import app
    from pathlib import Path

    text = Path("data/sample_transactions.csv").read_text()
    assert load_csv(text).shape == (2, 6)
    result = TestClient(app).post("/api/transactions/import", json={"csv": text})
    assert result.status_code == 200 and result.json()["count"] == 2
    with pytest.raises(ValueError):
        load_csv("amount,hour\n1,2")
    with pytest.raises(ValueError):
        load_csv(text.replace("35.50", "-1"))
