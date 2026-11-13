TITLE = "Sentinel Fraud"

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).parent / "web"
app = FastAPI(title=TITLE, version="1.0.0")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "project": TITLE}


from app.services.model import RiskService
from app.domain.schema import Transaction
from fastapi import Query, HTTPException
from pydantic import BaseModel, Field
from app.services.data import load_csv

service = RiskService()


@app.get("/api/overview")
def overview():
    b = service.ensure()
    xe = b["test_x"]
    p = b["models"][b["winner"]].predict_proba(xe)[:, 1]
    t = b["thresholds"][b["winner"]]
    rows = [
        {
            "id": f"TXN-{i:05d}",
            **{k: float(v) for k, v in row.items()},
            "probability": float(prob),
            "flagged": bool(prob >= t),
        }
        for (i, row), prob in zip(xe.iterrows(), p)
    ]
    rows.sort(key=lambda r: -r["probability"])
    return {
        "winner": b["winner"],
        "model_version": b["model_version"],
        "metrics": b["metrics"],
        "validation_ap": b["validation_ap"],
        "prevalence": b["prevalence"],
        "transactions": len(xe),
        "flagged": int((p >= t).sum()),
        "rows": rows[:30],
        "histogram": __import__("numpy")
        .histogram(p, bins=12, range=(0, 1))[0]
        .tolist(),
        "source": b.get("data_source", "Synthetic data; not a production fraud model"),
    }


@app.post("/api/predict")
def predict(body: Transaction, model: str | None = Query(default=None)):
    if model and model not in {"Logistic Regression", "Random Forest", "XGBoost"}:
        raise HTTPException(422, "Unknown model")
    return service.score(body.model_dump(), model)


class CSVImport(BaseModel):
    csv: str = Field(min_length=10, max_length=1_000_000)


@app.post("/api/transactions/import")
def import_transactions(body: CSVImport):
    try:
        frame = load_csv(body.csv)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(frame) > 1000:
        raise HTTPException(422, "Interactive batch scoring supports up to 1,000 rows")
    b = service.ensure()
    name = b["winner"]
    p = b["models"][name].predict_proba(frame)[:, 1]
    service.observe(frame)
    threshold = b["thresholds"][name]
    rows = [
        {
            "id": f"CSV-{i+1:05d}",
            **row,
            "probability": float(prob),
            "flagged": bool(prob >= threshold),
        }
        for i, (row, prob) in enumerate(zip(frame.to_dict("records"), p))
    ]
    return {
        "rows": sorted(rows, key=lambda r: -r["probability"]),
        "count": len(rows),
        "flagged": sum(r["flagged"] for r in rows),
        "model": name,
        "model_version": b["model_version"],
        "note": "Imported raw rows are not persisted; only aggregate distribution counts are monitored.",
    }


from app.ops import install_operations

operations_store = install_operations(app)


@app.get("/api/ready")
def ready():
    import os

    if os.getenv("APP_ENV") == "production":
        try:
            bundle = service.ensure()
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(
                503, "Pinned model artifact is unavailable or invalid"
            ) from exc
        return {
            "ready": True,
            "model_version": bundle["model_version"],
            "mode": "artifact serving",
        }
    return {
        "ready": True,
        "model_loaded": service.bundle is not None,
        "mode": "development demo",
    }


@app.get("/api/model-monitoring")
def model_monitoring():
    from app.monitoring import FeatureMonitor

    if service.bundle is None:
        return {"status": "no model loaded", "samples": 0}
    return FeatureMonitor().report(service.bundle["model_version"])
