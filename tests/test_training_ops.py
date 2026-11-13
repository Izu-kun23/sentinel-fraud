import json
import numpy as np
import pytest
from app.training import build_artifact, load_artifact
from app.services.model import RiskService, dataset
from app.monitoring import FeatureMonitor


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return build_artifact(tmp_path_factory.mktemp("training"))


def test_explicit_artifact_and_serving_canary(artifact, monkeypatch):
    manifest, directory, record_path = artifact
    assert all(manifest["quality_gates"].values())
    assert json.loads(record_path.read_text())["passed"]
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(directory))
    monkeypatch.setenv("MODEL_ARTIFACT_SHA256", manifest["artifact_sha256"])
    service = RiskService()
    bundle = service.ensure()
    assert bundle["model_version"] == manifest["model_version"]
    score = service.score(manifest["serving_canary"]["input"])
    assert np.isclose(score["probability"], manifest["serving_canary"]["probability"])


def test_bad_hash_rejected_before_deserialization(artifact, monkeypatch):
    _, directory, _ = artifact
    monkeypatch.setattr(
        "app.training.joblib.load",
        lambda path: pytest.fail("Unverified deserialization"),
    )
    with pytest.raises(ValueError, match="integrity"):
        load_artifact(directory, "0" * 64)


def test_production_never_trains_on_request(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("MODEL_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(
        "app.services.model.train_models",
        lambda: pytest.fail("Training was invoked during production serving"),
    )
    with pytest.raises(RuntimeError, match="request-time training is disabled"):
        RiskService().ensure()


def test_aggregate_distribution_monitoring(tmp_path):
    x, _ = dataset()
    monitor = FeatureMonitor(tmp_path / "monitor.db")
    shifted = x.iloc[:100].copy()
    shifted["amount"] = 1000000
    shifted["distance_km"] = 20000
    monitor.observe("v1", x, shifted)
    report = monitor.report("v1")
    assert report["samples"] == 100 and report["alerts_evaluated"]
    assert "amount" in report["alerts"] and "distance_km" in report["alerts"]
    assert monitor.report("v2")["samples"] == 0
    assert "accuracy" in report["scope"]  # explicitly disclaims accuracy inference
