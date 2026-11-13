"""Explicit training artifacts. Serving never loads untrusted or unverified pickle files."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from app.domain.schema import FEATURES
from app.ops import canonical, digest, file_digest, release_info, save_run
from app.services.data import load_csv
from app.services.model import dataset, splits, train_models


def build_artifact(output, csv_path=None):
    if csv_path:
        x, y = load_csv(Path(csv_path).read_text(), labelled=True)
        source = "operator-supplied labelled CSV"
    else:
        x, y = dataset()
        source = "synthetic demonstration, seed 42"
    xt, xv, xe, yt, yv, ye = splits(x, y)
    bundle = train_models(x, y)
    baseline_ap = float(yv.mean())  # AP of a constant-prior classifier.
    selected_ap = bundle["validation_ap"][bundle["winner"]]
    gates = {
        "finite_features": bool(np.isfinite(x.to_numpy()).all()),
        "feature_order": list(x.columns) == FEATURES,
        "disjoint_splits": not (
            set(xt.index) & set(xv.index)
            or set(xt.index) & set(xe.index)
            or set(xv.index) & set(xe.index)
        ),
        "validation_ap_exceeds_prior": bool(selected_ap > baseline_ap),
        "finite_thresholds": all(
            np.isfinite(t) and 0 <= t <= 1 for t in bundle["thresholds"].values()
        ),
    }
    configuration = {
        "seed": 42,
        "split": [0.6, 0.2, 0.2],
        "feature_order": FEATURES,
        "selection": "validation average precision",
        "threshold_selection": "validation F1",
        "dataset_kind": source,
        "rows": len(x),
        "dataset_sha256": digest(x.assign(fraud=y).to_csv(index=False).encode()),
    }
    metrics = {
        "validation_ap": bundle["validation_ap"],
        "baseline_validation_ap": baseline_ap,
        "test_final_evaluation": bundle["metrics"],
        "selected_model": bundle["winner"],
    }
    split_hashes = {
        label: digest(canonical(frame.index.tolist()))
        for label, frame in [("train", xt), ("validation", xv), ("test", xe)]
    }
    record, record_path = save_run(
        Path(output) / "runs",
        configuration,
        metrics,
        gates,
        {"split_index_sha256": split_hashes},
    )
    if not record["passed"]:
        raise ValueError(
            "Candidate failed validation gates; no model artifact was produced"
        )
    directory = Path(output) / "models" / record["run_id"]
    directory.mkdir(parents=True, mode=0o700)
    bundle["data_source"] = source
    artifact = directory / "models.joblib"
    joblib.dump(bundle, artifact, compress=3)
    artifact.chmod(0o600)
    manifest = {
        "artifact_format": "trusted-joblib-v1",
        "model_version": record["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": file_digest(artifact),
        "feature_order": FEATURES,
        "quality_gates": gates,
        "data": configuration,
        "release": record["release"],
        "selected_model": bundle["winner"],
        "model_parameters": {
            name: repr(model) for name, model in bundle["models"].items()
        },
        "serving_canary": {
            "input": xe.iloc[0].to_dict(),
            "probability": float(
                bundle["models"][bundle["winner"]].predict_proba(xe.iloc[[0]])[0, 1]
            ),
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    return manifest, directory, record_path


def load_artifact(directory, expected_sha256):
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError(
            "MODEL_ARTIFACT_SHA256 must pin an operator-reviewed trusted artifact"
        )
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    artifact = directory / "models.joblib"
    # Integrity is checked BEFORE deserialization. A hash is not a substitute for trust.
    if (
        file_digest(artifact) != expected_sha256
        or manifest["artifact_sha256"] != expected_sha256
    ):
        raise ValueError("Model artifact integrity check failed")
    if (
        manifest.get("feature_order") != FEATURES
        or not all(manifest.get("quality_gates", {}).values())
        or not manifest.get("quality_gates")
    ):
        raise ValueError("Model manifest schema or validation gates failed")
    for package in ["scikit-learn", "numpy", "xgboost"]:
        if manifest["release"]["packages"].get(package) != release_info()[
            "packages"
        ].get(package):
            raise ValueError(f"Training and serving {package} versions differ")
    bundle = joblib.load(artifact)
    if bundle["winner"] != manifest["selected_model"]:
        raise ValueError("Selected model disagrees with its manifest")
    import pandas as pd

    canary = manifest["serving_canary"]
    score = float(
        bundle["models"][bundle["winner"]].predict_proba(
            pd.DataFrame([canary["input"]], columns=FEATURES)
        )[0, 1]
    )
    if not np.isclose(score, canary["probability"], rtol=1e-8, atol=1e-10):
        raise ValueError("Training/serving canary prediction differs")
    bundle["model_version"] = manifest["model_version"]
    bundle["manifest"] = manifest
    return bundle
