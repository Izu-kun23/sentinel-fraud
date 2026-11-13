"""Aggregate distribution monitoring; never persists raw transactions or labels."""

import json, os, sqlite3
from pathlib import Path
import numpy as np


class FeatureMonitor:
    def __init__(self, path=None):
        self.path = str(path or os.getenv("MODEL_MONITOR_DB", "data/model-monitor.db"))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS distributions(version TEXT PRIMARY KEY,payload TEXT)"
            )

    @staticmethod
    def baseline(frame):
        result = {}
        for feature in frame.columns:
            finite = np.unique(np.quantile(frame[feature], np.linspace(0, 1, 11)))
            if len(finite) < 2:
                finite = np.array([float(finite[0]) - 0.5, float(finite[0]) + 0.5])
            edges = np.concatenate(([-np.inf], finite[1:-1], [np.inf]))
            # JSON stores interior edges only; infinities are reconstructed.
            result[feature] = {
                "interior_edges": edges[1:-1].tolist(),
                "baseline_counts": np.histogram(frame[feature], edges)[0].tolist(),
                "observed_counts": [0] * (len(edges) - 1),
            }
        return {"samples": 0, "features": result}

    def observe(self, version, baseline_frame, frame):
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM distributions WHERE version=?", (version,)
            ).fetchone()
            state = json.loads(row[0]) if row else self.baseline(baseline_frame)
            for feature, distribution in state["features"].items():
                edges = [-np.inf, *distribution["interior_edges"], np.inf]
                counts = np.histogram(frame[feature], edges)[0]
                distribution["observed_counts"] = (
                    np.array(distribution["observed_counts"]) + counts
                ).tolist()
            state["samples"] += len(frame)
            db.execute(
                "INSERT INTO distributions(version,payload) VALUES(?,?) ON CONFLICT(version) DO UPDATE SET payload=excluded.payload",
                (version, json.dumps(state, allow_nan=False)),
            )

    def report(self, version):
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT payload FROM distributions WHERE version=?", (version,)
            ).fetchone()
        if not row:
            return {
                "model_version": version,
                "samples": 0,
                "alerts_evaluated": False,
                "features": {},
                "scope": "No raw transactions or labels are stored.",
            }
        state = json.loads(row[0])
        scores = {}
        for feature, distribution in state["features"].items():
            baseline = np.array(distribution["baseline_counts"], dtype=float) + 0.5
            current = np.array(distribution["observed_counts"], dtype=float) + 0.5
            baseline /= baseline.sum()
            current /= current.sum()
            scores[feature] = float(
                np.sum((current - baseline) * np.log(current / baseline))
            )
        return {
            "model_version": version,
            "samples": state["samples"],
            "alerts_evaluated": state["samples"] >= 100,
            "psi": scores,
            "alerts": (
                [feature for feature, score in scores.items() if score > 0.2]
                if state["samples"] >= 100
                else []
            ),
            "scope": "Heuristic distribution shift only, not fraud accuracy or concept-drift detection. Labels are required for precision/recall monitoring.",
        }
