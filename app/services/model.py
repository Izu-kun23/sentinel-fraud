"""Leakage-safe training, validation-only threshold selection and held-out evaluation."""

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from app.domain.schema import FEATURES


def dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "amount": np.round(rng.lognormal(3.8, 1.2, n), 2),
            "hour": rng.integers(0, 24, n),
            "distance_km": rng.exponential(80, n),
            "transactions_24h": rng.poisson(3, n),
            "account_age_days": rng.integers(1, 2200, n),
            "international": rng.binomial(1, 0.12, n),
        }
    )
    logit = (
        -5
        + 1.25 * (x.amount > 220)
        + 1.4 * (x.hour < 5)
        + 1.6 * (x.distance_km > 160)
        + 1.6 * (x.transactions_24h > 5)
        + 1.5 * (x.account_age_days < 90)
        + 1.0 * x.international
    )
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="fraud")
    return x, y


def splits(x, y):
    train, test, yt, ytest = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    train, val, yt, yval = train_test_split(
        train, yt, test_size=0.25, stratify=yt, random_state=42
    )
    return train, val, test, yt, yval, ytest


def threshold(y, p):
    precision, recall, values = precision_recall_curve(y, p)
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    )
    return float(values[int(np.argmax(f1))])


def metrics(y, p, t):
    pred = p >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "threshold": t,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_models(x=None, y=None):
    from xgboost import XGBClassifier

    if x is None:
        x, y = dataset()
    xt, xv, xe, yt, yv, ye = splits(x, y)
    candidates = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=2,
            eval_metric="logloss",
        ),
    }
    results = {}
    validation = {}
    thresholds = {}
    for name, model in candidates.items():
        model.fit(xt, yt)
        pv = model.predict_proba(xv)[:, 1]
        t = threshold(yv, pv)
        thresholds[name] = t
        validation[name] = float(average_precision_score(yv, pv))
        results[name] = metrics(ye, model.predict_proba(xe)[:, 1], t)
    winner = max(validation, key=validation.get)
    return {
        "models": candidates,
        "winner": winner,
        "thresholds": thresholds,
        "metrics": results,
        "validation_ap": validation,
        "prevalence": float(y.mean()),
        "test_x": xe,
        "test_y": ye,
        "train_x": xt,
        "train_y": yt,
    }


class RiskService:
    def __init__(self):
        self.bundle = None

    def ensure(self):
        if self.bundle is None:
            import os

            artifact_directory = os.getenv("MODEL_ARTIFACT_DIR")
            if artifact_directory:
                from app.training import load_artifact

                self.bundle = load_artifact(
                    artifact_directory, os.getenv("MODEL_ARTIFACT_SHA256")
                )
            elif os.getenv("APP_ENV") == "production":
                raise RuntimeError(
                    "Production serving requires a pinned MODEL_ARTIFACT_DIR and MODEL_ARTIFACT_SHA256; request-time training is disabled"
                )
            else:
                self.bundle = train_models()
                from app.ops import release_info, digest, canonical

                release = release_info()
                self.bundle["model_version"] = (
                    "development-synthetic-"
                    + digest(
                        canonical(
                            {
                                "app": release["application_sha256"],
                                "dependencies": release["dependency_lock_sha256"],
                            }
                        )
                    )[:12]
                )
        return self.bundle

    def score(self, data, model_name=None):
        b = self.ensure()
        name = model_name or b["winner"]
        model = b["models"][name]
        frame = pd.DataFrame([data], columns=FEATURES)
        p = float(model.predict_proba(frame)[0, 1])
        t = b["thresholds"][name]
        self.observe(frame)
        return {
            "probability": p,
            "threshold": t,
            "flagged": p >= t,
            "model": name,
            "data_source": b.get("data_source", "synthetic demonstration"),
            "model_version": b["model_version"],
            "explanation": self.explain(frame, name),
        }

    def observe(self, frame):
        import sqlite3
        from app.monitoring import FeatureMonitor

        b = self.ensure()
        try:
            FeatureMonitor().observe(b["model_version"], b["train_x"], frame)
        except sqlite3.Error:
            pass

    def explain(self, frame, name):
        import shap

        b = self.ensure()
        model = b["models"][name]
        if name == "Logistic Regression":
            scaler, linear = model.steps[0][1], model.steps[1][1]
            values = shap.LinearExplainer(linear, scaler.transform(b["train_x"]))(
                scaler.transform(frame)
            ).values[0]
            space = "log-odds"
        else:
            values = np.asarray(shap.TreeExplainer(model)(frame).values)
            if values.ndim == 3:
                values = values[:, :, 1]
            values = values[0]
            space = "log-odds" if name == "XGBoost" else "probability"
        return {
            "method": "SHAP",
            "output_space": space,
            "contributions": sorted(
                [{"feature": f, "value": float(v)} for f, v in zip(FEATURES, values)],
                key=lambda a: abs(a["value"]),
                reverse=True,
            ),
        }
