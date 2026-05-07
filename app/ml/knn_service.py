from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings


class KNNModelService:
    def model_path(self, symbol: str) -> Path:
        return settings.model_dir / f"{symbol}_knn.joblib"

    def metrics_path(self, symbol: str) -> Path:
        return settings.model_dir / f"{symbol}_metrics.joblib"

    def train(
        self, symbol: str, dataset: pd.DataFrame, feature_cols: list[str], n_neighbors: int
    ) -> dict:
        X = dataset[feature_cols]
        y = dataset["label"]
        split = int(len(dataset) * 0.8)
        if split <= 20 or len(dataset) - split < 5:
            raise ValueError("Not enough data to train model")

        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        param_grid = {
            "knn__n_neighbors": sorted({n_neighbors, 3, 5, 7, 9}),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["minkowski", "manhattan"],
        }
        search = GridSearchCV(pipeline, param_grid=param_grid, scoring="f1", cv=3, n_jobs=1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        pred = best_model.predict(X_valid)
        metrics = {
            "accuracy": float(accuracy_score(y_valid, pred)),
            "precision": float(precision_score(y_valid, pred, zero_division=0)),
            "recall": float(recall_score(y_valid, pred, zero_division=0)),
            "f1": float(f1_score(y_valid, pred, zero_division=0)),
            "valid_positive_ratio": float(np.mean(y_valid)),
        }

        joblib.dump({"model": best_model, "features": feature_cols}, self.model_path(symbol))
        joblib.dump(metrics, self.metrics_path(symbol))
        return {
            "metrics": metrics,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "model_path": str(self.model_path(symbol)),
        }

    def load_model(self, symbol: str) -> tuple[Pipeline, list[str]]:
        payload = joblib.load(self.model_path(symbol))
        return payload["model"], payload["features"]

    def predict_latest(self, symbol: str, dataset: pd.DataFrame) -> dict:
        model, features = self.load_model(symbol)
        latest = dataset.iloc[[-1]]
        proba_up = float(model.predict_proba(latest[features])[0][1])
        label = int(proba_up >= 0.5)
        return {"label": label, "prob_up": proba_up}

    def load_metrics(self, symbol: str) -> dict[str, float]:
        return joblib.load(self.metrics_path(symbol))
