from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings


class KNNModelService:
    def model_path(self, symbol: str) -> Path:
        return settings.model_dir / f"{symbol}_knn.joblib"

    def metrics_path(self, symbol: str) -> Path:
        return settings.model_dir / f"{symbol}_metrics.joblib"

    def range_model_path(self, symbol: str) -> Path:
        return settings.model_dir / f"{symbol}_knn_range.joblib"

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
        self._train_range_models(symbol, dataset, feature_cols, n_neighbors)
        return {
            "metrics": metrics,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "model_path": str(self.model_path(symbol)),
        }

    def _train_range_models(
        self, symbol: str, dataset: pd.DataFrame, feature_cols: list[str], n_neighbors: int
    ) -> None:
        X = dataset[feature_cols]
        split = int(len(dataset) * 0.8)
        if split <= 20 or len(dataset) - split < 5:
            raise ValueError("Not enough data to train range model")

        X_train = X.iloc[:split]
        horizon_models: dict[str, Pipeline] = {}
        target_cols = [
            "upside_3d",
            "downside_3d",
            "upside_5d",
            "downside_5d",
            "upside_10d",
            "downside_10d",
        ]
        param_grid = {
            "knn__n_neighbors": sorted({n_neighbors, 3, 5, 7, 9}),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["minkowski", "manhattan"],
        }
        for col in target_cols:
            y_train = dataset[col].iloc[:split]
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsRegressor()),
                ]
            )
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=3,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            horizon_models[col] = search.best_estimator_

        payload = {"models": horizon_models, "features": feature_cols}
        joblib.dump(payload, self.range_model_path(symbol))

    def load_model(self, symbol: str) -> tuple[Pipeline, list[str]]:
        payload = joblib.load(self.model_path(symbol))
        return payload["model"], payload["features"]

    def load_range_models(self, symbol: str) -> tuple[dict[str, Pipeline], list[str]]:
        payload = joblib.load(self.range_model_path(symbol))
        return payload["models"], payload["features"]

    def predict_latest(self, symbol: str, dataset: pd.DataFrame) -> dict:
        model, features = self.load_model(symbol)
        latest = dataset.iloc[[-1]]
        proba_up = float(model.predict_proba(latest[features])[0][1])
        label = int(proba_up >= 0.5)
        return {"label": label, "prob_up": proba_up}

    def predict_latest_ranges(self, symbol: str, dataset: pd.DataFrame) -> dict[str, float]:
        models, features = self.load_range_models(symbol)
        latest = dataset.iloc[[-1]]
        output: dict[str, float] = {}
        for target, model in models.items():
            output[target] = float(model.predict(latest[features])[0])
        return output

    def load_metrics(self, symbol: str) -> dict[str, float]:
        return joblib.load(self.metrics_path(symbol))
