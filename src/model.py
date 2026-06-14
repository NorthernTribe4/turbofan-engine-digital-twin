"""Train, persist and load the RUL regressor."""

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config


def train_model(X, y, params=None):
    model = HistGradientBoostingRegressor(**(params or config.MODEL_PARAMS))
    model.fit(X, y)
    return model


def save_bundle(model, scaler, path=config.MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": config.FEATURE_COLUMNS},
        path,
    )


def load_bundle(path=config.MODEL_PATH):
    return joblib.load(path)
