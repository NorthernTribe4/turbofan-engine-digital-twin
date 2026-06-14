"""Turn a held-out engine into a per-cycle timeline.

Shared by the dashboard and the result figures so both use the exact
same feature and prediction path the model was trained on.
"""

import numpy as np
import pandas as pd

from . import config
from .features import build_features
from .preprocessing import apply_scaler


def engine_timeline(test_df, scaler, model, unit, true_rul):
    eng_raw = test_df[test_df["unit"] == unit].reset_index(drop=True)
    eng_scaled = apply_scaler(eng_raw, scaler)
    feats = build_features(eng_scaled)

    X = feats[config.FEATURE_COLUMNS]
    valid = X.dropna()
    preds = pd.Series(np.nan, index=feats.index)
    if not valid.empty:
        preds.loc[valid.index] = model.predict(valid)

    # Reference RUL: we know the true value at the last observed cycle,
    # so walk it back up the engine's history and cap it.
    last_cycle = eng_raw["cycle"].max()
    reference = (true_rul + (last_cycle - eng_raw["cycle"])).clip(upper=config.RUL_CAP)

    return {
        "unit": int(unit),
        "cycles": eng_raw["cycle"].to_numpy(),
        "raw": eng_raw,
        "pred": preds.to_numpy(),
        "reference": reference.to_numpy(),
        "true_rul": float(true_rul),
    }


def smoothed_rul(preds, step, window=config.ALERT_SMOOTHING):
    """Mean of the last few valid predictions up to the current step.
    Keeps the alert from flickering on a single noisy prediction."""
    history = preds[: step + 1]
    history = history[~np.isnan(history)]
    if history.size == 0:
        return None
    return float(np.mean(history[-window:]))


def alert_state(rul):
    if rul is None:
        return "WARMING UP"
    if rul <= config.CRIT_RUL:
        return "CRITICAL"
    if rul <= config.WARN_RUL:
        return "WARNING"
    return "OK"
