import numpy as np
import pandas as pd

from src import config
from src.evaluate import cmapss_score, rmse
from src.features import build_features
from src.labeling import add_rul


def _toy_frame(n_cycles=40, unit=1):
    cycles = np.arange(1, n_cycles + 1)
    data = {"unit": unit, "cycle": cycles}
    for s in config.SENSOR_NAMES:
        data[s] = np.linspace(0, 1, n_cycles) + 0.01 * np.random.randn(n_cycles)
    return pd.DataFrame(data)


def test_rul_is_capped():
    df = _toy_frame(n_cycles=200)
    labelled = add_rul(df)
    assert labelled["rul"].max() == config.RUL_CAP
    assert labelled["rul"].iloc[-1] == 0


def test_rul_resets_per_engine():
    df = pd.concat([_toy_frame(30, unit=1), _toy_frame(50, unit=2)], ignore_index=True)
    labelled = add_rul(df)
    last_u1 = labelled[labelled["unit"] == 1]["rul"].iloc[-1]
    last_u2 = labelled[labelled["unit"] == 2]["rul"].iloc[-1]
    assert last_u1 == 0 and last_u2 == 0


def test_features_no_cross_engine_leak():
    # Engine 2's features must be identical whether or not engine 1
    # sits in front of it in the same frame.
    e2 = _toy_frame(30, unit=2)
    combined = pd.concat([_toy_frame(30, unit=1), e2], ignore_index=True)
    feats_combined = build_features(combined)
    feats_alone = build_features(e2.copy())

    col = f"{config.KEPT_SENSORS[0]}_mean"
    a = feats_combined[feats_combined["unit"] == 2][col].reset_index(drop=True)
    b = feats_alone[col].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_feature_columns_present():
    df = _toy_frame(40)
    feats = build_features(df)
    for col in config.FEATURE_COLUMNS:
        assert col in feats.columns


def test_rmse_zero_on_match():
    y = [10, 20, 30]
    assert rmse(y, y) == 0.0


def test_cmapss_penalises_late_more():
    # A late prediction (too much life) should cost more than an early one.
    late = cmapss_score([50], [60])
    early = cmapss_score([50], [40])
    assert late > early
