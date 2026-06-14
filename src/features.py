"""Rolling-window features.

For each kept sensor we summarise the last N cycles of an engine with
three numbers: the mean (smoothed level), the standard deviation
(jitter, which tends to rise near failure) and the slope (trend).
Windows are computed per engine so one unit never leaks into the next.
"""

import numpy as np
import pandas as pd

from . import config


def _slope(values):
    # Least-squares slope of the sensor against cycle index.
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]


def build_features(df, window=config.ROLL_WINDOW, min_periods=config.ROLL_MIN_PERIODS):
    grouped = df.groupby("unit", sort=False)
    series = []
    for sensor in config.KEPT_SENSORS:
        roll = grouped[sensor].rolling(window, min_periods=min_periods)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        slope = roll.apply(_slope, raw=True).reset_index(level=0, drop=True)
        series.append(mean.rename(f"{sensor}_mean"))
        series.append(std.rename(f"{sensor}_std"))
        series.append(slope.rename(f"{sensor}_slope"))

    feats = pd.concat(series, axis=1).sort_index()
    keep = ["unit", "cycle"] + (["rul"] if "rul" in df.columns else [])
    return pd.concat([df[keep], feats], axis=1)
