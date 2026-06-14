"""Sensor normalisation.

Trees do not need scaled inputs, but normalising keeps the feature
values on a comparable range and makes the stream chart readable.
The scaler is fit on the training engines only and reused everywhere.
"""

from sklearn.preprocessing import StandardScaler

from . import config


def fit_scaler(df):
    scaler = StandardScaler()
    scaler.fit(df[config.KEPT_SENSORS])
    return scaler


def apply_scaler(df, scaler):
    out = df.copy()
    out[config.KEPT_SENSORS] = scaler.transform(df[config.KEPT_SENSORS])
    return out
