"""Evaluation metrics: RMSE and the official C-MAPSS asymmetric score."""

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def cmapss_score(y_true, y_pred):
    # Late predictions (overestimating remaining life) are penalised
    # harder than early ones: the late branch grows faster.
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    return float(np.sum(penalty))
