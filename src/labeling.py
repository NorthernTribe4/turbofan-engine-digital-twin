"""Piecewise-linear RUL labels with a cap."""

from . import config


def add_rul(df, cap=config.RUL_CAP):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return out
