"""Load the raw C-MAPSS text files into tidy DataFrames."""

import pandas as pd

from . import config


def load_raw(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=config.COLUMN_NAMES)


def load_train(subset="FD001"):
    return load_raw(config.DATA_RAW / f"train_{subset}.txt")


def load_test(subset="FD001"):
    return load_raw(config.DATA_RAW / f"test_{subset}.txt")


def load_rul(subset="FD001"):
    rul = pd.read_csv(
        config.DATA_RAW / f"RUL_{subset}.txt", header=None, names=["rul"]
    )
    rul.index += 1  # align index with 1-based unit numbers
    return rul["rul"]


def sensor_variation(df):
    """Per-sensor standard deviation, sorted. Used to confirm which
    columns are flat before dropping them."""
    return df[config.SENSOR_NAMES].std().sort_values()
