"""Project configuration: paths, column layout, and tuned constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODELS_DIR / "rul_model.joblib"

# Raw C-MAPSS files are whitespace-delimited with no header row.
INDEX_NAMES = ["unit", "cycle"]
SETTING_NAMES = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# These sensors stay constant under FD001's single operating condition,
# so they carry no signal and are dropped. Confirmed against the data
# by data_loader.sensor_variation().
DROP_SENSORS = [f"sensor_{i}" for i in (1, 5, 6, 10, 16, 18, 19)]
KEPT_SENSORS = [s for s in SENSOR_NAMES if s not in DROP_SENSORS]

# One feature block per kept sensor: smoothed level, spread, and trend.
FEATURE_COLUMNS = [
    f"{s}_{stat}" for s in KEPT_SENSORS for stat in ("mean", "std", "slope")
]

# RUL is clipped to a plateau early in life where degradation has not started.
RUL_CAP = 125

# Rolling window over each engine's recent cycles.
ROLL_WINDOW = 30
ROLL_MIN_PERIODS = 5

# Alert thresholds on predicted RUL, in cycles.
WARN_RUL = 50
CRIT_RUL = 20
ALERT_SMOOTHING = 5

# Sensors drawn in the dashboard stream chart.
DISPLAY_SENSORS = ["sensor_2", "sensor_4", "sensor_11", "sensor_15"]

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
}
