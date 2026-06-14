"""Train the RUL model on FD001 and write metrics and figures to results/.

Run:  python train.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_loader import load_test, load_train, load_rul, sensor_variation
from src.evaluate import cmapss_score, rmse
from src.features import build_features
from src.labeling import add_rul
from src.model import save_bundle, train_model
from src.preprocessing import apply_scaler, fit_scaler
from src.serve import engine_timeline


def build_training_set(train_df, scaler):
    labelled = add_rul(train_df)
    scaled = apply_scaler(labelled, scaler)
    feats = build_features(scaled)
    feats = feats.dropna(subset=config.FEATURE_COLUMNS)
    return feats[config.FEATURE_COLUMNS], feats["rul"]


def evaluate_on_test(test_df, rul_truth, scaler, model):
    scaled = apply_scaler(test_df, scaler)
    feats = build_features(scaled)
    last = feats.groupby("unit").tail(1)
    preds = model.predict(last[config.FEATURE_COLUMNS])

    true = rul_truth.loc[last["unit"].to_numpy()].to_numpy()
    true_capped = true.clip(max=config.RUL_CAP)
    return {
        "rmse": rmse(true, preds),
        "rmse_capped": rmse(true_capped, preds),
        "cmapss_score": cmapss_score(true, preds),
        "n_test_engines": int(len(preds)),
    }


def plot_sensor_degradation(train_df, unit=1):
    eng = train_df[train_df["unit"] == unit]
    fig, ax = plt.subplots(figsize=(9, 5))
    for sensor in config.DISPLAY_SENSORS:
        values = eng[sensor].to_numpy(dtype=float)
        lo, hi = values.min(), values.max()
        norm = (values - lo) / (hi - lo) if hi > lo else values * 0
        ax.plot(eng["cycle"], norm, label=sensor)
    ax.set_title(f"Sensor drift over the life of engine {unit} (FD001 train)")
    ax.set_xlabel("Operating cycle")
    ax.set_ylabel("Normalised reading")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "sensor_degradation.png", dpi=120)
    plt.close(fig)


def plot_rul_prediction(test_df, rul_truth, scaler, model, unit):
    tl = engine_timeline(test_df, scaler, model, unit, rul_truth.loc[unit])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tl["cycles"], tl["reference"], label="Reference RUL", linewidth=2)
    ax.plot(tl["cycles"], tl["pred"], label="Predicted RUL", linestyle="--")
    ax.set_title(f"Predicted vs reference RUL, test engine {unit}")
    ax.set_xlabel("Operating cycle")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "rul_prediction.png", dpi=120)
    plt.close(fig)


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    test_df = load_test()
    rul_truth = load_rul()

    # Sanity check: confirm the flat sensors really are flat before relying on it.
    variation = sensor_variation(train_df)
    flat = variation[variation < 1e-2].index.tolist()
    print("Flat sensors detected:", flat)
    print("Dropping:", config.DROP_SENSORS)

    scaler = fit_scaler(train_df)
    X, y = build_training_set(train_df, scaler)
    print(f"Training rows: {len(X)}  features: {X.shape[1]}")

    model = train_model(X, y)
    save_bundle(model, scaler)
    print(f"Saved model to {config.MODEL_PATH}")

    metrics = evaluate_on_test(test_df, rul_truth, scaler, model)
    print("Test metrics:", json.dumps(metrics, indent=2))
    with open(config.RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    plot_sensor_degradation(train_df, unit=1)
    plot_rul_prediction(test_df, rul_truth, scaler, model, unit=24)
    print(f"Wrote figures and metrics to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
