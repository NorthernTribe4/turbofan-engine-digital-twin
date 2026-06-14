# Turbofan Engine Digital Twin

This project estimates how many flights a jet engine has left before it fails.
It uses NASA's C-MAPSS dataset, which records sensor readings from simulated
turbofan engines run from healthy all the way to breakdown. The model reads an
engine's recent sensor history and predicts its Remaining Useful Life (RUL): the
number of operating cycles left before failure. A Plotly Dash dashboard replays
a held-out engine cycle by cycle so you can watch the sensors drift and the RUL
estimate fall, with an alert when the engine nears the end of its life.

I built it as a portfolio project to work through the full path from raw
run-to-failure data to a working predictive-maintenance tool.

## What it does

- Loads the FD001 subset of C-MAPSS and drops the sensors that never move under
  its single operating condition.
- Labels each cycle with a capped RUL so the model is not asked to predict an
  exact countdown while the engine is still healthy.
- Builds rolling-window features (mean, spread and trend over the last 30 cycles)
  for each sensor that carries a signal.
- Trains a gradient-boosted regression model and scores it with RMSE and the
  official C-MAPSS scoring function.
- Serves a dashboard with an engine picker, health and RUL cards, a live sensor
  chart, and a predicted-vs-reference RUL chart.

## Setup

Windows, PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You need the FD001 data files before training. See [data/README.md](data/README.md)
for where to get them and where to put them.

## Train the model

```powershell
python train.py
```

This fits the model, prints the test metrics, saves the model under `models\`,
and writes the figures and metrics to `results\`.

## Run the dashboard

```powershell
python -m dashboard.app
```

Open http://127.0.0.1:8050 in a browser. Pick an engine from the dropdown and it
replays one cycle per second. The Pause button stops and starts the playback.

## Dataset and method

C-MAPSS has four subsets. This project uses FD001: 100 training engines and 100
test engines, all running under one operating condition with a single fault mode
(high-pressure compressor wear). The training engines run to failure; the test
engines stop partway through, and a separate file gives the true RUL at that
point.

Each row is one engine at one cycle: a unit number, the cycle count, three
operational settings, and 21 sensor measurements. Under FD001's single condition,
seven of those sensors sit at a constant value and get dropped (sensors 1, 5, 6,
10, 16, 18, 19), which leaves 14.

RUL is capped at 125 cycles. Early in an engine's life the sensors look about the
same whether failure is 200 or 250 cycles away, so a straight-line label would
push the model to guess at something the data does not show yet. Clipping the
label to a plateau lets it focus on the stretch of life where wear actually
shows up.

The model is scikit-learn's `HistGradientBoostingRegressor`. It handles the
tabular features well, trains in a few seconds, needs no GPU, and is easy to
reason about for a project meant to be read and understood.

## Results

On the 100 held-out FD001 test engines:

| Metric | Value |
|--------|-------|
| RMSE | 14.8 cycles |
| RMSE (true RUL capped at 125) | 13.6 cycles |
| C-MAPSS score | 319 |

The C-MAPSS score weighs late predictions (claiming more life than the engine
has) more heavily than early ones, since a missed failure is the expensive
mistake.

Training writes two figures to `results\`:

- `sensor_degradation.png` shows several sensors drifting over one engine's life.
- `rul_prediction.png` shows the predicted RUL tracking the reference curve for a
  test engine.

## Project layout

```
src/            data loading, labelling, features, model, evaluation
dashboard/      Plotly Dash app
data/raw/       dataset files (not committed)
models/         saved model (not committed)
results/        metrics and figures
tests/          unit tests for the pipeline
train.py        train, evaluate, and write results
```

## Notes

The model is trained only on FD001. The other three subsets add multiple
operating conditions and a second fault mode, which need a per-condition
normalisation step before the same approach holds up. That is the natural next
step rather than something this version covers.
