# Turbofan Engine Digital Twin: Technical Plan

A predictive-maintenance project built on NASA's C-MAPSS run-to-failure data. The
software reads sensor histories from engines that were run until they broke,
learns how they degrade, and estimates how many operating cycles each one has
left. The end product is a Plotly Dash dashboard where you pick an engine, watch
its sensors move, and see a live Remaining Useful Life (RUL) estimate with an
alert as failure gets close.

This document is the build spec. Anyone implementing it should be able to work
from here without asking follow-up questions. Where a number or a name matters,
it's written down.

---

## 1. Problem statement and what "digital twin" means here

Jet engines are expensive and they fail in ways that are costly and dangerous if
nobody sees them coming. Operators want to swap a part shortly before it wears
out, not on a fixed calendar and not after it has already failed. That means
estimating, at any moment, how much useful life a specific engine has left.

We frame this as a regression problem. Given the recent sensor history of one
engine, predict its Remaining Useful Life: the number of operating cycles until
failure.

"Digital twin" gets used loosely, so here is the concrete version for this
project. The twin is a software model of one physical engine that:

- takes the same time-series inputs the real engine produces (operational
  settings plus 21 sensor readings, one row per cycle),
- carries a fitted degradation model that maps recent sensor behaviour to a
  health estimate,
- outputs a live RUL number and a health index that update every cycle as new
  readings arrive,
- raises an alert when the estimate crosses a maintenance threshold.

It is not a physics simulation of combustion or airflow. It is a data-driven
mirror: feed it what the real engine is doing and it tells you how close that
engine is to the end of its life. That distinction matters for scoping. We are
building a learned estimator with a live interface, not a thermodynamic model.

---

## 2. The dataset

The data is NASA's C-MAPSS (Commercial Modular Aero-Propulsion System
Simulation) turbofan degradation set, originally from the 2008 PHM challenge. It
ships as plain-text files and is widely used as a benchmark, so results are easy
to sanity-check against published work.

### 2.1 The four subsets

The set is split into four problems, FD001 through FD004. They differ along two
axes: how many operating conditions the engines run under, and how many fault
modes are present.

| Subset | Train units | Test units | Operating conditions | Fault modes |
|--------|-------------|------------|----------------------|-------------|
| FD001  | 100         | 100        | 1 (sea level)        | 1 (HPC degradation) |
| FD002  | 260         | 259        | 6                    | 1 (HPC degradation) |
| FD003  | 100         | 100        | 1 (sea level)        | 2 (HPC degradation, fan degradation) |
| FD004  | 248         | 248        | 6                    | 2 (HPC degradation, fan degradation) |

HPC means high-pressure compressor. More operating conditions means the raw
sensor values swing around with flight regime rather than only with wear, so the
degradation signal is buried and you have to normalise per condition before it
shows up. More fault modes means the engines don't all degrade the same way, so
one simple model fits worse.

### 2.2 Row structure

Every file is whitespace-delimited with no header. Each row is one engine at one
cycle, and the columns are always in this order:

1. `unit_number`, which engine (resets per file)
2. `time_in_cycles`, operating cycle count for that engine, starting at 1
3. `op_setting_1`
4. `op_setting_2`
5. `op_setting_3`
6. `sensor_1` through `sensor_21`, 21 measurements

So 26 columns total: 2 identifiers, 3 operational settings, 21 sensors.

In the training files, each engine runs from healthy to failure, so the last row
for a unit is the cycle where it failed. In the test files, each engine's series
stops some time *before* failure, and a separate `RUL_FD00X.txt` file gives the
true remaining life at that final cycle. The test target is the RUL at the last
observed cycle of each test engine.

The 21 sensors map to physical measurements (fan inlet temperature, HPC outlet
pressure, core speed, bypass ratio, coolant bleed, and so on). For modelling we
care less about the physical name of each one and more about whether it carries a
usable signal in the subset we're working with.

### 2.3 Why start with FD001

FD001 is the simplest case: one operating condition, one fault mode. The sensor
trends are clean, you don't need per-condition normalisation, and the degradation
shows up directly in the raw readings. That makes it the right place to get the
full pipeline working end to end, load, label, feature-engineer, train,
evaluate, serve, before taking on the harder subsets. Once FD001 works, the same
code extends to FD002-FD004 by adding an operating-condition clustering and
normalisation step. The plan targets FD001 as the deliverable and notes the
extension as a stretch.

### 2.4 Dead sensors in FD001

Because FD001 runs under a single operating condition, several sensors never move
,  they sit at a constant value (or take two or three discrete values with no
trend) across every engine and every cycle. A constant column carries zero
information for a model and just adds noise and width, so we drop it.

In FD001 the following seven sensors are flat and get dropped:

- `sensor_1`
- `sensor_5`
- `sensor_6`
- `sensor_10`
- `sensor_16`
- `sensor_18`
- `sensor_19`

That leaves 14 informative sensors:

```
sensor_2,  sensor_3,  sensor_4,  sensor_7,  sensor_8,  sensor_9,
sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17,
sensor_20, sensor_21
```

The executor should not hard-trust this list blindly. The drop set is correct for
FD001, but the data loader must also compute per-column variance at load time and
log any sensor whose standard deviation is effectively zero, so the choice is
verifiable from the data and the same logic carries over when we move to other
subsets (where the dead set differs).

---

## 3. RUL labelling

The training files give you cycle counts but not RUL directly. You derive it: for
each engine, RUL at a given cycle is `max_cycle_for_that_engine - current_cycle`.
At the failure cycle RUL is 0; one cycle earlier it's 1, and so on.

### 3.1 Why raw linear RUL is a poor target

If you label RUL as a straight line running from the engine's full life down to
zero, you are telling the model that a brand-new engine at cycle 1 with 250
cycles left is in a meaningfully different state from one with 240 left. It is
not. Early in life the engine is healthy and the sensors look basically identical
whether failure is 250 or 200 cycles away. The degradation signal hasn't started
yet. Forcing the model to predict a high, precise RUL from readings that contain
no information about it teaches it to chase noise. It pays a large training
penalty for early-life rows it can never get right, which pulls capacity away
from the late-life rows that actually matter for a maintenance alert.

### 3.2 Piecewise-linear RUL with a cap

The fix is the standard one in the C-MAPSS literature: clip RUL at a ceiling.
Pick a cap `R_max`. For any cycle where the true remaining life is above the cap,
label it as the cap. Below the cap, label it linearly as the true remaining life.

```
RUL_label = min(max_cycle - current_cycle, R_max)
```

This gives a flat plateau at `R_max` for the healthy early phase, then a linear
ramp down to 0 over the final `R_max` cycles. It matches reality: "healthy, plenty
of life left" is one state, and only once wear sets in does the exact countdown
become learnable.

### 3.3 Why 125

The common choice, and the one this project uses, is `R_max = 125`. It's the
value used in much of the published C-MAPSS work, which makes our numbers
comparable to the benchmark. It also lands in a sensible place for FD001: the
shortest engines fail around 128 cycles and the longest run past 350, so 125 sits
near the low end of total lifetimes. That means even short-lived engines spend
most of their observed history on the informative ramp rather than the flat top,
while long-lived engines still get their pointless early cycles flattened out.
Set the cap as a single constant in config so it's easy to sweep later, but 125
is the default and the justification above is the reason.

---

## 4. Feature engineering

A single cycle's readings are noisy and don't show trend. Degradation is a
direction over time, not a level, so the features have to capture how each sensor
has been *moving* over the recent past.

### 4.1 Rolling-window statistics

For each engine, walk its cycles in order and compute, over a trailing window of
the last `N` cycles, three statistics per kept sensor:

- **mean**, the smoothed current level, denoised
- **std**, how much the sensor is jittering, which itself tends to rise near
  failure
- **slope**, the trend, fitted as the least-squares slope of the sensor against
  cycle index over the window (equivalently the coefficient of a degree-1 polyfit)

With 14 sensors that's 42 engineered features. The plan also keeps the raw
current value of each kept sensor and the current cycle number, since the cycle
count alone is weakly informative. That's a compact, fully numeric feature table
that any tree model handles directly with no scaling required.

Windows are computed strictly within a single engine. The rolling calculation
must reset at each `unit_number` boundary so one engine's tail never leaks into
the next engine's head.

### 4.2 Window size

Default `N = 30` cycles. It's long enough to average out sensor noise and get a
stable slope, and short enough to stay responsive when degradation accelerates
near the end. Like the cap, it lives in config as a single constant so it can be
tuned. 30 is the starting value and a reasonable one given FD001 lifetimes run
from roughly 128 to 360 cycles.

### 4.3 The cold-start problem

At the beginning of each engine's life the window isn't full, at cycle 5 there
are only 5 prior readings, not 30. Two rules handle this cleanly:

1. **Minimum periods.** Compute the rolling stats with a minimum of `min_periods`
   observations (default the smaller of 5 and `N`). For cycles with at least
   `min_periods` rows, compute over whatever is available so far. The mean and std
   are well-defined on a short window; the slope needs at least 2 points and is
   set to 0 when fewer are present.
2. **No row dropping in training, honest handling in serving.** We keep early
   rows rather than discarding them, because their RUL labels sit on the capped
   plateau anyway (RUL = 125) and the model only needs to learn "healthy" there,
   which a partial window supports fine. In the live dashboard the same partial-
   window logic runs, so the twin produces a prediction from cycle `min_periods`
   onward and shows a "warming up" state before that.

Pin the exact `min_periods` value in config and apply identical window logic in
training and serving. Any mismatch between the two is a silent bug, so they share
one function.

---

## 5. Model

### 5.1 Primary: HistGradientBoostingRegressor

The primary model is scikit-learn's `HistGradientBoostingRegressor`. Reasons:

- **No heavy dependencies.** It's in scikit-learn, which we already need. No
  PyTorch, no GPU, no CUDA setup on a Windows portfolio machine.
- **Strong on tabular data.** Gradient-boosted trees are the default winner on
  structured feature tables like ours, and the histogram variant trains fast on
  the tens of thousands of rows FD001 produces.
- **Handles the feature scales as-is.** Trees don't care that temperature and
  pressure live on different scales, so we skip a normalisation step that would
  otherwise be one more place to introduce a train/serve mismatch.
- **Explainable.** Feature importances come for free, and the model is simple
  enough to reason about, which matters for a portfolio piece where you want to
  explain *why* it predicts what it does.

`RandomForestRegressor` is a fine alternative with the same dependency profile and
is worth keeping as a one-line swap for comparison, but HistGBR is the primary
because it usually scores better here and trains faster.

Starting hyperparameters (set in config, tune later):

```
learning_rate     = 0.05
max_iter          = 500
max_leaf_nodes    = 31
min_samples_leaf  = 20
l2_regularization = 0.0
early_stopping    = True   (validation_fraction = 0.1)
random_state      = 42
```

### 5.2 Stretch goal: LSTM

An LSTM (or a small 1D-CNN) on the raw sensor sequences is the natural stretch.
The trade-off is real and worth stating plainly:

- **Upside:** it learns temporal structure directly from the sequence and can
  squeeze out a better C-MAPSS score than hand-built window features, which is
  what most top published results use.
- **Downside:** it pulls in a deep-learning framework, needs more care to train,
  is slower, harder to explain, and is overkill for getting a working twin on the
  board. It also raises the train/serve complexity because you have to feed it
  fixed-length padded sequences live.

Decision: **HistGBR is the primary and the shipping model.** The LSTM is documented
as an optional follow-up behind the same `model.py` interface (`train`, `predict`,
`save`, `load`) so it can be dropped in without touching the dashboard.

---

## 6. Evaluation

Two metrics, both reported. Held-out evaluation uses the official test split: each
test engine's last observed cycle, scored against the provided `RUL_FD001.txt`.

### 6.1 RMSE

Root mean squared error between predicted and true RUL, in cycles. Easy to read
and the standard headline number.

```
RMSE = sqrt( mean( (pred_i - true_i)^2 ) )
```

One note: predictions are evaluated against the capped label convention used in
the literature for a fair comparison, but we also report RMSE against the raw
uncapped test RUL so the number is honest about real-world error near end of life.

### 6.2 The C-MAPSS asymmetric score

The official challenge metric penalises late predictions more than early ones,
because predicting more life than the engine has is the dangerous mistake, you
miss the failure. Let `d_i = pred_i - true_i` (positive means the model
overestimated remaining life, i.e. predicted *late*).

```
        sum_i ( exp(-d_i / 13) - 1 )   for d_i <  0   (early prediction)
score =
        sum_i ( exp( d_i / 10) - 1 )   for d_i >= 0   (late prediction)
```

Because the late branch divides by 10 and the early branch by 13, a late error
grows faster than an early error of the same size, so the model is pushed toward
conservative (early) predictions. Lower score is better; a perfect model scores 0.
Report both the total score and RMSE side by side. Add a unit test that checks the
score on a tiny hand-computed example so the asymmetry can't silently break.

---

## 7. Dashboard

Plotly Dash, pure Python. No hand-written JavaScript. Layout in `layout.py`,
interactivity in `callbacks.py`, reusable figure/card builders in `components.py`,
app wiring in `dashboard.py`.

### 7.1 Layout

A single page, header plus two rows:

- **Header:** project title and an engine selector.
- **KPI row:** four cards.
- **Chart row:** sensor-stream chart on the left, RUL prediction-vs-reference
  chart on the right.

### 7.2 Components

**Engine selector.** A dropdown listing the held-out engines. Picking one resets
the stream to that engine's cycle 1.

**KPI cards (four).**

- *Current cycle*, where the playback head is in the selected engine's life.
- *Predicted RUL*, the twin's current estimate in cycles.
- *Health index*, a 0-100 figure derived from predicted RUL, `100 * min(pred,
  R_max) / R_max`. 100 is fresh, 0 is failure. Gives a quick at-a-glance state
  without reading the raw cycle number.
- *Alert status*, a coloured state: OK / Warning / Critical (see 7.4).

**Sensor-stream line chart.** Plots a handful of the most informative kept sensors
(for example `sensor_2`, `sensor_4`, `sensor_11`, `sensor_15`) over cycles up to
the current playback head, so the lines grow as playback advances. A vertical
marker shows the current cycle. Sensors are min-max scaled to a shared axis only
for display so different-scale lines sit together; the model still uses raw
values.

**RUL prediction-vs-reference chart.** Two lines over cycles: the twin's predicted
RUL at each cycle so far, and the ground-truth piecewise-linear RUL reference for
the same engine. Watching the prediction line track (or wobble around) the
reference is the core "is the twin working" view. The capped reference makes the
plateau-then-ramp shape visible.

### 7.3 Simulated live streaming

There's no real engine feeding data, so "live" is a replay of a held-out engine's
recorded history. A `dcc.Interval` component ticks on a timer (default once per
second). Each tick advances the playback head by one cycle. On each step the app:

1. takes the selected engine's rows up to the current cycle,
2. runs the same feature-engineering function used in training on that partial
   history,
3. feeds the latest feature row to the loaded model for a prediction,
4. updates the four KPI cards and both charts.

The cycle stepper and the held-out-engine bookkeeping live in `src/stream.py` so
the Dash callbacks stay thin. Play/pause is the Interval's `disabled` flag; a
slider or step button can scrub the head manually. State (selected engine, current
cycle) is held in a `dcc.Store` so callbacks stay stateless and there are no
globals.

The engines used for streaming must come from the held-out test split, never from
training data, so the dashboard demonstrates honest unseen-engine behaviour.

### 7.4 Alert threshold logic

Three states driven by predicted RUL against two constants in config:

```
predicted_RUL >  WARN_CYCLES (default 50)         -> OK        (green)
WARN_CYCLES >= predicted_RUL > CRIT_CYCLES (20)    -> Warning   (amber)
predicted_RUL <= CRIT_CYCLES (default 20)          -> Critical  (red)
```

To stop the alert flickering when a noisy prediction bounces across a threshold,
the state is driven by the rolling mean of the last few predictions, not a single
raw value. The smoothing window is a config constant (default 5). The alert card
and the KPI colours both read from this one computed state.

---

## 8. Repository layout

```
turbofan-engine-digital-twin/
├── PLAN.md                     # this document
├── README.md                   # what it is, setup, run, screenshots
├── LICENSE                     # MIT
├── requirements.txt            # pinned dependencies
├── .gitignore                  # venv, data, models, caches, pyc
├── pyproject.toml              # tool config (black, isort, pytest) and metadata
│
├── data/
│   ├── raw/                    # downloaded C-MAPSS txt files (gitignored)
│   └── processed/              # cached feature tables, parquet (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # all constants: paths, sensor lists, window, cap, thresholds, model params
│   ├── data_loader.py          # read raw txt into typed DataFrames, flag dead sensors
│   ├── labels.py               # piecewise-linear capped RUL
│   ├── features.py             # rolling mean/std/slope window builder (shared by train and serve)
│   ├── model.py                # train / predict / save / load wrapper around HistGBR
│   ├── evaluate.py             # RMSE and C-MAPSS asymmetric score
│   └── stream.py               # held-out engine cycle stepper for the dashboard
│
├── app/
│   ├── __init__.py
│   ├── dashboard.py            # Dash app object and server entrypoint
│   ├── layout.py               # page layout, selector, card and chart containers
│   ├── callbacks.py            # interval tick, selector change, store updates
│   └── components.py           # KPI card and figure builder functions
│
├── scripts/
│   ├── prepare_data.py         # check raw files present, validate shape, report dead sensors
│   └── train.py                # build features, train model, evaluate, save artifact
│
├── models/
│   └── .gitkeep                # trained model lands here (artifact gitignored)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py     # column count, dtypes, dead-sensor detection
│   ├── test_labels.py          # cap behaviour, boundary cycles, per-unit reset
│   ├── test_features.py        # window math, cold-start, no cross-engine leakage
│   └── test_evaluate.py        # RMSE and asymmetric score on hand-computed cases
│
└── docs/
    └── data_download.md        # where to get the C-MAPSS files and where to put them
```

Every Python module stays under the size limits in the house style (small,
focused files). If `callbacks.py` or `features.py` grows past comfort, split by
concern rather than letting one file balloon.

---

## 9. Dependencies

Pinned to recent stable versions. Pins use `~=` so patch updates are allowed but
majors are held. Confirm against the current PyPI releases at build time and
adjust if a pin no longer resolves on the target Python.

```
# requirements.txt
numpy~=2.1
pandas~=2.2
scikit-learn~=1.5
plotly~=5.24
dash~=2.18
joblib~=1.4
pyarrow~=17.0
```

```
# dev (can live in requirements-dev.txt or an extras group)
pytest~=8.3
black~=24.8
isort~=5.13
```

Rationale, one line each:

- **numpy**, array math under everything; pandas and scikit-learn need it anyway.
- **pandas**, load the text files, group by engine, compute rolling windows.
- **scikit-learn**, the model (`HistGradientBoostingRegressor`) and metrics.
- **plotly**, the figures rendered in the dashboard.
- **dash**, the web dashboard framework, pure-Python interactivity.
- **joblib**, persist and reload the trained model artifact.
- **pyarrow**, fast parquet read/write for the cached feature table; optional but
  cheap and it keeps processed-data caching painless.
- **pytest**, the test runner.
- **black, isort**, formatting and import order, so the code stays PEP 8 without
  hand-fussing.

Target Python 3.11 or 3.12. State the chosen version in the README.

---

## 10. Build order

Numbered milestones from empty folder to a working dashboard. Each one ends in
something runnable or testable.

1. **Repo skeleton.** Create the folder tree, `git init` locally (no remote, no
   push, no CI). Add `README.md`, `LICENSE` (MIT), `.gitignore`, `requirements.txt`,
   `pyproject.toml`, and empty `__init__.py` files. Create and activate the venv,
   install dependencies.
2. **Data acquisition guide and config.** Write `docs/data_download.md` and
   `src/config.py` with every constant from this plan (paths, sensor lists, window
   = 30, cap = 125, thresholds, model params). Nothing in the codebase hard-codes
   these elsewhere.
3. **Data loader.** `src/data_loader.py`: read the raw txt files into DataFrames
   with the right column names and dtypes, attach the per-engine max cycle,
   compute per-column variance, and log dead sensors. `scripts/prepare_data.py`
   validates the raw files are present and shaped right. Tests pass.
4. **Labels.** `src/labels.py`: piecewise-linear capped RUL, reset per engine.
   Tests cover the cap, the boundary cycles, and the per-unit reset.
5. **Features.** `src/features.py`: rolling mean/std/slope with the cold-start
   rules, computed strictly within each engine. Tests prove the window math and
   prove no value crosses an engine boundary.
6. **Model.** `src/model.py`: train/predict/save/load around HistGBR. Run a first
   training pass via a temporary entrypoint to confirm it fits and produces sane
   predictions.
7. **Evaluation.** `src/evaluate.py`: RMSE and the asymmetric score, with the
   hand-computed unit test. Wire `scripts/train.py` to build features, train,
   evaluate on the held-out test split, print RMSE and score, and save the model
   to `models/`.
8. **Tune to a baseline.** Sweep `max_iter`, `learning_rate`, window size, and the
   cap a little. Record the FD001 RMSE and C-MAPSS score in the README so there's a
   number to point at.
9. **Dashboard shell.** `app/`: Dash app with the engine selector, four empty KPI
   cards, and two empty charts laid out. It runs and renders.
10. **Wire the stream.** `src/stream.py` plus the callbacks: Interval tick advances
    the cycle, features recompute on the partial history, the model predicts, KPI
    cards and both charts update. Alert state with smoothing.
11. **Polish and document.** Tidy the styling, add play/pause and a scrub control,
    finish the README with setup, run commands, and a screenshot. Final test run,
    final formatting pass.

PowerShell commands for setup and run (Windows target):

```powershell
python -m venv venv ; venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
python scripts\prepare_data.py
python scripts\train.py
python -m app.dashboard
```

---

## 11. Risks and edge cases

**Data leakage between train and test windows.** The rolling features must never
let one engine's data bleed into another's, and the test engines must never appear
in training. Two concrete guards: compute every window grouped by `unit_number` so
the calculation resets at each engine, and keep the official train/test split
intact, fit only on train units, evaluate only on the test units' final cycles.
If you add a validation split for early stopping, split by engine, not by row, or
rows from the same engine end up on both sides and the validation score lies.

**Engines shorter than the window.** A few engines fail early, FD001's shortest
is around 128 cycles, which is fine, but the cold-start logic still has to hold for
the first `N` cycles of every engine. The `min_periods` rule covers it: stats are
defined on partial windows, slope falls back to 0 below 2 points. Test explicitly
with a synthetic engine shorter than the window so the code can't divide by zero or
emit NaNs. The dashboard shows a "warming up" state until `min_periods` cycles
exist rather than rendering a garbage prediction.

**Prediction instability near end of life.** Right before failure the sensors move
fast and the per-cycle prediction can jump around. Two mitigations already in the
plan: the cap keeps the target well-behaved, and the alert state reads a rolling
mean of recent predictions instead of the raw latest value so it doesn't flicker.
Expect and accept some noise in the raw predicted-RUL line near cycle-zero; that's
honest model behaviour, and the smoothed alert is what drives the maintenance
decision.

**Train/serve feature mismatch.** The single biggest silent-bug risk. The
dashboard must call the exact same feature function as training, with the same
window and `min_periods`. Keep one implementation in `src/features.py` and import
it in both places. Never reimplement the window math in the callback.

**Missing or malformed raw data.** The files aren't shipped in the repo (they're
gitignored). `prepare_data.py` must fail loudly with a clear message pointing at
`docs/data_download.md` if the files are absent or the column count is wrong,
rather than letting a downstream step crash with a confusing error.

**Distribution shift to FD002-FD004.** The FD001 model will not transfer to the
multi-condition subsets without a per-condition normalisation step, because the
raw sensor values there move with flight regime, not just wear. That's expected
and out of scope for the FD001 deliverable. Note it in the README so nobody points
the trained model at FD004 and reports it as broken.

---

## House rules for the executor

- Filenames carry no vendor or assistant name. No authorship attribution to any
  tool, vendor, or model anywhere in code, comments, docs, or commit messages. No
  tool-generated footers and no co-authorship trailers in commits.
- All prose (README, the data guide, this plan) reads in a plain human voice.
  Avoid em-dash overuse, three-item lists for their own sake, and filler words like
  delve, robust, seamless, leverage. Don't over-hedge.
- Code is idiomatic PEP 8 Python. Comments are brief and section-level, flag what a
  developer would actually flag, not every line. No bloated machine-sounding module
  docstrings. No emoji.
- Windows target. All shell commands in docs use PowerShell style, e.g.
  `python -m venv venv ; venv\Scripts\Activate.ps1 ; pip install -r requirements.txt`.
- Python only. Dashboard is Plotly Dash. Model is scikit-learn.
- Standard repo files present: README.md, .gitignore, LICENSE (MIT),
  requirements.txt. Run `git init` locally only. Do not create a remote, do not
  push, do not add CI or GitHub Actions. The human handles pushing.
- Default repo name is `turbofan-engine-digital-twin`; the human may rename.

