"""Plotly Dash dashboard for the turbofan digital twin.

Pick a held-out engine and the app replays its recorded cycles one at a
time, recomputing the model's RUL prediction as new readings arrive and
raising an alert as the engine nears failure.

Run:  python -m dashboard.app
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from src import config
from src.data_loader import load_rul, load_test
from src.model import load_bundle
from src.serve import alert_state, engine_timeline, smoothed_rul

# Load the model and data once at startup.
BUNDLE = load_bundle()
MODEL = BUNDLE["model"]
SCALER = BUNDLE["scaler"]
TEST_DF = load_test()
RUL_TRUTH = load_rul()
UNITS = sorted(TEST_DF["unit"].unique().tolist())

TICK_MS = 1000

ALERT_COLORS = {
    "OK": "#2e7d32",
    "WARNING": "#f9a825",
    "CRITICAL": "#c62828",
    "WARMING UP": "#555f6b",
}

# Per-engine timelines are cached so the interval tick stays cheap.
_TIMELINES = {}


def get_timeline(unit):
    if unit not in _TIMELINES:
        _TIMELINES[unit] = engine_timeline(
            TEST_DF, SCALER, MODEL, unit, RUL_TRUTH.loc[unit]
        )
    return _TIMELINES[unit]


# --- styling helpers -------------------------------------------------------

CARD_STYLE = {
    "flex": "1",
    "padding": "18px",
    "margin": "8px",
    "borderRadius": "10px",
    "background": "#1e2630",
    "color": "#e6edf3",
    "textAlign": "center",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.4)",
}

LABEL_STYLE = {"fontSize": "13px", "color": "#9aa7b4", "letterSpacing": "0.5px"}
VALUE_STYLE = {"fontSize": "30px", "fontWeight": "600", "marginTop": "6px"}


def kpi_card(label, value_id, value_style=None):
    style = dict(VALUE_STYLE, **(value_style or {}))
    return html.Div(
        [html.Div(label, style=LABEL_STYLE), html.Div(id=value_id, style=style)],
        style=CARD_STYLE,
    )


# --- layout ----------------------------------------------------------------

app = Dash(__name__)
app.title = "Turbofan Digital Twin"

app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "background": "#0d1117",
        "minHeight": "100vh",
        "padding": "24px",
    },
    children=[
        html.H1(
            "Turbofan Engine Digital Twin",
            style={"color": "#e6edf3", "marginBottom": "4px"},
        ),
        html.Div(
            "Live remaining-useful-life estimate from streaming sensor data (FD001).",
            style={"color": "#9aa7b4", "marginBottom": "20px"},
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px"},
            children=[
                html.Span("Engine", style={"color": "#9aa7b4"}),
                dcc.Dropdown(
                    id="engine-select",
                    options=[{"label": f"Engine {u}", "value": u} for u in UNITS],
                    value=UNITS[0],
                    clearable=False,
                    style={"width": "200px"},
                ),
                html.Button(
                    "Pause",
                    id="play-button",
                    n_clicks=0,
                    style={
                        "padding": "8px 18px",
                        "borderRadius": "8px",
                        "border": "none",
                        "background": "#2563eb",
                        "color": "white",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "marginTop": "20px"},
            children=[
                kpi_card("HEALTH INDEX", "kpi-health"),
                kpi_card("PREDICTED RUL", "kpi-rul"),
                kpi_card("CURRENT CYCLE", "kpi-cycle"),
                kpi_card("ALERT STATUS", "kpi-alert"),
            ],
        ),
        html.Div(
            style={"display": "flex", "marginTop": "8px"},
            children=[
                dcc.Graph(id="sensor-chart", style={"flex": "1"}),
                dcc.Graph(id="rul-chart", style={"flex": "1"}),
            ],
        ),
        dcc.Interval(id="tick", interval=TICK_MS, n_intervals=0),
        dcc.Store(id="unit-store", data=UNITS[0]),
        dcc.Store(id="step-store", data=0),
    ],
)


# --- callbacks -------------------------------------------------------------


@app.callback(
    Output("unit-store", "data"),
    Output("step-store", "data", allow_duplicate=True),
    Input("engine-select", "value"),
    prevent_initial_call=True,
)
def select_engine(unit):
    get_timeline(unit)
    return unit, 0


@app.callback(
    Output("step-store", "data"),
    Input("tick", "n_intervals"),
    State("step-store", "data"),
    State("unit-store", "data"),
)
def advance(_, step, unit):
    tl = get_timeline(unit)
    return min(step + 1, len(tl["cycles"]) - 1)


@app.callback(
    Output("tick", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
)
def toggle_play(n_clicks):
    paused = n_clicks % 2 == 1
    return paused, "Play" if paused else "Pause"


@app.callback(
    Output("kpi-health", "children"),
    Output("kpi-rul", "children"),
    Output("kpi-cycle", "children"),
    Output("kpi-alert", "children"),
    Output("kpi-alert", "style"),
    Output("sensor-chart", "figure"),
    Output("rul-chart", "figure"),
    Input("step-store", "data"),
    State("unit-store", "data"),
)
def render(step, unit):
    tl = get_timeline(unit)
    step = min(step, len(tl["cycles"]) - 1)

    cycle = int(tl["cycles"][step])
    rul = smoothed_rul(tl["pred"], step)
    state = alert_state(rul)

    health = "--" if rul is None else f"{100 * min(rul, config.RUL_CAP) / config.RUL_CAP:.0f}"
    rul_text = "warming up" if rul is None else f"{rul:.0f}"

    alert_style = dict(
        VALUE_STYLE, color="white", background=ALERT_COLORS[state],
        borderRadius="8px", padding="4px",
    )

    return (
        health,
        rul_text,
        str(cycle),
        state,
        alert_style,
        sensor_figure(tl, step),
        rul_figure(tl, step),
    )


def sensor_figure(tl, step):
    raw = tl["raw"]
    cycles = tl["cycles"][: step + 1]
    fig = go.Figure()
    for sensor in config.DISPLAY_SENSORS:
        values = raw[sensor].to_numpy(dtype=float)
        lo, hi = values.min(), values.max()
        norm = (values - lo) / (hi - lo) if hi > lo else values * 0
        fig.add_trace(
            go.Scatter(x=cycles, y=norm[: step + 1], mode="lines", name=sensor)
        )
    fig.update_layout(
        title="Sensor stream (normalised)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Cycle",
        yaxis_title="Normalised reading",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def rul_figure(tl, step):
    cycles = tl["cycles"][: step + 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles, y=tl["reference"][: step + 1], mode="lines",
            name="Reference", line=dict(color="#58a6ff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=tl["pred"][: step + 1], mode="lines",
            name="Predicted", line=dict(color="#f78166", dash="dash"),
        )
    )
    fig.add_hline(
        y=config.WARN_RUL, line_dash="dot", line_color="#f9a825",
        annotation_text="warn",
    )
    fig.update_layout(
        title="Remaining useful life",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Cycle",
        yaxis_title="RUL (cycles)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False)
