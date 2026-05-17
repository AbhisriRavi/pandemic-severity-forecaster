"""
Pandemic Severity Forecaster — Streamlit dashboard.

Run with:
    streamlit run app.py

Three tabs:
1. **Forecast**: pick a country, see actual + predicted next 14 days with intervals.
2. **Compare**: pick countries, compare attendance/case rates over time.
3. **Insights**: continent-level breakdown and model performance.

This dashboard reads from data/processed/ and reports/forecasts/. If those
don't exist, it shows clear setup instructions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Pandemic Severity Forecaster",
    page_icon="🦠",
    layout="wide",
)

PROCESSED = Path("data/processed")
FORECASTS = Path("reports/forecasts")

# -----------------------------------------------------------------------------
# Data loading (cached)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_features():
    path = PROCESSED / "features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_snapshots():
    path = PROCESSED / "snapshots.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_predictions():
    out = {}
    for f in FORECASTS.glob("*_predictions.parquet"):
        model_name = f.stem.replace("_predictions", "")
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        out[model_name] = df
    return out


@st.cache_data(show_spinner=False)
def load_metrics():
    import json
    path = FORECASTS / "summary_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# -----------------------------------------------------------------------------
# Setup-time guard
# -----------------------------------------------------------------------------

features = load_features()
snapshots = load_snapshots()
predictions = load_predictions()
metrics = load_metrics()

st.title("Pandemic Severity Forecaster")
st.caption(
    "Live epidemiological data from [disease.sh](https://disease.sh/). "
    "Built as a portfolio project for the LIDA Data Scientist Programme."
)

if features is None or snapshots is None:
    st.error("Setup not complete — data files missing.")
    st.markdown("""
    Run these commands first:
    ```bash
    python -m src.data.fetch
    python -m src.features.build
    python -m src.models.train --models all
    ```
    """)
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This dashboard forecasts COVID-19 case counts 14 days ahead "
        "for 100+ countries using three models (LightGBM, Prophet, LSTM) "
        "and audits performance by continent."
    )
    st.markdown(
        f"**Data range:** {features['date'].min():%b %Y} → "
        f"{features['date'].max():%b %Y}"
    )
    st.markdown(f"**Countries:** {features['country'].nunique()}")
    st.markdown("---")
    st.caption("Code: [GitHub](#) · API: [disease.sh](https://disease.sh/)")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_forecast, tab_compare, tab_insights = st.tabs(["Forecast", "Compare", "Insights"])


# ----------------------------------- Tab 1: Forecast -------------------------

with tab_forecast:
    st.subheader("Forecast for a single country")
    col_l, col_r = st.columns([1, 3])

    with col_l:
        country = st.selectbox(
            "Country",
            sorted(features["country"].unique()),
            index=0,
        )
        model_choice = st.radio(
            "Model",
            sorted(predictions.keys()) if predictions else ["(no predictions yet)"],
        )
        show_intervals = st.checkbox("Show 80% prediction interval", value=True)

    sub = features[features["country"] == country].sort_values("date")
    snap_row = snapshots[snapshots["country"] == country]
    population = int(snap_row["population"].iloc[0]) if not snap_row.empty else None

    with col_r:
        if not sub.empty:
            cols = st.columns(3)
            cols[0].metric("Population", f"{population:,}" if population else "—")
            cols[1].metric(
                "Latest 7-day mean new cases",
                f"{int(sub['new_cases_smoothed'].iloc[-1]):,}",
            )
            cols[2].metric(
                "Estimated R-effective",
                f"{sub['r_effective_approx'].iloc[-1]:.2f}",
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["new_cases_smoothed"],
        mode="lines", name="Actual (7-day smoothed)",
        line=dict(color="black", width=2),
    ))

    if predictions and model_choice in predictions:
        p = predictions[model_choice]
        p_country = p[p["country"] == country].sort_values("date")
        if not p_country.empty:
            fig.add_trace(go.Scatter(
                x=p_country["date"], y=p_country["y_pred"],
                mode="lines+markers", name=f"{model_choice} forecast",
                line=dict(color="royalblue", dash="dash"),
            ))
            if show_intervals and "y_pred_q10" in p_country.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([p_country["date"], p_country["date"][::-1]]),
                    y=pd.concat([p_country["y_pred_q90"], p_country["y_pred_q10"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(65,105,225,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% interval",
                    showlegend=True,
                ))

    fig.update_layout(
        title=f"{country}: actual vs forecast",
        xaxis_title="", yaxis_title="New cases (7-day mean)",
        hovermode="x unified", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------- Tab 2: Compare --------------------------

with tab_compare:
    st.subheader("Compare countries")
    multi_countries = st.multiselect(
        "Pick up to 6 countries",
        sorted(features["country"].unique()),
        default=["India", "USA", "UK", "Brazil"][:4]
            if all(c in features["country"].values for c in ["India", "USA", "UK", "Brazil"])
            else sorted(features["country"].unique())[:4],
        max_selections=6,
    )

    metric_choice = st.radio(
        "Metric",
        ["new_cases_smoothed", "cases_per_million", "r_effective_approx", "cfr_rolling_28d"],
        horizontal=True,
    )

    if multi_countries:
        sub = features[features["country"].isin(multi_countries)]
        fig = px.line(
            sub, x="date", y=metric_choice, color="country",
            title=f"{metric_choice} over time",
        )
        fig.update_layout(height=480, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    if multi_countries:
        st.markdown("##### Summary statistics over the full period")
        summary = features[features["country"].isin(multi_countries)].groupby("country").agg(
            mean_new_cases=("new_cases_smoothed", "mean"),
            peak_new_cases=("new_cases_smoothed", "max"),
            mean_r_eff=("r_effective_approx", "mean"),
            mean_cfr=("cfr_rolling_28d", "mean"),
        ).round(3)
        st.dataframe(summary, use_container_width=True)


# ----------------------------------- Tab 3: Insights -------------------------

with tab_insights:
    st.subheader("Model performance & fairness audit")

    if not metrics:
        st.info(
            "No model metrics yet. Run `python -m src.models.train --models all` "
            "to generate them."
        )
    else:
        # Headline metrics
        rows = []
        for model, m in metrics.items():
            rows.append({
                "Model": model,
                "MAPE (%)": m["overall_mape"],
                "MAE": m["overall_mae"],
                "RMSE": m["overall_rmse"],
                "80% PI coverage": (
                    f"{m['interval_coverage_80']*100:.1f}%"
                    if m.get("interval_coverage_80") else "—"
                ),
                "n_predictions": m["n_predictions"],
            })
        head_df = pd.DataFrame(rows).sort_values("MAPE (%)")
        st.markdown("##### Overall accuracy")
        st.dataframe(head_df.set_index("Model"), use_container_width=True)

        # Continent breakdown — the fairness audit
        st.markdown("##### Performance by continent (fairness audit)")
        continent_rows = []
        for model, m in metrics.items():
            for continent, vals in m.get("by_continent", {}).items():
                continent_rows.append({
                    "Model": model,
                    "Continent": continent,
                    "MAPE (%)": vals["mape"],
                    "n": vals["n"],
                })
        if continent_rows:
            cont_df = pd.DataFrame(continent_rows)
            fig = px.bar(
                cont_df, x="Continent", y="MAPE (%)",
                color="Model", barmode="group",
                title="Forecast error by continent — does the model generalise?",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "If any continent has materially higher MAPE, the model is "
                "less reliable there — a calibration issue worth flagging."
            )

        # Horizon decay
        st.markdown("##### Error growth with forecast horizon")
        horizon_rows = []
        for model, m in metrics.items():
            for h, vals in m.get("by_horizon_day", {}).items():
                horizon_rows.append({
                    "Model": model,
                    "Horizon (days ahead)": int(h),
                    "MAPE (%)": vals["mape"],
                })
        if horizon_rows:
            h_df = pd.DataFrame(horizon_rows)
            fig2 = px.line(
                h_df, x="Horizon (days ahead)", y="MAPE (%)",
                color="Model", markers=True,
                title="MAPE vs forecast horizon",
            )
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)
