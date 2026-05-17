# Pandemic Severity Forecaster

> Forecast country-level pandemic severity 14 days ahead using live epidemiological data — comparing classical ML, time-series, and deep-learning models, with an interactive dashboard for outbreak comparison.

## What this project does

1. **Pulls live data** from the [disease.sh](https://disease.sh/) open API: historical COVID-19 time series for 200+ countries, plus country-level vaccination, demographics, and current state.
2. **Engineers epidemiologically meaningful features** — 7-day growth rate, effective reproduction number (R-effective) approximation, case fatality rate, recovery rate, population-adjusted incidence.
3. **Forecasts case counts 14 days ahead** with three models:
   - **LightGBM** — global model with quantile regression for uncertainty
   - **Prophet** — per-country seasonal + trend decomposition
   - **LSTM (PyTorch)** — multivariate sequence model with attention-free architecture
4. **Compares outbreak severity across countries** — which factors most predicted COVID severity? Vaccination rollout speed? Demographics? Healthcare capacity proxies?
5. **Ships as a Streamlit dashboard** — interactive country comparison, forecast visualisation, severity scoring.

## Why this matters

Pandemic preparedness is one of the clearest cases for data science in the public interest. Better short-horizon forecasting helps health systems allocate ICU capacity, staff, and supplies. The cross-country comparison highlights what policy and structural factors made a difference — directly relevant to health-informatics research priorities.

## Quick start

```bash
# 1. Clone or unzip, then:
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or: .venv\Scripts\Activate.ps1                     # Windows PowerShell

pip install -r requirements.txt

# 2. Fetch data (takes ~30 seconds, no API key needed)
python -m src.data.fetch

# 3. Build features
python -m src.features.build

# 4. Train all three models
python -m src.models.train --models all

# 5. Launch the dashboard
streamlit run app.py
```

## Repo layout

```
.
├── app.py                  # Streamlit dashboard entry point
├── data/
│   ├── raw/                # JSON pulled from disease.sh (gitignored)
│   └── processed/          # Cleaned panels, features (gitignored)
├── notebooks/              # 01_explore → 04_results
├── src/
│   ├── data/
│   │   ├── client.py       # disease.sh API client with retries + caching
│   │   └── fetch.py        # Orchestrates downloads
│   ├── features/
│   │   └── build.py        # Epi features: R-eff, growth rate, etc.
│   ├── models/
│   │   ├── lightgbm.py     # Quantile-regression LightGBM
│   │   ├── prophet.py      # Per-country Prophet
│   │   ├── lstm.py         # PyTorch LSTM
│   │   ├── evaluate.py     # Metrics, rolling-origin CV
│   │   └── train.py        # CLI orchestrator
│   └── viz/
│       └── plots.py        # Shared plotting helpers
├── reports/                # Figures, JSON results
└── tests/                  # Unit tests
```

## Data source

[disease.sh](https://disease.sh/) is a free, open, no-auth API aggregating data from Johns Hopkins, ECDC, and government sources. Updated daily. We use:

- `/v3/covid-19/historical/{country}?lastdays=all` — daily cases/deaths/recoveries
- `/v3/covid-19/countries/{country}` — current state + population + demographics
- `/v3/covid-19/vaccine/coverage/countries/{country}` — vaccination time series
- `/v3/influenza/{params}` — current influenza-like illness data (CDC)

## Methods

**Target:** 14-day-ahead case count per country (smoothed with 7-day rolling mean to suppress weekday reporting effects).

**Features per country-day:**
- Lagged cases (1, 3, 7, 14, 21, 28 days)
- 7-day and 14-day rolling growth rate
- R-effective approximation (Wallinga-Teunis 2004, simplified)
- Population, population density, median age, GDP per capita (static covariates)
- Vaccination coverage (where available)
- Calendar: day of week, month, days since outbreak start

**Validation:** Rolling-origin (expanding window) cross-validation with 4 splits, 14-day test horizon each.

**Metrics:** MAPE, MAE, pinball loss (for quantile predictions), prediction-interval coverage.

**Subgroup analysis:** Performance broken down by country income group (World Bank classification) — to see whether the model generalises equally well to lower-income countries (data quality there is often poorer).

## What's not in scope

- **Predicting deaths or hospital admissions** — could be added but the 14-day cases forecast is already a meaningful health-system signal.
- **Patient-level modelling** — would require data we don't have access to.
- **Real-time prediction serving** — the dashboard refreshes on demand; productionising as a service is out of scope.

## Author
Abhisri Ravi · MSc Advanced Computer Science (AI), University of Leeds · 2026

## Licence
Code: MIT. Underlying API data: free for non-commercial use per disease.sh terms.
