"""
PyTorch LSTM forecaster — multivariate sequence-to-one architecture.

Why an LSTM for this problem
============================
Cases on day t depend on cases over the last few weeks via the underlying
SEIR-like dynamics (infected → infectious → recovered, with a delay of
several days). An LSTM is well-suited to capture this kind of temporal
dependency without us having to hand-engineer every lag.

Architecture
============
- Input: a window of W=28 days × F features (lag values + epi state + static)
- One LSTM layer (hidden=64) → dropout → linear head → single output
  (point forecast of new_cases_smoothed at t+H)
- Trained per-country? No. We train *one* global model on (country, window)
  pairs, with country embedded via static covariates. Same reasoning as
  the LightGBM: cross-country transfer beats data-starved per-country fits.
- Loss: MSE on log1p-transformed target (variance grows with the level,
  so log-space training is far more stable for case counts).

How this complements LightGBM
=============================
- LightGBM uses point features at a single time step (with explicit lags).
- LSTM consumes the raw sequence and can in principle learn timing patterns
  the GBM can't express through hand-crafted lags.
- Empirically: GBM usually wins on tabular forecasting; LSTM is the credible
  deep-learning baseline. We report both honestly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.evaluate import Split, iter_split_data

log = logging.getLogger(__name__)

# We learn on log1p-transformed cases for variance stabilisation
def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0))


def _expm1(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(x), 0)


# Features consumed by the LSTM at each timestep
SEQ_FEATURES: list[str] = [
    "new_cases_smoothed",   # the level we're predicting
    "growth_rate_7d",
    "r_effective_approx",
    "cfr_rolling_28d",
    "cases_per_million",
    "vaccination_coverage",
    "dow",
    "is_weekend",
]
# Static features broadcast across the window
STATIC_FEATURES: list[str] = ["population", "days_since_outbreak"]
TARGET_COL = "new_cases_smoothed"


@dataclass
class LSTMConfig:
    window_days: int = 28
    horizon_days: int = 14
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 256
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "cpu"   # set to "cuda" if you have a GPU


def _build_sequences(
    features_df: pd.DataFrame,
    cfg: LSTMConfig,
    train_end: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Slice each country's series into overlapping windows.

    Parameters
    ----------
    features_df : full features panel
    cfg : LSTMConfig
    train_end : if given, only build windows where the target date is ≤ train_end
                (used for training); set to None for inference.

    Returns
    -------
    X        : array [N, W, F_seq + F_static]
    y        : array [N], log1p target
    y_true   : array [N], the actual (non-log) target (for inverse-transform later)
    meta     : DataFrame [N rows] with country and target_date (for joining predictions)
    """
    windows = []
    targets = []
    targets_raw = []
    meta_rows = []

    feature_cols = SEQ_FEATURES + STATIC_FEATURES

    for country, sub in features_df.sort_values(["country", "date"]).groupby("country"):
        sub = sub.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        if len(sub) < cfg.window_days + cfg.horizon_days + 1:
            continue
        # Fill remaining NaN features with 0
        sub_filled = sub[feature_cols].fillna(0).astype(float).values

        max_i = len(sub) - cfg.horizon_days
        for i in range(cfg.window_days, max_i):
            target_date = sub["date"].iloc[i + cfg.horizon_days]
            if train_end is not None and target_date > train_end:
                continue
            window = sub_filled[i - cfg.window_days : i]   # shape (W, F)
            target_val = sub[TARGET_COL].iloc[i + cfg.horizon_days]
            if not np.isfinite(target_val):
                continue
            windows.append(window)
            targets.append(_log1p(np.array([target_val]))[0])
            targets_raw.append(target_val)
            meta_rows.append({
                "country": country,
                "forecast_made_on": sub["date"].iloc[i],
                "target_date": target_date,
            })

    if not windows:
        return (
            np.empty((0, cfg.window_days, len(feature_cols))),
            np.array([]),
            np.array([]),
            pd.DataFrame(),
        )

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    y_true = np.asarray(targets_raw, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, y_true, meta


class _LSTMNet:
    """Wraps construction so we don't import torch at module level."""

    def __init__(self, n_features: int, cfg: LSTMConfig):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, n_feat: int, c: LSTMConfig):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_feat,
                    hidden_size=c.hidden_size,
                    num_layers=c.num_layers,
                    batch_first=True,
                    dropout=c.dropout if c.num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(c.dropout)
                self.head = nn.Linear(c.hidden_size, 1)

            def forward(self, x):  # noqa: D401
                # x: (B, W, F)
                out, _ = self.lstm(x)
                h = out[:, -1, :]  # last timestep
                h = self.dropout(h)
                return self.head(h).squeeze(-1)

        self.model = Net(n_features, cfg)


def run_lstm(
    features_df: pd.DataFrame,
    splits: list[Split],
    config: LSTMConfig | None = None,
) -> pd.DataFrame:
    """Train + predict across rolling-origin splits. Returns predictions
    DataFrame with the same schema as run_lightgbm (but without quantiles)."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError("torch not installed: pip install torch") from e

    cfg = config or LSTMConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("LSTM device: %s", cfg.device)

    all_rows: list[dict] = []

    for split_idx, split, train, test in iter_split_data(features_df, splits):
        log.info("  Split %d: %s", split_idx + 1, split)

        # Training windows — target dates must fall within the train window
        X_train, y_train, _, _ = _build_sequences(train, cfg, train_end=split.train_end)
        if X_train.shape[0] < 200:
            log.warning("    Too few training windows (%d), skipping", X_train.shape[0])
            continue

        # Build inference windows: forecast-made dates in train, target dates
        # in the test window
        infer_window_start = split.test_start - pd.Timedelta(days=cfg.horizon_days)
        infer_window_end = split.test_end - pd.Timedelta(days=cfg.horizon_days)
        # We pass the FULL feature dataframe and filter by forecast_made_on
        X_infer, _, _, meta_infer = _build_sequences(features_df, cfg, train_end=None)
        if X_infer.shape[0] == 0:
            continue
        mask = (
            (meta_infer["forecast_made_on"] >= infer_window_start)
            & (meta_infer["forecast_made_on"] <= infer_window_end)
        )
        X_infer = X_infer[mask.values]
        meta_infer = meta_infer.loc[mask.values].reset_index(drop=True)
        if X_infer.shape[0] == 0:
            continue

        # Per-feature standardisation using training-set statistics only
        mean = X_train.reshape(-1, X_train.shape[2]).mean(axis=0)
        std = X_train.reshape(-1, X_train.shape[2]).std(axis=0) + 1e-6
        X_train_std = (X_train - mean) / std
        X_infer_std = (X_infer - mean) / std

        # Convert to tensors
        device = torch.device(cfg.device)
        X_train_t = torch.from_numpy(X_train_std).to(device)
        y_train_t = torch.from_numpy(y_train).to(device)
        X_infer_t = torch.from_numpy(X_infer_std).to(device)

        net = _LSTMNet(X_train.shape[2], cfg).model.to(device)
        optim = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        loss_fn = torch.nn.MSELoss()
        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        net.train()
        for epoch in range(cfg.n_epochs):
            total = 0.0
            n_batch = 0
            for xb, yb in loader:
                pred = net(xb)
                loss = loss_fn(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += loss.item()
                n_batch += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info(
                    "    Epoch %2d/%d  MSE(log) = %.4f",
                    epoch + 1, cfg.n_epochs, total / max(n_batch, 1),
                )

        # Inference
        net.eval()
        with torch.no_grad():
            preds_log = net(X_infer_t).cpu().numpy()
        preds = _expm1(preds_log)

        # Join with true values
        for i, row in meta_infer.iterrows():
            mask_t = (
                (features_df["country"] == row["country"])
                & (features_df["date"] == row["target_date"])
            )
            true_match = features_df.loc[mask_t, TARGET_COL]
            if true_match.empty:
                continue
            y_true = float(true_match.iloc[0])
            if not np.isfinite(y_true):
                continue
            all_rows.append({
                "model": "lstm",
                "split": split_idx,
                "country": row["country"],
                "date": row["target_date"],
                "y_true": y_true,
                "y_pred": float(preds[i]),
            })

    return pd.DataFrame(all_rows)
