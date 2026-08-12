"""
data_loader.py -- Bring your own CSV. No labeled shock date required.

This is the piece that answers "how do I use this without an example
dataset / without knowing where the shock was?" There are two supported
workflows:

  WORKFLOW A -- you know the event date
  --------------------------------------
      from data_loader import load_and_diagnose
      result = load_and_diagnose(
          "my_deposit_flows.csv", value_col="net_flow_pct",
          date_col="date", shock_date="2025-03-10",   # the day the news broke
          name="My Bank -- March deposit run",
      )

  WORKFLOW B -- you DON'T know the exact event date (auto-detect)
  -----------------------------------------------------------------
      result = load_and_diagnose(
          "my_transactions.csv", value_col="anomaly_score",
          date_col="date", shock_date=None,             # let it find the shock
          name="Suspicious transaction pattern",
      )

Either way you get back the same OnsagerResult: tau_equilibrium, tau_shock,
validity_score, linear_response_holds. No ground-truth label is required
for either workflow -- ground truth was ONLY used in backtest.py to grade
the diagnostic against synthetic scenarios we built ourselves. On your own
real data, there is no "answer key"; you're using the tool to generate a
diagnosis, not to check one.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from core import OnsagerDiagnostic, OnsagerResult


def load_series_from_csv(path: str, value_col: str, date_col: str | None = None) -> pd.Series:
    """Read a CSV and return a single numeric column as a plain array-indexed
    pandas Series (sorted by date if a date column is given)."""
    df = pd.read_csv(path)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    series = df[value_col].astype(float).interpolate().bfill().ffill()
    return series


def detect_shock_index(values: np.ndarray, min_pre: int = 60, min_post: int = 20,
                        z_window: int = 20, z_threshold: float = 3.0) -> int | None:
    """
    Auto-detect the most likely 'shock' point in a series with NO labeled
    event date: at each candidate point (after min_pre, leaving at least
    min_post points afterward), compute how many rolling standard deviations
    away from its own trailing mean/trend that point's LEVEL SHIFT is, using
    a simple before/after window comparison. Return the index of the biggest
    such break, or None if nothing clears the threshold (i.e. the series
    looks like pure equilibrium noise with no detectable shock at all --
    itself a valid and useful answer).

    This is intentionally simple (a moving z-score of the local mean shift)
    rather than a full changepoint-detection library, so it has no extra
    dependencies and is easy to audit line by line. Swap in `ruptures` or
    a Bayesian changepoint model for production use on noisier data.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    best_idx, best_z = None, 0.0

    for t in range(min_pre, n - min_post):
        pre = values[max(0, t - z_window): t]
        post = values[t: t + z_window]
        if len(pre) < 5 or len(post) < 5:
            continue
        pooled_std = np.std(pre) + 1e-9
        shift = abs(post.mean() - pre.mean())
        z = shift / pooled_std
        if z > best_z:
            best_z, best_idx = z, t

    if best_idx is None or best_z < z_threshold:
        return None
    return best_idx


def load_and_diagnose(path: str, value_col: str, date_col: str | None = None,
                       shock_date=None, name: str | None = None,
                       **diagnostic_kwargs) -> OnsagerResult:
    """
    One-call convenience wrapper: CSV in, OnsagerResult out.

    shock_date : a date string/Timestamp if you know the event date
                 (Workflow A), or None to auto-detect it (Workflow B).
    """
    df = pd.read_csv(path)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    series = df[value_col].astype(float).interpolate().bfill().ffill().values

    if shock_date is not None and date_col is not None:
        shock_ts = pd.to_datetime(shock_date)
        matches = np.where(df[date_col] >= shock_ts)[0]
        if len(matches) == 0:
            raise ValueError(f"shock_date {shock_date} is after the last date in the CSV.")
        shock_index = int(matches[0])
        detection_note = f"shock date given explicitly: {shock_date}"
    else:
        shock_index = detect_shock_index(series)
        if shock_index is None:
            raise ValueError(
                "No shock could be auto-detected in this series (no point deviates "
                "enough from its trailing baseline). Either the series is pure "
                "equilibrium noise with no event to diagnose, or lower z_threshold "
                "in detect_shock_index()."
            )
        detection_note = f"shock auto-detected at row {shock_index}"

    print(f"[{name or path}] {detection_note}")

    diag = OnsagerDiagnostic(series, shock_index, name=name or os.path.basename(path),
                              **diagnostic_kwargs)
    return diag.run()
