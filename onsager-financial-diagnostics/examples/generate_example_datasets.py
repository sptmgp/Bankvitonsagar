"""
generate_example_datasets.py -- Builds the CSVs in examples/data/.

These are SYNTHETIC (built with the same AR(1) engine as src/modules.py),
but exported in the shape a real export from a core banking system, a
BI tool, or a CRM would actually look like: a date column + one value
column, daily frequency. They exist so you can run the CSV loader (both
workflows -- known shock date, and auto-detected) against something
concrete before pointing it at your own data.

Run once to (re)create examples/data/*.csv:
    python examples/generate_example_datasets.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from modules import (
    bank_liquidity_normal, bank_liquidity_panic, credit_risk_rate_hike,
    market_spread_shock, supply_chain_disruption, energy_cost_shock,
    retail_campaign_effect, fraud_anomaly_normal, fraud_anomaly_manipulation,
    coaching_feedback_loop,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2024-01-02"


def to_csv(series, shock_index, filename, value_col, extra_note=""):
    dates = pd.bdate_range(start=START_DATE, periods=len(series))  # business days, like real market/ops data
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), value_col: np.round(series, 4)})
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    shock_date = dates[shock_index].strftime("%Y-%m-%d")
    print(f"{filename:38} rows={len(df):4}  shock on {shock_date}  ({extra_note})")
    return shock_date


DATASETS = [
    (bank_liquidity_normal, "bank_deposit_flows_resilient.csv", "net_deposit_flow_pct",
     "reassured bank run -- linear response should hold"),
    (bank_liquidity_panic, "bank_deposit_flows_panic.csv", "net_deposit_flow_pct",
     "genuine bank run -- linear response should FAIL"),
    (credit_risk_rate_hike, "credit_delinquency_rate.csv", "delinquency_rate_change_bps",
     "IFRS 9 / CECL style delinquency series"),
    (market_spread_shock, "level3_bid_ask_spread.csv", "spread_bps",
     "illiquid derivative bid-ask spread"),
    (supply_chain_disruption, "supply_chain_inventory_cost.csv", "inventory_cost_index",
     "port closure / supplier disruption"),
    (energy_cost_shock, "energy_opex_index.csv", "energy_cost_index",
     "ESG / energy price shock"),
    (retail_campaign_effect, "retail_daily_sales_index.csv", "sales_index",
     "marketing campaign 'consumer memory' effect"),
    (fraud_anomaly_normal, "transaction_pattern_legitimate.csv", "anomaly_score",
     "one-off bulk payment run, NOT fraud"),
    (fraud_anomaly_manipulation, "transaction_pattern_manipulation.csv", "anomaly_score",
     "sustained manipulation -- should FAIL linear response"),
    (coaching_feedback_loop, "sales_coaching_score_gap.csv", "score_gap_to_target",
     "AI-scored sales-call coaching feedback loop"),
]


def make_multi_episode_dataset(n_episodes=4, filename="credit_delinquency_rate_multi_episode.csv",
                                value_col="delinquency_rate_change_bps"):
    """
    A single long CSV containing several independent instances of 'the same
    kind of shock' (several separate rate-hike episodes over multiple years)
    back to back, with clean spacing between them. This is what you'd
    realistically have for something like quarterly rate decisions -- not
    a simulator you can re-run 300 times, but a handful of REAL past
    instances of a similar event. Demonstrates diagnose_multiple_shock_episodes()
    in examples/run_examples.py.
    """
    blocks, shock_indices, offset = [], [], 0
    for _ in range(n_episodes):
        series, shock_idx, _params = credit_risk_rate_hike()
        blocks.append(series)
        shock_indices.append(offset + shock_idx)
        offset += len(series)
    full_series = np.concatenate(blocks)
    to_csv(full_series, shock_indices[0], filename, value_col,
           f"{n_episodes} independent rate-hike episodes concatenated")

    dates = pd.bdate_range(start=START_DATE, periods=len(full_series))
    episodes_manifest = pd.DataFrame({
        "episode": range(1, n_episodes + 1),
        "shock_row": shock_indices,
        "shock_date": [dates[i].strftime("%Y-%m-%d") for i in shock_indices],
    })
    episodes_manifest.to_csv(os.path.join(OUT_DIR, "MULTI_EPISODE_SHOCKS.csv"), index=False)
    print(f"  -> {n_episodes} embedded shock episodes at rows {shock_indices}")


def main():
    manifest = []
    for generator, filename, value_col, note in DATASETS:
        series, shock_index, _params = generator()
        shock_date = to_csv(series, shock_index, filename, value_col, note)
        manifest.append({
            "file": filename, "value_col": value_col, "date_col": "date",
            "known_shock_date": shock_date, "note": note,
        })
    pd.DataFrame(manifest).to_csv(os.path.join(OUT_DIR, "MANIFEST.csv"), index=False)
    print(f"\nWrote {len(manifest)} example CSVs + MANIFEST.csv to {OUT_DIR}")

    print("\nMulti-episode dataset (for diagnose_multiple_shock_episodes demo):")
    make_multi_episode_dataset()


if __name__ == "__main__":
    main()
