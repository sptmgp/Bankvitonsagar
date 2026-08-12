"""
run_examples.py -- Demonstrates BOTH usage workflows against the example CSVs.

    python examples/run_examples.py

For every CSV in examples/data/:
  1. WORKFLOW A: run the diagnostic using the known shock date from MANIFEST.csv
     (as if you were an analyst who knows exactly when the rate hike / news
     story / campaign happened).
  2. WORKFLOW B: run the diagnostic again with shock_date=None, forcing it to
     auto-detect the shock from the data alone (as if you had no idea an event
     had even happened, and were just scanning a metric for trouble).

Then prints both side by side, so you can see how much the auto-detected
shock index differs from the true one, and how that affects validity_score.
This is the honest answer to "how does this work without an example
dataset" -- Workflow B needs nothing but the raw numbers.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_and_diagnose, detect_shock_index, load_series_from_csv
from core import diagnose_multiple_shock_episodes

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    manifest = pd.read_csv(os.path.join(DATA_DIR, "MANIFEST.csv"))

    print(f"{'Dataset':38}{'Workflow':10}{'Shock row':>10}{'tau_eq':>9}{'tau_shock':>11}{'Validity':>10}  Verdict")
    print("-" * 108)

    for _, row in manifest.iterrows():
        path = os.path.join(DATA_DIR, row["file"])

        # --- Workflow A: known shock date ---
        result_a = load_and_diagnose(
            path, value_col=row["value_col"], date_col=row["date_col"],
            shock_date=row["known_shock_date"], name=row["file"],
        )
        verdict_a = "HOLDS" if result_a.linear_response_holds else "FAILS"
        print(f"{row['file']:38}{'known-date':10}{result_a.shock_index:>10}"
              f"{result_a.tau_equilibrium:>9}{result_a.tau_shock:>11}"
              f"{result_a.validity_score:>10.2f}  {verdict_a}")

        # --- Workflow B: blind auto-detection, no date given at all ---
        series = load_series_from_csv(path, value_col=row["value_col"], date_col=row["date_col"])
        auto_idx = detect_shock_index(series.values)
        if auto_idx is None:
            print(f"{'':38}{'auto-detect':10}{'--':>10}   (no shock cleared the detection threshold)")
        else:
            result_b = load_and_diagnose(
                path, value_col=row["value_col"], date_col=row["date_col"],
                shock_date=None, name=row["file"] + " (auto)",
            )
            verdict_b = "HOLDS" if result_b.linear_response_holds else "FAILS"
            print(f"{'':38}{'auto-detect':10}{result_b.shock_index:>10}"
                  f"{result_b.tau_equilibrium:>9}{result_b.tau_shock:>11}"
                  f"{result_b.validity_score:>10.2f}  {verdict_b}")
        print()


if __name__ == "__main__":
    main()

    # -----------------------------------------------------------------
    # WORKFLOW C: multiple real historical episodes of the same kind of
    # shock, averaged together -- the realistic fix for single-path noise
    # when you don't have a simulator to Monte Carlo (see README).
    # -----------------------------------------------------------------
    print("=" * 108)
    print("Workflow C: averaging MULTIPLE real historical episodes of the same shock")
    print("=" * 108)

    multi_path = os.path.join(DATA_DIR, "credit_delinquency_rate_multi_episode.csv")
    shocks_manifest = pd.read_csv(os.path.join(DATA_DIR, "MULTI_EPISODE_SHOCKS.csv"))
    full_series = load_series_from_csv(multi_path, value_col="delinquency_rate_change_bps", date_col="date").values

    episodes = [(full_series, int(row["shock_row"])) for _, row in shocks_manifest.iterrows()]

    # compare: diagnosing episode 1 ALONE vs. averaging all episodes together
    from core import OnsagerDiagnostic
    single_result = OnsagerDiagnostic(*episodes[0], name="episode 1 only").run()
    print(f"Single episode (episode 1 only):     tau_eq={single_result.tau_equilibrium:>6} "
          f"tau_shock={single_result.tau_shock:>7}  validity={single_result.validity_score:.3f}")

    for n in (2, 3, 4):
        multi_result = diagnose_multiple_shock_episodes(episodes[:n], name=f"{n} episodes averaged")
        print(f"Averaged across {n} episodes:{'':11}tau_eq={multi_result.tau_equilibrium:>6} "
              f"tau_shock={multi_result.tau_shock:>7}  validity={multi_result.validity_score:.3f}")

    print("\nNote: with only a handful of real episodes (2-4, as is realistic for something like")
    print("rate-hike cycles), improvement isn't perfectly monotonic -- that's expected small-sample")
    print("behavior, not a bug. The value of averaging is seeing whether independent episodes broadly")
    print("AGREE with each other, rather than anchoring a decision on a single noisy snapshot.")
