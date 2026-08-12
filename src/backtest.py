"""
backtest.py -- Run the Onsager diagnostic across every domain module and
write a results.json the dashboard can render. Also scores the diagnostic's
own accuracy against the ground-truth labels in modules.py (was this
scenario built to obey linear response or break it?), so "performance"
here means something falsifiable, not just a demo running without errors.

Usage:
    python src/backtest.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from core import OnsagerDiagnostic
from modules import MODULES, GROUND_TRUTH_LINEAR


def run_all():
    results = []
    correct = 0
    for name, generator in MODULES.items():
        series, shock_index, params = generator()
        diag = OnsagerDiagnostic(series, shock_index, name=name, shock_params=params)
        result = diag.run()
        row = result.__dict__.copy()
        row["ground_truth_linear"] = GROUND_TRUTH_LINEAR[name]
        row["correct"] = (result.linear_response_holds == GROUND_TRUTH_LINEAR[name])
        correct += row["correct"]
        results.append(row)

        status = "HOLDS (linear response)" if result.linear_response_holds else "FAILS (non-linear regime)"
        mark = "correct" if row["correct"] else "MISCLASSIFIED"
        print(f"{name:42} tau_eq={result.tau_equilibrium:>7} "
              f"tau_shock={result.tau_shock:>7}  validity={result.validity_score:>5.2f}  "
              f"-> {status:26} [{mark}]")

    return results, correct


def main():
    results, correct = run_all()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {len(results)} module results to {os.path.abspath(out_path)}")

    n_hold = sum(r["linear_response_holds"] for r in results)
    accuracy = correct / len(results)
    print(f"\nSummary: linear response held in {n_hold}/{len(results)} scenarios.")
    print(f"Diagnostic accuracy vs. ground truth: {correct}/{len(results)} = {accuracy*100:.0f}%")
    print("(ground truth = whether each scenario was constructed to obey or break")
    print(" linear response -- the diagnostic itself never sees this label)")


if __name__ == "__main__":
    main()
