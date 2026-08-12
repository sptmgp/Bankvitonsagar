"""
metered_api_demo.py -- How a "pay per diagnosis" API would mechanically work.

This wraps the REAL OnsagerDiagnostic engine from core.py behind a small
Flask API, with an in-memory ledger simulating a $0.01-per-call charge.

WHAT THIS DOES NOT PROVE
-------------------------
- That anyone wants to buy this. Zero customers have been validated.
- That $0.01/call is a real, sustainable, or competitive price.
- That this technique is a substitute for machine learning in general.
  It answers ONE narrow question -- "does this system's post-shock recovery
  match its everyday fluctuation pattern?" -- and only for systems where
  that assumption is plausible in the first place (see README.md's
  Limitations section). Ride-share pricing, ad auctions, and traffic
  routing are not, in general, that kind of system.

WHAT THIS DOES SHOW
--------------------
The actual mechanics: request in, real computation, response out, a
per-call charge logged -- so you can see exactly what such a product
would look like end-to-end before deciding whether to build a real one.

Run it:
    pip install flask
    python metered_api_demo.py
    # in another terminal:
    curl -X POST http://localhost:5000/v1/diagnose \
      -H "Content-Type: application/json" \
      -d '{"series": [0,1,0.6,0.9,...], "shock_index": 50}'
    curl http://localhost:5000/v1/usage
"""

from flask import Flask, request, jsonify
import sys, os, time, uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core import OnsagerDiagnostic  # the real engine, not a toy

app = Flask(__name__)

PRICE_PER_CALL = 0.01     # illustrative only -- not a validated price
COST_PER_CALL_ESTIMATE = 0.0005  # rough cloud-compute guess -- not measured

# in-memory ledger (a real system would use a database + Stripe metered billing)
LEDGER = []


@app.route("/v1/diagnose", methods=["POST"])
def diagnose():
    t0 = time.time()
    body = request.get_json(force=True)

    series = body.get("series")
    shock_index = body.get("shock_index")
    name = body.get("name", "api-request")

    if not series or shock_index is None:
        return jsonify({"error": "series and shock_index are required"}), 400
    if not (0 < shock_index < len(series)):
        return jsonify({"error": "shock_index must be inside the series"}), 400

    diag = OnsagerDiagnostic(series, shock_index, name=name)
    result = diag.run()
    latency_ms = round((time.time() - t0) * 1000, 3)

    call_id = str(uuid.uuid4())
    LEDGER.append({
        "call_id": call_id,
        "charge_usd": PRICE_PER_CALL,
        "est_cost_usd": COST_PER_CALL_ESTIMATE,
        "latency_ms": latency_ms,
    })

    return jsonify({
        "call_id": call_id,
        "tau_equilibrium": result.tau_equilibrium,
        "tau_shock": result.tau_shock,
        "validity_score": result.validity_score,
        "linear_response_holds": result.linear_response_holds,
        "latency_ms": latency_ms,
        "charged_usd": PRICE_PER_CALL,
        "note": "Illustrative billing only -- no real charge occurred.",
    })


@app.route("/v1/usage", methods=["GET"])
def usage():
    total_calls = len(LEDGER)
    total_revenue = round(sum(c["charge_usd"] for c in LEDGER), 4)
    total_cost = round(sum(c["est_cost_usd"] for c in LEDGER), 4)
    avg_latency = round(sum(c["latency_ms"] for c in LEDGER) / total_calls, 3) if total_calls else 0

    return jsonify({
        "total_calls": total_calls,
        "gross_revenue_usd": total_revenue,
        "estimated_cost_usd": total_cost,
        "estimated_margin_usd": round(total_revenue - total_cost, 4),
        "avg_latency_ms": avg_latency,
        "disclaimer": "Simulated ledger for demonstration. No real payments processed.",
    })


if __name__ == "__main__":
    print("Metered API demo running at http://localhost:5000")
    print("This is a mechanics demo, not a live product -- see the module docstring.")
    app.run(debug=True, port=5000)
