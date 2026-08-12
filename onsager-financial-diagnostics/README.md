# Onsager Financial Diagnostics

A falsifiable statistical test -- not a metaphor -- for whether **linear response theory**
holds for a given financial or operational system, right now, based only on its everyday
fluctuation data.

**[Open the live dashboard](dashboard/dashboard.html)** (charts are hand-drawn with plain
Canvas -- no Chart.js, no external charting library, nothing that can silently fail to
load. The page only reaches the internet for Google Fonts and an optional CSV-parsing
helper, both of which degrade gracefully if unavailable; upload processing and all math
happen entirely client-side).

**[Read the full documentation](docs/quickstart.html)** -- bilingual (English/Spanish
toggle), covers the physics this whole method is built on, all three real-data workflows,
all 10 scenarios in depth, accounting & audit applications across banking/credit/market
risk/fraud/ESG, and two separate plugins (an RG-LLM trading agent, and a LatAm regulatory
context map for Ecuador/Colombia/Mexico/Peru/Chile/Brazil).

## The one-sentence idea

Onsager's regression hypothesis, from equilibrium statistical mechanics, claims that a
system's **spontaneous fluctuations** decay back to baseline with the *same* characteristic
time constant as its **response to an external shock**. This project tests that claim
directly, on ten realistic financial/operational scenarios, instead of just asserting it.

If it holds: you can forecast how a system will recover from a *future* shock using nothing
but its *everyday* noise -- no need to wait for (or simulate) an actual crisis.

If it doesn't hold: the system is in a non-linear regime -- a real panic, a genuine
structural break -- and that mismatch is itself the useful signal. A model that can't tell
you when its own assumptions break down is more dangerous than no model at all; that failure
mode is a large part of what went wrong with pre-2008 risk models that assumed markets
always mean-revert.

## What's actually in this repo

```
onsager-financial-diagnostics/
├── README.md              <- you are here
├── LICENSE                <- MIT
├── .gitignore
├── requirements.txt       <- numpy, pandas
├── results.json           <- output of the last backtest run
├── src/
│   ├── core.py            <- the statistical method (domain-agnostic)
│   ├── modules.py         <- 10 realistic scenario generators (domain-specific)
│   ├── backtest.py        <- runs everything, scores accuracy, writes results.json
│   └── data_loader.py     <- CSV loader + auto shock-detection (bring your own data)
├── examples/
│   ├── generate_example_datasets.py  <- (re)builds examples/data/*.csv
│   ├── run_examples.py               <- demonstrates all 3 real-data workflows
│   └── data/                         <- 10 example CSVs + 1 multi-episode CSV + MANIFEST.csv
├── dashboard/
│   └── dashboard.html     <- self-contained HTML dashboard (embeds results.json,
│                              "Bring your own data" panel, dependency-free Canvas charts)
├── docs/
│   └── quickstart.html    <- full bilingual (EN/ES) documentation site: the physics,
│                              workflows, all 10 scenarios, accounting/audit applications,
│                              LatAm regulatory context plugin, trading plugin
├── assets/
│   ├── logo-icon.svg / logo-badge.svg  <- the project mark
│   └── relaxation-animation.gif        <- animated hero visual
└── metered_api_demo.py    <- Flask wrapper demo (see docs/quickstart.html for context)
```

### `src/core.py` -- the method

For each scenario:

1. Take the **pre-shock window** (normal operation) and compute its autocorrelation
   function (ACF). Fit an exponential decay to it: `ACF(lag) ~ exp(-lag / tau_equilibrium)`.
2. Take the **post-shock window** and Monte Carlo-average many simulated recovery paths
   (300 by default) from the same starting point, to get a clean expected recovery curve
   (a single noisy path is a bad estimator of a relaxation time -- averaging is the
   textbook fix). Fit the same kind of exponential decay to get `tau_shock`.
3. Compare them: `validity_score = 1 - |tau_equilibrium - tau_shock| / max(...)`.
4. If `validity_score >= 0.55` (configurable), the diagnostic says **linear response
   holds** for this system right now. Otherwise it flags a **non-linear regime**.

This class knows nothing about banking, credit, or fraud -- it only knows how to fit and
compare relaxation curves. All the domain realism lives in `modules.py`.

### `src/modules.py` -- ten realistic scenarios

Each scenario is an AR(1) process (the standard way to simulate "fluctuations that decay
exponentially," which is exactly what Onsager's hypothesis assumes for the equilibrium
side) that gets hit with a shock partway through.

**Eight scenarios are built so the post-shock dynamics genuinely match the pre-shock
dynamics** -- linear response *should* hold, and mostly does (see results below):

| # | Scenario | Maps to |
|---|---|---|
| 1 | Bank liquidity (resilient bank) | Deposit run resolved by reassurance |
| 3 | Credit risk -- IFRS 9 rate-hike response | Delinquency-rate migration forecasting |
| 4 | Market risk -- Level-3 spread shock | Bid-ask spread recovery, derivative valuation |
| 5 | Supply chain disruption | Inventory cost after a port closure |
| 6 | ESG / energy cost shock | Operating cost response to an energy price spike |
| 7 | Retail campaign effect | "Consumer memory" after a marketing push |
| 8 | Fraud detection (legitimate anomaly) | A one-off bulk transaction, not fraud |
| 10 | AI coaching feedback loop | Trainee score convergence after coaching |

**Two scenarios are deliberately built to *break* linear response** -- a different, harsher
dynamic kicks in after the shock (a real panic / a real manipulation), specifically to test
whether the diagnostic correctly refuses to call these "normal":

| # | Scenario | Maps to |
|---|---|---|
| 2 | Bank liquidity (panic / bank run) | Self-reinforcing withdrawal cascade |
| 9 | Fraud detection (real manipulation) | Sustained, non-reverting transaction shift |

### `src/backtest.py` -- runs it and scores it honestly

Because each scenario's ground truth (was it built to hold or break linear response?) is
known, `backtest.py` scores the diagnostic's own accuracy -- this isn't a demo that always
"works," it's a testable claim with a pass/fail rate.

### `dashboard/dashboard.html` -- the results, visually

A single self-contained HTML file (results are embedded directly in it -- no server, no
build step, no external charting library -- charts are hand-drawn with plain Canvas, so
they can't silently fail to load the way a CDN-hosted chart library can). Shows, per
scenario: the raw series with the shock marked, a second chart comparing the two decay
curves the verdict is actually based on, both fitted relaxation times, the validity score,
a plain-English explanation of what it means, and whether the diagnostic's call matched
the scenario's ground truth. Also has a **"Bring your own data"** panel to run the same
diagnostic on your own CSV, entirely in-browser -- see the section below.

## Results from the last run

```
Bank liquidity (resilient bank)            tau_eq=1.37   tau_shock=1.48   validity=0.93  -> HOLDS      [correct]
Bank liquidity (panic / bank run)          tau_eq=0.87   tau_shock=55.0   validity=0.02  -> FAILS      [correct]
Credit risk -- IFRS 9 rate-hike response   tau_eq=2.10   tau_shock=2.03   validity=0.96  -> HOLDS      [correct]
Market risk -- Level-3 spread shock        tau_eq=3.46   tau_shock=1.68   validity=0.49  -> FAILS      [MISCLASSIFIED]
Supply chain disruption                    tau_eq=1.18   tau_shock=1.80   validity=0.66  -> HOLDS      [correct]
ESG / energy cost shock                    tau_eq=2.15   tau_shock=1.32   validity=0.61  -> HOLDS      [correct]
Retail campaign effect                     tau_eq=3.76   tau_shock=1.17   validity=0.31  -> FAILS      [MISCLASSIFIED]
Fraud detection (legitimate anomaly)       tau_eq=1.30   tau_shock=1.36   validity=0.96  -> HOLDS      [correct]
Fraud detection (real manipulation)        tau_eq=0.99   tau_shock=262.85 validity=0.00  -> FAILS      [correct]
AI coaching feedback loop                  tau_eq=1.22   tau_shock=1.53   validity=0.80  -> HOLDS      [correct]

Diagnostic accuracy vs. ground truth: 8/10 = 80%
```

**Read honestly, not favorably:**

- **Both deliberately-broken scenarios were caught correctly**, with very low validity
  scores (0.02 and 0.00) -- exactly the behavior you want from an early-warning tool: it
  doesn't just fail quietly, it fails loudly.
- **Two of the eight "should-hold" scenarios were misclassified** (Market risk, Retail
  campaign) due to finite-sample estimation noise in the single-path ACF fit -- a real,
  disclosed limitation, not hidden. This is normal for exponential-decay fits on ~150-200
  data points; longer histories or averaging across multiple independent windows would
  reduce this false-negative rate. The `validity_threshold` (default 0.55) is also a design
  choice you can tune against your own false-positive/false-negative tradeoff.
- 80% accuracy on ten scenarios is **not** a claim of production-grade reliability -- it's
  what an honest first backtest of a new diagnostic looks like.

## Quickstart

```bash
git clone <this-repo>
cd onsager-financial-diagnostics
pip install -r requirements.txt
python src/backtest.py          # regenerates results.json, prints the table above
open dashboard/dashboard.html   # or just double-click it
```

No API keys, no network calls required for the core method. All data is synthetic by
default -- swap in real time series (deposit flows, delinquency rates, transaction logs,
etc.) by passing your own `(series, shock_index)` into `core.OnsagerDiagnostic` directly.

## Using this on your own data (no example dataset required)

Everything above (`backtest.py`, the dashboard) runs on synthetic data with a known
"ground truth" -- that ground truth exists **only** so we can grade the diagnostic against
scenarios we built ourselves. On your own real data, there is no answer key, and you don't
need one: the diagnostic just needs a numeric series and a shock point.

### Fastest way: upload a CSV directly in the dashboard, no install required

`dashboard/dashboard.html` has a **"Bring your own data"** panel at the top. Open the file
in any browser, upload a CSV, pick the value column, optionally give it a shock row index
(leave blank to auto-detect), and click **Run diagnosis**. It appears as a new card at the
top of the grid, above the built-in scenarios -- look for the teal **"YOUR DATA"** tag and
the **✕ remove** button, since those are the only things that distinguish it from the
built-in demo cards. Each result gets the same two charts as the built-in scenarios (the
raw series with the shock marked, and a second chart comparing the two decay curves the
verdict is actually based on) plus a plain-English explanation of what the numbers mean,
written for someone who isn't going to reverse-engineer τ and validity by hand.

Two things worth knowing before you rely on this, found by actually testing it against the
example CSVs in `examples/data/`:

- **Auto-detection can miss a real shock if it decays fast.** It compares 20-row rolling
  averages before/after each candidate point, so a spike that's back to baseline within
  ~4 rows gets diluted almost to nothing by that averaging window and never clears the
  z-score threshold. If auto-detect comes back empty, look up the real shock row yourself
  (`MANIFEST.csv` in `examples/data/` has the known dates for the bundled examples) and
  enter it manually.
- **A manually-typed shock row carries over when you switch files.** Each file has its own
  row count and its own shock position, so a row number that was correct for one CSV can
  silently produce a real, computed, but meaningless result on the next one instead of
  erroring out. Clear the field (or re-check the number) every time you load a new file.

This runs entirely client-side -- the file never leaves the browser, there's no server,
and no data is uploaded anywhere. It's a JavaScript reimplementation of the same
autocorrelation → exponential-τ-fit → validity-score pipeline documented in `core.py`
(verified against the Python backtest: the resilient/panic bank-liquidity scenarios
reproduce validity scores of 0.93 and 0.02 in both implementations, and cross-checking
several of the bundled example CSVs directly against a Python re-implementation of the same
math produced matching validity scores to two decimal places). It is a faithful port of the
documented equations, not a byte-for-byte copy of the Python file -- cross-check against the
Python CLI below for anything you plan to rely on.

### Or from the command line, for more control

`src/data_loader.py` and `examples/` cover three realistic situations:

### Workflow A -- you know the event date

```python
from data_loader import load_and_diagnose

result = load_and_diagnose(
    "my_deposit_flows.csv", value_col="net_flow_pct", date_col="date",
    shock_date="2025-03-10",   # the day the news broke / the rate hike happened
    name="My Bank -- March deposit run",
)
print(result.validity_score, result.linear_response_holds)
```

### Workflow B -- you don't know exactly when the event happened

```python
result = load_and_diagnose(
    "my_transactions.csv", value_col="anomaly_score", date_col="date",
    shock_date=None,   # let it find the shock itself
    name="Suspicious transaction pattern",
)
```

Internally, `detect_shock_index()` scans the series for the point where a trailing
20-day window's mean shifts furthest (in standard-deviation terms) from the window before
it -- a simple, auditable changepoint heuristic (swap in `ruptures` or a Bayesian
changepoint model for noisier real-world data). If nothing clears the threshold, it says so
explicitly rather than forcing a guess -- "no detectable shock" is itself a valid finding
(the series looks like pure equilibrium noise).

### Workflow C -- you have several real historical instances of a similar shock

A single historical shock gives you exactly one noisy recovery path to fit a relaxation
time to -- much less reliable than the 300-path Monte Carlo average `backtest.py` uses on
simulated data (see **Limitations** below). If you have several *real* past instances of a
similar event (three previous rate-hike cycles, four past PR incidents), average them
directly instead:

```python
from core import diagnose_multiple_shock_episodes

episodes = [(series_2022, shock_idx_2022), (series_2023, shock_idx_2023), (series_2024, shock_idx_2024)]
result = diagnose_multiple_shock_episodes(episodes, name="Rate-hike response, 3 cycles")
```

### Ten example CSVs to practice on (`examples/data/`)

Run `python examples/generate_example_datasets.py` to (re)create them, then
`python examples/run_examples.py` to see all three workflows run against them.

| File | Column | Known shock | Should hold? |
|---|---|---|---|
| `bank_deposit_flows_resilient.csv` | `net_deposit_flow_pct` | 2024-10-08 | Yes |
| `bank_deposit_flows_panic.csv` | `net_deposit_flow_pct` | 2024-10-08 | **No** (panic) |
| `credit_delinquency_rate.csv` | `delinquency_rate_change_bps` | 2024-11-05 | Yes |
| `level3_bid_ask_spread.csv` | `spread_bps` | 2024-10-08 | Yes |
| `supply_chain_inventory_cost.csv` | `inventory_cost_index` | 2024-10-08 | Yes |
| `energy_opex_index.csv` | `energy_cost_index` | 2024-10-08 | Yes |
| `retail_daily_sales_index.csv` | `sales_index` | 2024-10-08 | Yes |
| `transaction_pattern_legitimate.csv` | `anomaly_score` | 2024-10-08 | Yes |
| `transaction_pattern_manipulation.csv` | `anomaly_score` | 2024-10-08 | **No** (manipulation) |
| `sales_coaching_score_gap.csv` | `score_gap_to_target` | 2024-09-10 | Yes |
| `credit_delinquency_rate_multi_episode.csv` | same, 4 episodes concatenated | see `MULTI_EPISODE_SHOCKS.csv` | Workflow C demo |

**What running `examples/run_examples.py` actually shows, honestly:**

- **Workflow A (known date) on single real-world paths is noticeably noisier than the
  ensemble-based `backtest.py` results.** Several "should hold" datasets come back as
  FAILS under Workflow A (e.g. `level3_bid_ask_spread.csv`, `retail_daily_sales_index.csv`)
  purely because a single historical recovery path is a much weaker estimator than a
  300-path Monte Carlo average. This is not a bug -- it's the real, disclosed cost of not
  having a simulator to fall back on, and it's exactly why Workflow C exists.
- **Workflow B (blind auto-detection) missed most of the shocks** in this example set at
  the default `z_threshold=3.0` -- these particular synthetic shocks are sized to be
  detectable by eye in a chart but not always by a simple rolling z-score. Lower
  `z_threshold` (e.g. to 2.0) to make detection more sensitive at the cost of more false
  positives, or use a proper changepoint library on real, noisier data.
- **Workflow C (multiple episodes)** narrows the estimate but, with only 2-4 real episodes,
  does **not** improve monotonically every time -- that's expected small-sample behavior.
  The value of averaging real episodes is seeing whether independent instances broadly
  *agree*, not manufacturing a clean trend from too little data.

If you take one thing from this section: **treat any validity score from a single
historical shock as a rough first read, not a verdict** -- exactly the same caution you'd
apply to any statistic estimated from n=1.

## Purpose

To give an accountant, auditor, or risk manager **one diagnostic lens** that works across
liquidity, credit, valuation, operations, fraud, and people-risk -- instead of eight
disconnected, ad hoc models -- and, critically, that tells you **when its own linear
assumption stops being trustworthy**, rather than silently extrapolating through a real
crisis.

## Novelty, stated carefully

Onsager's regression hypothesis is decades-old physics, and quant finance already borrows
adjacent tools (factor models, PCA, GARCH). The genuinely novel parts here are narrower:

1. Framing "relaxation time" as an **audit/compliance diagnostic** (justifying provisions,
   challenging model assumptions, flagging conduct risk) rather than a trading signal --
   a different professional use case than the physics or quant-finance literature usually
   targets.
2. **One conceptual framework, many domains** -- the value isn't any single application
   (each individually has precedent), it's carrying one falsifiable test into liquidity,
   credit, valuation, fraud, ESG, and people-risk conversations instead of learning eight
   disconnected frameworks.
3. Building the "does this even apply here?" check **into the tool itself**, via the
   validity score, rather than treating linear response as an unquestioned assumption.

## Limitations (please read before using this on anything real)

- **Not financial, audit, or investment advice.** This is a research/education
  methodology demo with synthetic data.
- **AR(1) synthetic data is a simplification.** Real financial time series have fatter
  tails, regime-dependent volatility, and cross-asset correlation this demo doesn't model.
- **The misclassification rate above is real and disclosed, not hidden.** Any real
  deployment needs a proper backtest against your own historical data before you trust the
  validity score for anything consequential.
- **Linear response theory has known failure modes in genuine crises** (which is exactly
  what scenarios 2 and 9 are built to demonstrate) -- the tool's job is partly to catch its
  own blind spot, not to claim it has none.
- **Single-path (real-world) estimation is measurably less reliable than the
  ensemble-averaged synthetic backtest above.** `backtest.py` gets to Monte Carlo-average
  300 simulated recovery paths per scenario; on your own real data you typically get exactly
  one historical path per event. See "Using this on your own data" above for what that
  actually does to validity scores (it's worse than you'd hope) and how averaging multiple
  real historical episodes (Workflow C) partially, but not perfectly, compensates.
