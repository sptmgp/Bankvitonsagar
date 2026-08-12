"""
modules.py -- Eight domain-realistic time series generators.

Each function returns (series, shock_index). All series are built from an
AR(1) process, which is the honest, standard way to simulate "equilibrium
fluctuations that relax exponentially" -- the autocorrelation of an AR(1)
process with coefficient rho decays as rho^lag, i.e. exactly the exponential
relaxation Onsager's hypothesis assumes, with tau_eq = -1 / ln(rho).

The point of this file is NOT to claim every financial system obeys linear
response -- it's to show a diagnostic that can tell the difference:

  - Modules 1-6 use the SAME rho before and after the shock -> the recovery
    genuinely follows the same relaxation law as the everyday fluctuations.
    Onsager's hypothesis should hold, and the diagnostic should say so.

  - Modules 7 (bank run) and 8 partially (market liquidity crisis) switch to
    a DIFFERENT, harsher dynamic after the shock (a classic panic/non-linear
    regime) -- Onsager's hypothesis should FAIL here, and a good diagnostic
    tool needs to say that clearly instead of forcing a linear fit onto it.
    This is the realistic, credible part: the tool's job is partly to know
    when it doesn't apply.
"""

import numpy as np

SEED = 7
RNG = np.random.default_rng(SEED)


def _ar1_series(n, rho, sigma, x0=0.0, rng=RNG):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, sigma)
    return x


def _shocked_series(n_pre, n_post, rho_pre, sigma_pre, shock_size,
                     rho_post=None, sigma_post=None, rng=RNG):
    """Build [equilibrium AR(1)] + [shock] + [post-shock AR(1)] as one series.
    Also returns the parameters, so the diagnostic can Monte Carlo-average
    many post-shock paths (a single noisy path is a bad way to estimate a
    relaxation time -- the standard fix is ensemble averaging)."""
    rho_post = rho_pre if rho_post is None else rho_post
    sigma_post = sigma_pre if sigma_post is None else sigma_post

    pre = _ar1_series(n_pre, rho_pre, sigma_pre, x0=0.0, rng=rng)
    post = _ar1_series(n_post, rho_post, sigma_post, x0=pre[-1] + shock_size, rng=rng)
    series = np.concatenate([pre, post])
    shock_index = n_pre
    params = dict(
        rho_pre=rho_pre, sigma_pre=sigma_pre,
        rho_post=rho_post, sigma_post=sigma_post,
        shock_size=shock_size, baseline=float(pre.mean()),
        start_value=float(pre[-1] + shock_size), n_post=n_post,
    )
    return series, shock_index, params


# ---------------------------------------------------------------------------
# 1. Bank liquidity -- deposit flow, NORMAL scenario (no panic cascade)
# ---------------------------------------------------------------------------
def bank_liquidity_normal(n_pre=200, n_post=80):
    """Daily net deposit flow (% of total deposits). A negative-news shock
    hits, but depositors are reassured quickly -- same relaxation dynamics
    as the everyday noise. This is the 'resilient bank' case."""
    return _shocked_series(n_pre, n_post, rho_pre=0.55, sigma_pre=0.4, shock_size=-6.0)


# ---------------------------------------------------------------------------
# 2. Bank liquidity -- PANIC scenario (the actual bank-run failure mode)
# ---------------------------------------------------------------------------
def bank_liquidity_panic(n_pre=200, n_post=80):
    """Same starting equilibrium, but the shock triggers a self-reinforcing
    withdrawal cascade (rho jumps toward 1 -- near-permanent, non-decaying
    outflow) -- the classic non-linear bank-run regime Onsager's hypothesis
    is NOT supposed to survive. This is deliberately the 'fails' example."""
    return _shocked_series(n_pre, n_post, rho_pre=0.55, sigma_pre=0.4, shock_size=-6.0,
                            rho_post=0.985, sigma_post=0.9)


# ---------------------------------------------------------------------------
# 3. Credit risk (IFRS 9 / CECL) -- default-rate response to a rate hike
# ---------------------------------------------------------------------------
def credit_risk_rate_hike(n_pre=220, n_post=90):
    """Daily change in portfolio delinquency rate (bps). A rate-hike shock
    pushes it up; historically this reverts at roughly the same pace as
    normal delinquency noise -- the case for using fluctuation data to
    forecast IFRS 9 stage migrations."""
    return _shocked_series(n_pre, n_post, rho_pre=0.62, sigma_pre=0.3, shock_size=4.5)


# ---------------------------------------------------------------------------
# 4. Market risk -- bid-ask spread response to a volatility shock
# ---------------------------------------------------------------------------
def market_spread_shock(n_pre=200, n_post=80):
    """Level-3 asset bid-ask spread (bps over mid). Widens sharply on a
    volatility shock; this version reverts at a similar rate to normal
    spread noise -- a market where the pricing model's implicit linear
    recovery assumption is reasonable."""
    return _shocked_series(n_pre, n_post, rho_pre=0.58, sigma_pre=0.35, shock_size=5.0)


# ---------------------------------------------------------------------------
# 5. Supply chain -- inventory holding cost after a port closure
# ---------------------------------------------------------------------------
def supply_chain_disruption(n_pre=200, n_post=80):
    return _shocked_series(n_pre, n_post, rho_pre=0.6, sigma_pre=0.3, shock_size=5.5)


# ---------------------------------------------------------------------------
# 6. ESG / energy costs -- operating cost response to an energy price spike
# ---------------------------------------------------------------------------
def energy_cost_shock(n_pre=200, n_post=80):
    return _shocked_series(n_pre, n_post, rho_pre=0.5, sigma_pre=0.35, shock_size=4.0)


# ---------------------------------------------------------------------------
# 7. Retail sales -- response to a marketing campaign ("consumer memory")
# ---------------------------------------------------------------------------
def retail_campaign_effect(n_pre=200, n_post=80):
    return _shocked_series(n_pre, n_post, rho_pre=0.45, sigma_pre=0.4, shock_size=6.0)


# ---------------------------------------------------------------------------
# 8. Forensic / fraud detection -- transaction-pattern anomaly
# ---------------------------------------------------------------------------
def fraud_anomaly_normal(n_pre=200, n_post=80):
    """A one-off irregular transaction batch that is NOT fraud -- e.g. a
    legitimate bulk payment run. Behavior reverts like ordinary noise."""
    return _shocked_series(n_pre, n_post, rho_pre=0.5, sigma_pre=0.3, shock_size=5.0)


def fraud_anomaly_manipulation(n_pre=200, n_post=80):
    """A genuine manipulation event: the transaction pattern does NOT relax
    back to the prior baseline at all (a sustained, deliberate shift) --
    the diagnostic should flag this as non-linear / non-relaxing."""
    return _shocked_series(n_pre, n_post, rho_pre=0.5, sigma_pre=0.3, shock_size=5.0,
                            rho_post=0.995, sigma_post=0.5)


# ---------------------------------------------------------------------------
# 9. AI coaching feedback loop -- trainee score convergence after coaching
# ---------------------------------------------------------------------------
def coaching_feedback_loop(n_pre=180, n_post=70):
    """Distance from target score (lower is better). A coaching session is
    the 'shock'; a well-coached trainee's score gap closes at roughly the
    same rate their score naturally fluctuates day-to-day."""
    return _shocked_series(n_pre, n_post, rho_pre=0.5, sigma_pre=0.35, shock_size=-5.0)


MODULES = {
    "Bank liquidity (resilient bank)": bank_liquidity_normal,
    "Bank liquidity (panic / bank run)": bank_liquidity_panic,
    "Credit risk -- IFRS 9 rate-hike response": credit_risk_rate_hike,
    "Market risk -- Level-3 spread shock": market_spread_shock,
    "Supply chain disruption": supply_chain_disruption,
    "ESG / energy cost shock": energy_cost_shock,
    "Retail campaign effect": retail_campaign_effect,
    "Fraud detection (legitimate anomaly)": fraud_anomaly_normal,
    "Fraud detection (real manipulation)": fraud_anomaly_manipulation,
    "AI coaching feedback loop": coaching_feedback_loop,
}

# Ground truth: was this scenario CONSTRUCTED to obey linear response
# (same dynamics before/after the shock) or to break it (different, harsher
# post-shock dynamics -- a real panic/manipulation regime)? Used only to
# score the diagnostic's own accuracy honestly in backtest.py -- the
# diagnostic itself never sees this label.
GROUND_TRUTH_LINEAR = {
    "Bank liquidity (resilient bank)": True,
    "Bank liquidity (panic / bank run)": False,
    "Credit risk -- IFRS 9 rate-hike response": True,
    "Market risk -- Level-3 spread shock": True,
    "Supply chain disruption": True,
    "ESG / energy cost shock": True,
    "Retail campaign effect": True,
    "Fraud detection (legitimate anomaly)": True,
    "Fraud detection (real manipulation)": False,
    "AI coaching feedback loop": True,
}
