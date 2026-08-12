"""
core.py -- The Onsager Regression Hypothesis, as an actual statistical test.

The claim being tested (not assumed) for every module in this project:

    "A system's spontaneous equilibrium fluctuations decay back to baseline
     with the same characteristic time constant (tau) as its recovery from
     an external shock."

If tau_equilibrium ~= tau_shock  -> linear response theory holds for this
system right now; you can use everyday fluctuation data to forecast how it
will recover from a future shock, cheaply, without waiting for a real crisis.

If tau_equilibrium is very different from tau_shock -> the system is in a
non-linear regime (a real panic, a genuine structural break). This is
itself the useful signal: it tells you when NOT to trust a linear model,
which is exactly the failure mode that sank a lot of pre-2008 risk models.

This module contains no domain-specific assumptions -- it only knows how
to fit exponential relaxation curves and compare two of them. All the
domain realism (what a "shock" looks like for a bank run vs. a marketing
campaign) lives in modules.py.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Standard sample autocorrelation function, lags 0..max_lag."""
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    acf = np.array([
        np.dot(x[: n - lag], x[lag:]) / denom if lag > 0 else 1.0
        for lag in range(max_lag + 1)
    ])
    return acf


def _fit_exponential_tau(curve: np.ndarray, min_points: int = 4, floor: float = 0.03) -> tuple[float, float]:
    """
    Fit curve[t] ~= exp(-t / tau) via linear regression on log(curve).
    Truncates once the curve drops below `floor` (relative to curve[0]==1) --
    past that point we're fitting numerical/sampling noise floor, not the
    actual decay, which silently corrupts the tau estimate if left in.

    Returns (tau, r_squared). tau = np.inf if no decay could be fit
    (e.g. the series never relaxes -- itself a diagnostic finding).
    """
    above_floor = curve > floor
    cutoff = len(curve)
    for i, ok in enumerate(above_floor):
        if not ok:
            cutoff = i
            break
    curve = curve[:cutoff]

    if len(curve) < min_points:
        return np.inf, 0.0

    t = np.arange(len(curve))
    y = np.log(curve)
    # simple least squares: y = a + b*t  =>  tau = -1/b
    A = np.vstack([np.ones_like(t, dtype=float), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    if b >= 0:
        return np.inf, 0.0  # never decays -> no finite relaxation time

    tau = -1.0 / b
    y_pred = a + b * t
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r_squared = 1 - ss_res / ss_tot
    return float(tau), float(r_squared)


@dataclass
class OnsagerResult:
    name: str
    tau_equilibrium: float
    tau_shock: float
    r2_equilibrium: float
    r2_shock: float
    validity_score: float       # 1.0 = perfect match, 0.0 = totally different
    linear_response_holds: bool
    equilibrium_acf: list
    shock_recovery_curve: list
    series: list
    shock_index: int


def _ensemble_shock_curve(rho_post, sigma_post, shock_size, baseline, start_value,
                           n_post, n_paths=300, seed=123):
    """
    Monte Carlo the post-shock recovery n_paths times from the same starting
    point and average |deviation from baseline| at each step. This is the
    correct way to estimate a relaxation time from a stochastic response --
    a single noisy path is a bad estimator; the ensemble average is what
    linear response theory actually predicts.
    """
    rng = np.random.default_rng(seed)
    curves = np.empty((n_paths, n_post))
    for p in range(n_paths):
        x = np.empty(n_post)
        x[0] = start_value
        for t in range(1, n_post):
            x[t] = rho_post * x[t - 1] + rng.normal(0, sigma_post)
        curves[p] = x
    mean_curve = curves.mean(axis=0)
    deviation = np.abs(mean_curve - baseline)
    deviation = deviation / (deviation[0] + 1e-9)
    return deviation


class OnsagerDiagnostic:
    """
    series         : full time series (equilibrium period, then a shock, then recovery) -- used
                     for the equilibrium ACF fit and for the dashboard plot.
    shock_index    : index in `series` where the external shock hits.
    shock_params   : optional dict with rho_post/sigma_post/shock_size/baseline/start_value/n_post.
                     If given, tau_shock is estimated from a Monte Carlo ensemble average (robust).
                     If omitted, falls back to fitting the single observed post-shock path directly
                     (noisier -- realistic for a live deployment where you only get ONE path per shock,
                     which is itself a reason validity scores should be read with that caveat in mind).
    pre_window     : how many points before the shock to use for the equilibrium fit.
    max_lag        : max lag used when fitting the equilibrium relaxation curve.
    validity_threshold : below this validity_score, we flag "linear response does NOT hold".
    """

    def __init__(self, series, shock_index: int, name: str = "", shock_params: dict | None = None,
                 pre_window: int = 150, max_lag: int = 30, validity_threshold: float = 0.55):
        self.series = np.asarray(series, dtype=float)
        self.shock_index = shock_index
        self.name = name
        self.shock_params = shock_params
        self.pre_window = pre_window
        self.max_lag = max_lag
        self.validity_threshold = validity_threshold

    def run(self) -> OnsagerResult:
        pre = self.series[max(0, self.shock_index - self.pre_window): self.shock_index]

        # --- equilibrium relaxation: how fast do NORMAL fluctuations decay? ---
        acf = _autocorrelation(pre, min(self.max_lag, len(pre) - 2))
        tau_eq, r2_eq = _fit_exponential_tau(acf)

        # --- shock relaxation: how fast does the system return to baseline
        #     AFTER the external perturbation? (ensemble average if possible) ---
        if self.shock_params is not None:
            deviation = _ensemble_shock_curve(
                rho_post=self.shock_params["rho_post"],
                sigma_post=self.shock_params["sigma_post"],
                shock_size=self.shock_params["shock_size"],
                baseline=self.shock_params["baseline"],
                start_value=self.shock_params["start_value"],
                n_post=self.shock_params["n_post"],
            )
        else:
            post = self.series[self.shock_index: self.shock_index + 60]
            baseline = pre.mean()
            deviation = np.abs(post - baseline)
            deviation = deviation / (deviation[0] + 1e-9)

        tau_shock, r2_shock = _fit_exponential_tau(deviation)

        # --- compare the two relaxation times: this IS the Onsager test ---
        if np.isinf(tau_eq) or np.isinf(tau_shock):
            validity = 0.0
        else:
            validity = 1 - abs(tau_eq - tau_shock) / max(tau_eq, tau_shock, 1e-9)
            validity = float(np.clip(validity, 0.0, 1.0))

        holds = validity >= self.validity_threshold

        return OnsagerResult(
            name=self.name,
            tau_equilibrium=round(float(tau_eq), 2) if np.isfinite(tau_eq) else -1,
            tau_shock=round(float(tau_shock), 2) if np.isfinite(tau_shock) else -1,
            r2_equilibrium=round(r2_eq, 3),
            r2_shock=round(r2_shock, 3),
            validity_score=round(validity, 3),
            linear_response_holds=holds,
            equilibrium_acf=[round(float(v), 4) for v in acf],
            shock_recovery_curve=[round(float(v), 4) for v in deviation],
            series=[round(float(v), 4) for v in self.series],
            shock_index=self.shock_index,
        )


def diagnose_multiple_shock_episodes(episodes: list, name: str = "",
                                      pre_window: int = 150, post_window: int = 60,
                                      max_lag: int = 30, validity_threshold: float = 0.55) -> OnsagerResult:
    """
    The realistic answer to 'I have real data but no simulator to Monte Carlo
    average' -- IF you have multiple real historical instances of a similar
    shock (three past rate hikes, four past PR crises, five past coaching
    cohorts...), average their observed recovery paths directly. This is the
    same statistical fix as the synthetic ensemble average in
    `_ensemble_shock_curve`, just built from real repeated events instead of
    simulation, and it is meaningfully more reliable than fitting a single
    noisy episode (see README: "Single-path vs. multi-episode reliability").

    episodes : list of (series, shock_index) tuples, one per historical
               instance of the same *kind* of shock (can be different
               absolute time periods, even different lengths).
    """
    acfs, raw_deviations = [], []
    for series, shock_index in episodes:
        series = np.asarray(series, dtype=float)
        pre = series[max(0, shock_index - pre_window): shock_index]
        post = series[shock_index: shock_index + post_window]
        if len(pre) < 10 or len(post) < 10:
            continue
        acfs.append(_autocorrelation(pre, min(max_lag, len(pre) - 2)))
        baseline = pre.mean()
        # NOTE: keep the SIGNED deviation and average that across episodes
        # (noise cancels because independent episodes have independent noise
        # realizations around the same underlying decay). Only take the
        # absolute value ONCE, after averaging. Averaging |deviation| per
        # episode first (as an earlier version of this function did) does
        # NOT cancel noise -- it averages together several non-zero noise
        # floors and never converges toward zero, which silently corrupts
        # the fitted tau_shock. This assumes all episodes are the same KIND
        # of shock in the same direction (e.g. several rate hikes, not a mix
        # of hikes and cuts) -- reasonable for "repeated instances of a
        # similar event," which is the intended use of this function.
        raw_deviations.append(post - baseline)

    if not acfs:
        raise ValueError("No usable episodes (need at least pre_window+post_window points each).")

    min_acf_len = min(len(a) for a in acfs)
    min_dev_len = min(len(d) for d in raw_deviations)
    mean_acf = np.mean([a[:min_acf_len] for a in acfs], axis=0)
    mean_signed_deviation = np.mean([d[:min_dev_len] for d in raw_deviations], axis=0)
    mean_deviation = np.abs(mean_signed_deviation)
    mean_deviation = mean_deviation / (mean_deviation[0] + 1e-9)

    tau_eq, r2_eq = _fit_exponential_tau(mean_acf)
    tau_shock, r2_shock = _fit_exponential_tau(mean_deviation)

    if np.isinf(tau_eq) or np.isinf(tau_shock):
        validity = 0.0
    else:
        validity = 1 - abs(tau_eq - tau_shock) / max(tau_eq, tau_shock, 1e-9)
        validity = float(np.clip(validity, 0.0, 1.0))

    return OnsagerResult(
        name=name,
        tau_equilibrium=round(float(tau_eq), 2) if np.isfinite(tau_eq) else -1,
        tau_shock=round(float(tau_shock), 2) if np.isfinite(tau_shock) else -1,
        r2_equilibrium=round(r2_eq, 3), r2_shock=round(r2_shock, 3),
        validity_score=round(validity, 3),
        linear_response_holds=validity >= validity_threshold,
        equilibrium_acf=[round(float(v), 4) for v in mean_acf],
        shock_recovery_curve=[round(float(v), 4) for v in mean_deviation],
        series=[round(float(v), 4) for v in episodes[0][0]],
        shock_index=episodes[0][1],
    )
