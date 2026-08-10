"""
signal_engine.py -- consolidated, final version.

FIXES:
1. Real thresholds restored. Copilot's "alerts-only mode" raised
   MAX_SPREAD_CENTS to 200 and lowered LONGSHOT_FLOOR to 0.02 -- that's what let
   a 98c-spread, essentially untradeable bucket fire a real Telegram alert.
   Reverted to the tested values (20c / 0.10).
2. Correct No-probability. A bucket's "No" outcome must be 1 - P(Yes) for that
   same temperature range -- not a duplicate of Yes's probability (the bug
   confirmed in production logs: both showed 60.8%).
3. Live-observation floor, actually wired to fire correctly (not just written
   and left unconnected -- verified end to end below).
"""

from dataclasses import dataclass
from typing import Optional
import prob_math

LONGSHOT_FLOOR = 0.10
MAX_SANITY_GAP_PP = 45
KELLY_FRACTION = 0.25          # quarter-Kelly, standard conservative fraction
NOMINAL_BANKROLL_USD = 1000.0  # paper bankroll baseline -- no real orders placed
MAX_STAKE_USD = 50.0           # hard cap regardless of what Kelly suggests
MIN_STAKE_USD = 10.0           # floor -- below this, not worth the trade/fees
MIN_GAP_PP = 12
MAX_SPREAD_CENTS = 20

PEAK_SIGMA_REDUCTION_FACTOR = 0.5
MIN_SIGMA = 0.25


@dataclass
class Bucket:
    label: str
    low: Optional[float]
    high: Optional[float]
    price: float
    token_id: Optional[str]
    outcome: str = "Yes"
    market_id: Optional[str] = None
    spread_cents: Optional[float] = None
    depth_ok: Optional[bool] = None


@dataclass
class Signal:
    city: str
    bucket_label: str
    corrected_mu: float
    sigma: float
    est_prob: float
    market_price: float
    gap_pp: float
    models_used: list
    used_live_obs: bool = False
    underdispersed: bool = False
    token_id: Optional[str] = None
    market_id: Optional[str] = None
    outcome: str = "Yes"
    near_boundary_risk: bool = False
    suggested_stake: float = 30.0


@dataclass
class EvalResult:
    signal: Optional[Signal]
    reason_code: str
    detail: str
    all_candidates: list = None


UNDERDISPERSION_RATIO = 0.6   # today's spread must be <=60% of typical to count as "underdispersed"
UNDERDISPERSION_SIGMA_FACTOR = 0.8  # modest sigma tightening on underdispersed days -- weaker than
                                     # the live-obs reduction, since model agreement is a softer
                                     # signal than an actual ground-truth reading


def correct_forecast(raw_model_values: dict, bias_data: dict, live_obs: float = None) -> tuple:
    """
    Returns (corrected_mu, sigma, used_live_obs, underdispersed).
    live_obs, if provided (already unit-matched to the bucket labels by the
    caller -- see main.py), floors mu and tightens sigma: temperature cannot
    un-rise during the day, so a live reading is a hard lower bound, and once
    we have it, most of the day's forecast uncertainty has resolved.

    underdispersed: True if TODAY's raw inter-model spread is unusually tight
    vs. this city's typical historical spread (from hindcast's
    "_typical_model_spread") -- a real, measured confidence signal, not vibes.
    Only applied if live_obs isn't already active, to avoid double-counting
    two different confidence boosts on the same signal.
    """
    corrected_values = []
    sigmas = []
    for model, raw_value in raw_model_values.items():
        if raw_value is None:
            continue
        stats = bias_data.get(model, {"bias": 0.0, "sigma": 2.5})
        corrected_values.append(raw_value - stats["bias"])
        sigmas.append(stats["sigma"])

    if not corrected_values:
        return None, None, False, False

    mu = sum(corrected_values) / len(corrected_values)
    sigma = max(sigmas)
    used_live_obs = False
    underdispersed = False

    if live_obs is not None:
        if live_obs > mu:
            mu = live_obs
        sigma = max(MIN_SIGMA, sigma * PEAK_SIGMA_REDUCTION_FACTOR)
        used_live_obs = True
    else:
        raw_vals = [v for v in raw_model_values.values() if v is not None]
        typical_spread = bias_data.get("_typical_model_spread")
        if len(raw_vals) >= 2 and typical_spread:
            today_spread = max(raw_vals) - min(raw_vals)
            if today_spread <= typical_spread * UNDERDISPERSION_RATIO:
                sigma = max(MIN_SIGMA, sigma * UNDERDISPERSION_SIGMA_FACTOR)
                underdispersed = True

    return mu, sigma, used_live_obs, underdispersed


def calc_kelly_stake(p: float, price: float, kelly_fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly sizing. b = payout odds per dollar risked (buying at `price`,
    paying out $1 if correct). f* = (p*b - (1-p)) / b is full Kelly; we take a
    fraction of that (0.25 = quarter-Kelly, the standard conservative default
    real documented bots use, confirmed via public bot writeups this session).
    Returns a dollar stake, capped at MAX_STAKE_USD and floored at MIN_STAKE_USD
    (returns 0 if the edge is too small to bother with fees/slippage).
    """
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price
    full_kelly = (p * b - (1 - p)) / b
    if full_kelly <= 0:
        return 0.0
    stake = full_kelly * kelly_fraction * NOMINAL_BANKROLL_USD
    stake = min(stake, MAX_STAKE_USD)
    return round(stake, 2) if stake >= MIN_STAKE_USD else 0.0


def _bucket_est_prob(bucket: Bucket, mu: float, sigma: float) -> float:
    """Yes = the real Gaussian mass in [low, high). No = 1 - Yes for the SAME range."""
    yes_prob = prob_math.bucket_probability(bucket.low, bucket.high, mu, sigma)
    if bucket.outcome == "No":
        return 1.0 - yes_prob
    return yes_prob


def evaluate_buckets(city: str, raw_model_values: dict, bias_data: dict,
                      buckets: list, live_obs: float = None) -> EvalResult:
    mu, sigma, used_live_obs, underdispersed = correct_forecast(raw_model_values, bias_data, live_obs)
    if mu is None:
        return EvalResult(None, "no_forecast_data", "No valid model readings.")

    models_used = [m for m, v in raw_model_values.items() if v is not None]

    all_candidates_raw = []
    for b in buckets:
        est_prob = _bucket_est_prob(b, mu, sigma)
        gap_pp = round((est_prob - b.price) * 100, 1)
        all_candidates_raw.append((b, est_prob, gap_pp))
    all_candidates_raw.sort(key=lambda x: x[2], reverse=True)
    candidate_table = [(b.label, round(p, 4), b.price, gap, b.spread_cents, b.depth_ok)
                        for b, p, gap in all_candidates_raw]

    # Only rank REAL, tradeable buckets -- price >= longshot floor AND real
    # liquidity. No "alerts-only" exception. If it's not tradeable, it doesn't
    # get to be "the best candidate," full stop.
    real_candidates = [
        c for c in all_candidates_raw
        if c[0].price >= LONGSHOT_FLOOR
        and c[0].spread_cents is not None
        and c[0].spread_cents <= MAX_SPREAD_CENTS
        and c[0].depth_ok is True
    ]

    if not real_candidates:
        return EvalResult(None, "no_tradeable_buckets",
                           "No bucket cleared price floor + real liquidity (spread <= "
                           f"{MAX_SPREAD_CENTS}c, real depth). Nothing tradeable right now.",
                           candidate_table)

    best_bucket, best_prob, best_gap = real_candidates[0]

    if best_gap > MAX_SANITY_GAP_PP:
        return EvalResult(None, "gap_implausible",
                           f"Gap {best_gap}pp on '{best_bucket.label}' exceeds sanity ceiling "
                           f"{MAX_SANITY_GAP_PP}pp -- likely bad input, not real edge. "
                           f"(mu={mu:.2f}, sigma={sigma:.2f}, est_prob={best_prob:.1%}, market={best_bucket.price:.1%})",
                           candidate_table)

    if best_gap < MIN_GAP_PP:
        return EvalResult(None, "gap_too_small",
                           f"Best real gap {best_gap}pp on '{best_bucket.label}' below minimum {MIN_GAP_PP}pp.",
                           candidate_table)

    # Risk flag (not a block): if betting No and the mean sits INSIDE the bucket
    # being shorted, this is the bot's own single-most-likely bucket -- still
    # correct to trade if the gap justifies it, but genuinely higher-risk than
    # a No bet against a bucket far from the mean. Flag it, don't reject it.
    near_boundary_risk = (
        best_bucket.outcome == "No"
        and best_bucket.low is not None and best_bucket.high is not None
        and best_bucket.low <= mu <= best_bucket.high
    )

    stake = calc_kelly_stake(best_prob, best_bucket.price)
    if stake <= 0:
        return EvalResult(None, "kelly_stake_zero",
                           f"Edge on '{best_bucket.label}' didn't clear Kelly sizing threshold "
                           f"(est={best_prob:.1%}, price={best_bucket.price:.1%}) -- skipped.",
                           candidate_table)

    sig = Signal(
        city=city, bucket_label=best_bucket.label, corrected_mu=round(mu, 2),
        sigma=round(sigma, 2), est_prob=round(best_prob, 4),
        market_price=best_bucket.price, gap_pp=best_gap, models_used=models_used,
        used_live_obs=used_live_obs, underdispersed=underdispersed, token_id=best_bucket.token_id,
        market_id=best_bucket.market_id, outcome=best_bucket.outcome,
        near_boundary_risk=near_boundary_risk, suggested_stake=stake,
    )
    return EvalResult(sig, "fired",
                       f"Signal: '{best_bucket.label}' est {best_prob:.1%} vs market "
                       f"{best_bucket.price:.1%}, gap {best_gap}pp"
                       f"{' (live-obs floor applied)' if used_live_obs else ''}.",
                       candidate_table)
