"""
signal_engine.py -- v3, the actual fix. Replaces the flat "N models agree = fixed
77.5% confidence on one bucket" heuristic with real bias-corrected, sigma-spread
probability. This is the piece responsible for the broken 77.5pp gaps.

Also implements the two hard filters agreed on:
  - Longshot floor: never buy anything priced under 10 cents (sourced: Whelan et
    al. 2026, contracts under 10c lose >60% of stake on average)
  - Sanity gap ceiling: discard any gap over 45pp as almost certainly a broken
    input, not a real edge
"""

from dataclasses import dataclass
from typing import Optional
import prob_math

LONGSHOT_FLOOR = 0.10
MAX_SANITY_GAP_PP = 45
MIN_GAP_PP = 12
MAX_SPREAD_CENTS = 20


@dataclass
class Bucket:
    label: str
    low: Optional[float]
    high: Optional[float]
    price: float
    token_id: Optional[str]
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


@dataclass
class EvalResult:
    signal: Optional[Signal]
    reason_code: str
    detail: str
    all_candidates: list = None  # [(label, est_prob, market_price, gap_pp), ...] full ranked table, for visibility


def correct_forecast(raw_model_values: dict, bias_data: dict) -> tuple:
    corrected_values = []
    sigmas = []
    for model, raw_value in raw_model_values.items():
        if raw_value is None:
            continue
        stats = bias_data.get(model, {"bias": 0.0, "sigma": 2.5})
        corrected = raw_value - stats["bias"]
        corrected_values.append(corrected)
        sigmas.append(stats["sigma"])

    if not corrected_values:
        return None, None

    mu = sum(corrected_values) / len(corrected_values)
    sigma = max(sigmas)
    return mu, sigma


def evaluate_buckets(city: str, raw_model_values: dict, bias_data: dict,
                      buckets: list) -> EvalResult:
    mu, sigma = correct_forecast(raw_model_values, bias_data)
    if mu is None:
        return EvalResult(None, "no_forecast_data", "No valid model readings.")

    models_used = [m for m, v in raw_model_values.items() if v is not None]

    candidates = []
    for b in buckets:
        est_prob = prob_math.bucket_probability(b.low, b.high, mu, sigma)
        gap_pp = round((est_prob - b.price) * 100, 1)
        candidates.append((b, est_prob, gap_pp))

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidate_table = [(b.label, round(p, 4), b.price, gap) for b, p, gap in candidates]

    if not candidates:
        return EvalResult(None, "no_buckets", "No buckets to evaluate.", candidate_table)

    best_bucket, best_prob, best_gap = candidates[0]

    if best_bucket.price < LONGSHOT_FLOOR:
        return EvalResult(None, "longshot_floor",
                           f"Best candidate '{best_bucket.label}' priced at {best_bucket.price:.2f}, "
                           f"below the {LONGSHOT_FLOOR} longshot floor -- skipped regardless of edge. "
                           f"NOTE: buckets priced near 0.00 are often untraded/stale snapshots, not real "
                           f"live prices -- check all_candidates for what's happening on liquid buckets.",
                           candidate_table)

    if best_gap > MAX_SANITY_GAP_PP:
        return EvalResult(None, "gap_implausible",
                           f"Gap {best_gap}pp on '{best_bucket.label}' exceeds sanity ceiling "
                           f"{MAX_SANITY_GAP_PP}pp -- likely bad input, not real edge. "
                           f"(mu={mu:.2f}, sigma={sigma:.2f}, est_prob={best_prob:.1%}, market={best_bucket.price:.1%})",
                           candidate_table)

    if best_gap < MIN_GAP_PP:
        return EvalResult(None, "gap_too_small",
                           f"Best gap {best_gap}pp below minimum {MIN_GAP_PP}pp.", candidate_table)

    if best_bucket.spread_cents is None or best_bucket.spread_cents > MAX_SPREAD_CENTS:
        return EvalResult(None, "liquidity_fail",
                           f"Spread {best_bucket.spread_cents}c on '{best_bucket.label}' exceeds "
                           f"max {MAX_SPREAD_CENTS}c (or no real book).", candidate_table)
    if best_bucket.depth_ok is False:
        return EvalResult(None, "liquidity_fail", f"Insufficient depth on '{best_bucket.label}'.", candidate_table)

    sig = Signal(
        city=city, bucket_label=best_bucket.label, corrected_mu=round(mu, 2),
        sigma=round(sigma, 2), est_prob=round(best_prob, 4),
        market_price=best_bucket.price, gap_pp=best_gap, models_used=models_used,
    )
    return EvalResult(sig, "fired",
                       f"Signal: '{best_bucket.label}' est {best_prob:.1%} vs market "
                       f"{best_bucket.price:.1%}, gap {best_gap}pp.", candidate_table)
