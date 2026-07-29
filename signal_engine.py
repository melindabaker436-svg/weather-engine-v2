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
    all_candidates: list = None  # [(label, est_prob, market_price, gap_pp, spread_cents, depth_ok, token_id), ...] full ranked table, for visibility


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


def _liquidity_score(spread_cents, depth_ok):
    # Map spread/depth to a simple liquidity score in [0,1]
    if depth_ok is False or spread_cents is None:
        return 0.0
    try:
        sc = float(spread_cents)
    except Exception:
        return 0.0
    if sc <= MAX_SPREAD_CENTS:
        return 1.0
    if sc <= MAX_SPREAD_CENTS * 5:
        return 0.5
    return 0.2


def evaluate_buckets(city: str, raw_model_values: dict, bias_data: dict,
                      buckets: list) -> EvalResult:
    mu, sigma = correct_forecast(raw_model_values, bias_data)
    if mu is None:
        return EvalResult(None, "no_forecast_data", "No valid model readings.")

    models_used = [m for m, v in raw_model_values.items() if v is not None]

    candidates = []
    for b in buckets:
        est_prob = prob_math.bucket_probability(b.low, b.high, mu, sigma)
        # clamp and guard against NaN
        if est_prob is None or not isinstance(est_prob, (int, float)):
            est_prob = 0.0
        est_prob = max(0.0, min(1.0, est_prob))
        gap_pp = round((est_prob - b.price) * 100, 1)
        # liquidity-aware adjusted gap: prefer liquid buckets
        liq = _liquidity_score(b.spread_cents, b.depth_ok)
        adjusted_gap = round(gap_pp * liq, 3)
        candidates.append((b, est_prob, gap_pp, adjusted_gap))

    # sort by adjusted_gap desc (favor liquid, high-edge candidates)
    candidates.sort(key=lambda x: x[3], reverse=True)

    # include spread/depth/token_id in the candidate table for better diagnostics
    candidate_table = [
        (b.label, round(p, 4), b.price, gap, b.spread_cents, b.depth_ok, b.token_id, round(adjusted, 4))
        for b, p, gap, adjusted in candidates
    ]

    if not candidates:
        return EvalResult(None, "no_buckets", "No buckets to evaluate.", candidate_table)

    # Event-level stale market detection: if the market looks dead (all tiny prices), skip the event
    prices = [b.price for b, _, _, _ in candidates]
    if max(prices) < 0.02 or sum(prices) < 0.05:
        return EvalResult(None, "market_stale_snapshot",
                          "Market prices appear to be a stale snapshot (all buckets tiny/untraded).",
                          candidate_table)

    # Iterate down candidates (by adjusted gap) and pick the first that passes hard checks
    rejection_reasons = []  # (label, reason, value)
    chosen = None
    for b, p, gap, adjusted in candidates:
        # sanity gap check on raw gap
        if gap > MAX_SANITY_GAP_PP:
            rejection_reasons.append((b.label, "gap_implausible", gap))
            continue

        # longshot floor
        if b.price < LONGSHOT_FLOOR:
            rejection_reasons.append((b.label, "below_longshot_floor", b.price))
            continue

        # require some depth; if no depth we skip
        if b.depth_ok is False or b.depth_ok is None:
            rejection_reasons.append((b.label, "insufficient_depth", b.depth_ok))
            continue

        # minimum gap
        if gap < MIN_GAP_PP:
            rejection_reasons.append((b.label, "gap_too_small", gap))
            continue

        # candidate passes all checks
        chosen = (b, p, gap, adjusted)
        break

    # No viable candidate found -> build a reason
    if not chosen:
        # Determine dominant rejection reason for clearer messages
        reasons = [r for _, r, _ in rejection_reasons]
        if reasons and all(r == "below_longshot_floor" for r in reasons):
            return EvalResult(None, "longshot_floor",
                              "All candidates below longshot floor. See candidate_table for details.",
                              candidate_table)
        if any(r in ("insufficient_depth",) for r in reasons):
            return EvalResult(None, "liquidity_fail",
                              "No candidate passed liquidity checks. See candidate_table for details.",
                              candidate_table)
        # fallback
        return EvalResult(None, "no_viable_candidate",
                          "No candidate passed selection filters. See candidate_table for details.",
                          candidate_table)

    # chosen candidate -> build signal
    best_bucket, best_prob, best_gap, best_adjusted = chosen
    sig = Signal(
        city=city, bucket_label=best_bucket.label, corrected_mu=round(mu, 2),
        sigma=round(sigma, 2), est_prob=round(best_prob, 4),
        market_price=best_bucket.price, gap_pp=best_gap, models_used=models_used,
    )
    return EvalResult(sig, "fired",
                      f"Signal: '{best_bucket.label}' est {best_prob:.1%} vs market "
                      f"{best_bucket.price:.1%}, gap {best_gap}pp (adj {best_adjusted}).",
                      candidate_table)
