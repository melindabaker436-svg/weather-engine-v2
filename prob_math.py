"""
prob_math.py -- standalone module, test this in isolation before trusting it
anywhere else. Implements: given a corrected forecast (mu) and a measured
historical error (sigma), what's the probability the ACTUAL resolved temperature
rounds into each whole-degree bucket?

Real Polymarket bucket structure (confirmed from a live fetch during this
conversation, not assumed): single whole-degree buckets like "27°C", plus two
open-ended tails like "26°C or below" and "36°C or higher". NOT ranges like
"32-33°C" -- that was an incorrect assumption in earlier versions of this code.

Since the resolution source rounds to the nearest whole degree, a bucket "N°C"
covers the continuous interval [N-0.5, N+0.5). Probability mass for that bucket
under a Normal(mu, sigma) distribution is CDF(N+0.5) - CDF(N-0.5).
"""

import math


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF via math.erf (stdlib, no scipy dependency needed)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def bucket_probability(low, high, mu: float, sigma: float) -> float:
    """
    Probability mass in [low, high) under Normal(mu, sigma).
    Pass low=None for an open-ended "or below" bucket (-inf, high).
    Pass high=None for an open-ended "or higher" bucket [low, +inf).
    """
    p_high = 1.0 if high is None else normal_cdf(high, mu, sigma)
    p_low = 0.0 if low is None else normal_cdf(low, mu, sigma)
    return max(0.0, p_high - p_low)
