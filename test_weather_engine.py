import math

import polymarket_client as pm
import prob_math
import signal_engine as se


def test_station_bucket_parsing_is_celsius_consistent():
    low, high = pm.parse_bucket_label(
        "Will the highest temperature in NYC be 88-89°F on August 20?"
    )
    assert math.isclose(low, (87.5 - 32) * 5 / 9)
    assert math.isclose(high, (89.5 - 32) * 5 / 9)


def test_no_is_complement_of_same_bucket():
    bucket = se.Bucket("27°C [No]", 26.5, 27.5, 0.70, "token", outcome="No")
    yes = prob_math.bucket_probability(26.5, 27.5, 20.0, 2.0)
    assert math.isclose(se._bucket_est_prob(bucket, 20.0, 2.0), 1 - yes)


def test_calibration_policy_rejects_expensive_no():
    bucket = se.Bucket("27°C [No]", 26.5, 27.5, 0.90, "token",
                       outcome="No", spread_cents=1, depth_ok=True)
    result = se.evaluate_buckets(
        "London",
        {"model": 20.0},
        {"model": {"bias": 0.0, "sigma": 2.0}},
        [bucket],
    )
    assert result.signal is None
    assert result.reason_code == "no_tradeable_buckets"