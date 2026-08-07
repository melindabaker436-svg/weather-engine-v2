"""
hindcast.py -- computes real per-city forecast bias and sigma from historical data.
Run ONCE (or periodically, e.g. weekly) via:
    python main.py hindcast

FIX (found via real Railway deploy logs, not assumed): the Previous Runs API does
NOT have a daily-max variant of its lead-time-offset fields. The confirmed working
field, per Open-Meteo's own docs example, is HOURLY: temperature_2m_previous_day1
(not "temperature_2m_max_previous_day1", which returned HTTP 400 in production).
This version fetches hourly data and computes each day's max locally.

Two Open-Meteo endpoints, consistent methodology, works globally:
  - Previous Runs API: what a model FORECASTED on a past date, at a fixed lead time
  - Historical Weather (archive, ERA5-based) API: what ACTUALLY happened

LEAK-FREE BY CONSTRUCTION: only ever compares a forecast made ~24h before a date
against what happened ON that date -- never lets later information leak in.
"""

import json
import datetime as dt
import statistics
import requests
from collections import defaultdict

PREVIOUS_RUNS_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"

HINDCAST_DAYS = 90
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]


def fetch_previous_run_daily_max(lat, lon, model, days: int, timeout=25):
    """
    Returns {date_str: forecasted_daily_high} for the last `days` days, using
    each hour's 24h-lead-time forecast (temperature_2m_previous_day1), grouped
    by local calendar date and reduced to a daily max locally -- there is no
    native daily-max field for previous-run offsets, confirmed against a live
    400 error in production.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m_previous_day1",
        "models": model,
        "past_days": days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = requests.get(PREVIOUS_RUNS_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("temperature_2m_previous_day1", [])

    daily_max = defaultdict(lambda: None)
    for t, v in zip(times, values):
        if v is None:
            continue
        date_str = t.split("T")[0]
        if daily_max[date_str] is None or v > daily_max[date_str]:
            daily_max[date_str] = v
    return dict(daily_max)


def fetch_historical_actual_series(lat, lon, days: int, timeout=25):
    """Returns {date_str: actual_high} for the last `days` days (ERA5 reanalysis)."""
    today = dt.date.today()
    start = (today - dt.timedelta(days=days)).isoformat()
    end = (today - dt.timedelta(days=1)).isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    values = daily.get("temperature_2m_max", [])
    return dict(zip(dates, values))


def run_hindcast_for_city(city: str, lat: float, lon: float, days: int = HINDCAST_DAYS):
    """Returns {model: {"bias": float, "sigma": float, "n": int}} for one city,
    plus "_typical_model_spread": the average day-to-day disagreement between
    the 3 models over the hindcast window. Used by the underdispersion filter
    (signal_engine.check_underdispersion): when TODAY's live inter-model spread
    is unusually tight relative to this baseline, that's a real, measurable
    confidence signal, not just a feeling."""
    try:
        actuals = fetch_historical_actual_series(lat, lon, days)
    except requests.RequestException as e:
        print(f"  [{city}] Failed to fetch actuals: {e}")
        return {m: {"bias": 0.0, "sigma": 2.5, "n": 0, "note": "actuals fetch failed"} for m in MODELS}

    results = {}
    per_model_forecasts = {}
    for model in MODELS:
        try:
            forecasts = fetch_previous_run_daily_max(lat, lon, model, days)
            per_model_forecasts[model] = forecasts
        except requests.RequestException as e:
            print(f"  [{city}] {model} fetch failed: {e}")
            results[model] = {"bias": 0.0, "sigma": 2.5, "n": 0, "note": "forecast fetch failed"}
            continue

        errors = [
            forecasts[date] - actuals[date]
            for date in forecasts
            if date in actuals and forecasts[date] is not None and actuals[date] is not None
        ]

        if len(errors) >= 5:
            results[model] = {
                "bias": round(statistics.mean(errors), 2),
                "sigma": round(statistics.pstdev(errors), 2) if len(errors) > 1 else 2.0,
                "n": len(errors),
            }
        else:
            results[model] = {"bias": 0.0, "sigma": 2.5, "n": len(errors),
                               "note": "insufficient overlapping data, using conservative default"}

    common_dates = set.intersection(*[set(v.keys()) for v in per_model_forecasts.values()]) if per_model_forecasts else set()
    daily_spreads = []
    for date in common_dates:
        vals = [per_model_forecasts[m][date] for m in per_model_forecasts if per_model_forecasts[m].get(date) is not None]
        if len(vals) >= 2:
            daily_spreads.append(max(vals) - min(vals))
    results["_typical_model_spread"] = round(statistics.mean(daily_spreads), 2) if daily_spreads else 2.0

    return results


def run_full_hindcast(cities: dict, days: int = HINDCAST_DAYS, out_path: str = "./bias_data.json"):
    print(f"Running {days}-day hindcast across {len(cities)} cities.")
    all_results = {}
    for city, cfg in cities.items():
        print(f"\n--- {city} ---")
        result = run_hindcast_for_city(city, cfg["lat"], cfg["lon"], days)

        # Print model-specific stats in the canonical MODELS order to avoid
        # accidentally iterating over summary keys (which may be non-dict values).
        for model in MODELS:
            stats = result.get(model)
            if not isinstance(stats, dict):
                # Defensive: if something unexpected is present, print it plainly.
                print(f"  {model}: {stats}")
                continue
            note = f" ({stats['note']})" if "note" in stats else ""
            print(f"  {model}: bias={stats['bias']:+.2f}, sigma={stats['sigma']:.2f}, n={stats['n']}{note}")

        # Print any additional summary keys (e.g. _typical_model_spread) separately.
        for key, val in result.items():
            if key in MODELS:
                continue
            print(f"  {key}: {val}")

        all_results[city] = result

    with open(out_path, "w") as f:
        json.dump({"generated": dt.datetime.utcnow().isoformat(), "data": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")
    return all_results


def load_bias_data(path: str = "./bias_data.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)["data"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
