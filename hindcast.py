"""
hindcast.py -- computes real per-city forecast bias and sigma from historical data.
Run ONCE (or periodically, e.g. weekly) via:
    python main.py hindcast

Two Open-Meteo endpoints, consistent methodology, works globally:
  - Previous Runs API: what a model FORECASTED on a past date, at a fixed lead time
  - Historical Weather (archive, ERA5-based) API: what ACTUALLY happened

LEAK-FREE BY CONSTRUCTION: only ever compares a forecast made ~LEAD_HOURS before a
date against what happened ON that date -- never lets later information leak in.

EFFICIENT BY CONSTRUCTION: each endpoint is called ONCE per city per model for the
whole historical window, not once per day -- the API returns the full date range
in a single response; we index into it locally rather than re-fetching per date.
"""

import json
import datetime as dt
import statistics
import requests

PREVIOUS_RUNS_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"

HINDCAST_DAYS = 90
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]


def fetch_previous_run_series(lat, lon, model, days: int, timeout=20):
    """
    Returns {date_str: forecasted_high} for the last `days` days, using each
    date's ~24h-lead-time forecast (the "previous_day1" field -- the forecast
    made one day before, not same-day or the final most-updated version).
    """
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max_previous_day1",
        "models": model,
        "past_days": days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = requests.get(PREVIOUS_RUNS_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    values = daily.get("temperature_2m_max_previous_day1", [])
    return dict(zip(dates, values))


def fetch_historical_actual_series(lat, lon, days: int, timeout=20):
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
    """Returns {model: {"bias": float, "sigma": float, "n": int}} for one city.
    bias = mean(forecast - actual). Positive = model runs WARM, negative = COLD."""
    try:
        actuals = fetch_historical_actual_series(lat, lon, days)
    except requests.RequestException as e:
        print(f"  [{city}] Failed to fetch actuals: {e}")
        return {m: {"bias": 0.0, "sigma": 2.5, "n": 0, "note": "actuals fetch failed"} for m in MODELS}

    results = {}
    for model in MODELS:
        try:
            forecasts = fetch_previous_run_series(lat, lon, model, days)
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
    return results


def run_full_hindcast(cities: dict, days: int = HINDCAST_DAYS, out_path: str = "./bias_data.json"):
    print(f"Running {days}-day hindcast across {len(cities)} cities "
          f"({len(MODELS) + 1} API calls per city -- fast, no per-day looping).")
    all_results = {}
    for city, cfg in cities.items():
        print(f"\n--- {city} ---")
        result = run_hindcast_for_city(city, cfg["lat"], cfg["lon"], days)
        for model, stats in result.items():
            note = f" ({stats['note']})" if "note" in stats else ""
            print(f"  {model}: bias={stats['bias']:+.2f}, sigma={stats['sigma']:.2f}, n={stats['n']}{note}")
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
