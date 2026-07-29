"""
main.py -- ties together hindcast.py, signal_engine.py, polymarket_client.py.

Modes:
    python main.py hindcast   -- runs the 90-day bias/sigma calibration once,
                                  saves to bias_data.json. RUN THIS FIRST.
    python main.py            -- runs the live signal-scanning loop.
"""

import os
import sys
import time
import datetime as dt
import requests

import hindcast
import polymarket_client as pm
import signal_engine as se

CITIES = {
    "London": {"lat": 51.5074, "lon": -0.1278,
               "keyword_variants": ["highest temperature in london"]},
    "New York": {"lat": 40.7128, "lon": -74.0060,
                 "keyword_variants": ["highest temperature in new york", "highest temperature in nyc"]},
    "Toronto": {"lat": 43.6532, "lon": -79.3832,
                "keyword_variants": ["highest temperature in toronto"]},
    "Paris": {"lat": 48.8566, "lon": 2.3522,
              "keyword_variants": ["highest temperature in paris"]},
    "Hong Kong": {"lat": 22.3193, "lon": 114.1694,
                  "keyword_variants": ["highest temperature in hong kong"]},
    "Seoul": {"lat": 37.5665, "lon": 126.9780,
              "keyword_variants": ["highest temperature in seoul"]},
    "Chicago": {"lat": 41.8781, "lon": -87.6298,
                "keyword_variants": ["highest temperature in chicago"]},
}

MODELS = hindcast.MODELS
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
BIAS_DATA_PATH = "./bias_data.json"
CHECK_INTERVAL_MINUTES = 10

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(text: str, timeout: int = 10) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] Not configured -- printing instead:")
        print(text)
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=timeout,
        )
        if not resp.ok:
            print(f"[telegram] Send failed: {resp.status_code} {resp.text}")
        return resp.ok
    except requests.RequestException as e:
        print(f"[telegram] Send failed: {e}")
        return False


def fetch_today_forecast(lat: float, lon: float, timeout: int = 15) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "models": ",".join(MODELS),
        "timezone": "auto", "forecast_days": 1,
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    result = {}
    for m in MODELS:
        key = f"temperature_2m_max_{m}"
        series = daily.get(key, daily.get("temperature_2m_max"))
        result[m] = series[0] if series else None
    return result


def format_signal_alert(signal: se.Signal) -> str:
    return (
        f"\U0001F321\uFE0F <b>SIGNAL</b>\n"
        f"City: {signal.city}\n"
        f"Bucket: {signal.bucket_label}\n"
        f"Corrected forecast: {signal.corrected_mu}\u00b0 (\u03c3={signal.sigma})\n"
        f"Models used: {', '.join(signal.models_used)}\n"
        f"Est. probability: {signal.est_prob:.1%}\n"
        f"Market price: {signal.market_price:.1%}\n"
        f"Gap: +{signal.gap_pp}pp\n"
        f"-- bias-corrected, longshot floor + sanity ceiling + liquidity checks passed."
    )


def run_check(bias_data: dict, target_date: str = None):
    target_date = target_date or (dt.date.today() + dt.timedelta(days=1)).isoformat()

    for city, cfg in CITIES.items():
        print(f"--- {city} ---")

        try:
            raw_values = fetch_today_forecast(cfg["lat"], cfg["lon"])
        except requests.RequestException as e:
            print(f"  Forecast fetch failed: {e}")
            continue

        city_bias = bias_data.get(city, {})
        if not city_bias:
            print(f"  No hindcast data for {city} yet -- run 'python main.py hindcast' first.")
            continue

        event = pm.find_matching_event(cfg["keyword_variants"], target_date)
        if event is None:
            print(f"  No matching Polymarket event found for {target_date}.")
            continue

        raw_buckets = pm.get_market_buckets(event)
        buckets = []
        for rb in raw_buckets:
            spread_cents, depth_ok = pm.get_spread_and_depth(rb["token_id"])
            buckets.append(se.Bucket(
                label=rb["label"], low=rb["low"], high=rb["high"], price=rb["price"],
                token_id=rb["token_id"], spread_cents=spread_cents, depth_ok=depth_ok,
            ))

        if not buckets:
            print("  Event found but no parseable buckets.")
            continue

        # Evaluate once, then if suspicious tiny prices are detected try a single quick re-fetch
        result = se.evaluate_buckets(city, raw_values, city_bias, buckets)
        print(f"  {result.reason_code}: {result.detail}")

        # If candidate table exists, detect very small market prices that might be stale/untraded
        def _min_price_from_candidate_table(ct):
            prices = []
            if not ct:
                return None
            for row in ct:
                if len(row) > 2 and isinstance(row[2], (int, float)):
                    prices.append(row[2])
            return min(prices) if prices else None

        min_price = _min_price_from_candidate_table(result.all_candidates)
        # Retry conditions: very small min price and we got a stale/longshot/liquidity/no_viable result
        if min_price is not None and min_price < 0.01 and result.reason_code in (
                "longshot_floor", "market_stale_snapshot", "no_viable_candidate", "liquidity_fail"):
            print("  [refetch] Detected very small market prices (<1%), re-fetching market data and retrying once...")
            try:
                time.sleep(1)
                raw_buckets = pm.get_market_buckets(event)
                buckets = []
                for rb in raw_buckets:
                    spread_cents, depth_ok = pm.get_spread_and_depth(rb["token_id"])
                    buckets.append(se.Bucket(
                        label=rb["label"], low=rb["low"], high=rb["high"], price=rb["price"],
                        token_id=rb["token_id"], spread_cents=spread_cents, depth_ok=depth_ok,
                    ))
                result_retry = se.evaluate_buckets(city, raw_values, city_bias, buckets)
                print(f"  Retry -> {result_retry.reason_code}: {result_retry.detail}")
                result = result_retry
            except Exception as e:
                print(f"  [refetch] Retry failed: {e}")

        if result.all_candidates:
            print("  Full candidate table (est_prob | market | gap | spread_cents | depth_ok):")
            for row in result.all_candidates[:5]:
                # support both old 4-tuple and new 6-tuple formats
                label = row[0]
                p = row[1] if len(row) > 1 else None
                price = row[2] if len(row) > 2 else None
                gap = row[3] if len(row) > 3 else None
                spread = row[4] if len(row) > 4 else None
                depth_ok = row[5] if len(row) > 5 else None
                p_str = f"{p:.1%}" if isinstance(p, (int, float)) else "N/A"
                price_str = f"{price:.1%}" if isinstance(price, (int, float)) else "N/A"
                gap_str = f"{gap:+.1f}pp" if isinstance(gap, (int, float)) else "N/A"
                print(f"    {label}: {p_str} | {price_str} | {gap_str} | spread={spread}c | depth_ok={depth_ok}")

        if result.signal:
            send_telegram(format_signal_alert(result.signal))
            print("  -> Telegram alert sent.")


def run_self_test(bias_data):
    print("=" * 50)
    print("STARTUP SELF-TEST")
    print("=" * 50)
    if not bias_data:
        print("[!] No bias_data.json found -- run 'python main.py hindcast' before "
              "trusting any signals.")
    else:
        print(f"[OK] bias_data.json loaded for {len(bias_data)} cities.")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        ok = send_telegram("\u2705 Weather engine v2 started. Bias-corrected, longshot floor active.")
        print(f"[{'OK' if ok else 'FAIL'}] Telegram")
    else:
        print("[!] Telegram not configured -- alerts will print to console only.")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hindcast":
        hindcast.run_full_hindcast(CITIES, out_path=BIAS_DATA_PATH)
        sys.exit(0)

    bias_data = hindcast.load_bias_data(BIAS_DATA_PATH) or {}
    run_self_test(bias_data)

    check_number = 0
    while True:
        check_number += 1
        # use timezone-aware UTC to avoid deprecation warning
        print(f"\n[Check #{check_number} -- {dt.datetime.now(dt.timezone.utc).isoformat()}]")
        try:
            bias_data = hindcast.load_bias_data(BIAS_DATA_PATH) or bias_data
            run_check(bias_data)
        except Exception as e:
            print(f"[main loop] Unexpected error: {e}")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)
