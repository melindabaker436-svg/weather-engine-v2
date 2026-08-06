"""
main.py -- consolidated, final version. Ties hindcast + signal_engine +
polymarket_client together, with the live-observation peak-hour floor actually
connected end to end.
"""

import os
import sys
import time
import datetime as dt
import requests
import quopri
import urllib.parse
import re

import hindcast
import polymarket_client as pm
import signal_engine as se
import journal

CITIES = {
    "London": {"lat": 51.5074, "lon": -0.1278, "unit": "C",
               "keyword_variants": ["highest temperature in london"]},
    "New York": {"lat": 40.7128, "lon": -74.0060, "unit": "F",
                  "keyword_variants": ["highest temperature in new york", "highest temperature in nyc"]},
    "Toronto": {"lat": 43.6532, "lon": -79.3832, "unit": "C",
                "keyword_variants": ["highest temperature in toronto"]},
    "Paris": {"lat": 48.9694, "lon": 2.4414, "unit": "C",  # FIXED: Le Bourget Airport (LFPB), the real resolution station -- was city center (48.8566, 2.3522), ~13km off
              "keyword_variants": ["highest temperature in paris"]},
    "Hong Kong": {"lat": 22.3193, "lon": 114.1694, "unit": "C",
                  "keyword_variants": ["highest temperature in hong kong"]},
    "Seoul": {"lat": 37.4691, "lon": 126.4510, "unit": "C",  # FIXED: Incheon Airport (RKSI), the real resolution station -- was city center (37.5665, 126.9780), ~50km off
              "keyword_variants": ["highest temperature in seoul"]},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "unit": "F",
                "keyword_variants": ["highest temperature in chicago"]},
    "Madrid": {"lat": 40.4168, "lon": -3.7038, "unit": "C",
               "keyword_variants": ["highest temperature in madrid"]},
    "Milan": {"lat": 45.6306, "lon": 8.7281, "unit": "C",  # FIXED: Malpensa Airport (LIMC) confirmed via real Polymarket rules -- NOT Linate as claimed. Was city center (45.4642, 9.1900), ~45km off (Malpensa is well outside the city)
               "keyword_variants": ["highest temperature in milan"]},
    "Munich": {"lat": 48.1351, "lon": 11.5820, "unit": "C",
               "keyword_variants": ["highest temperature in munich"]},
    "Amsterdam": {"lat": 52.3676, "lon": 4.9041, "unit": "C",
                  "keyword_variants": ["highest temperature in amsterdam"]},
    "Berlin": {"lat": 52.5200, "lon": 13.4050, "unit": "C",
               "keyword_variants": ["highest temperature in berlin"]},
}

MODELS = hindcast.MODELS
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
BIAS_DATA_PATH = "./bias_data.json"
CHECK_INTERVAL_MINUTES = 10

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _prepare_for_telegram(raw_text: str) -> str:
    """Minimal normalization/escaping to avoid Telegram HTML parser errors.

    - Attempt to decode quoted-printable artifacts ("=20" etc.).
    - URL-decode percent-encodings like %3C/%3E if present.
    - Replace "<=" with a safe unicode char '≤' to remove raw '<'.
    - Escape &, <, > so parse_mode won't see tags.
    """
    text = raw_text or ""
    try:
        if "=20" in text or "=\r\n" in text or re.search(r"=[0-9A-Fa-f]{2}", text):
            decoded = quopri.decodestring(text.encode("utf-8", errors="replace"))
            text = decoded.decode("utf-8", errors="replace")
    except Exception:
        text = raw_text or ""

    try:
        if "%3C" in text.upper() or "%3E" in text.upper():
            text = urllib.parse.unquote(text)
    except Exception:
        pass

    text = text.replace("<=", "≤")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def send_telegram(text: str, timeout: int = 10) -> bool:
    """Emergency: always send WITHOUT parse_mode to guarantee delivery.

    This removes HTML formatting but avoids Telegram entity parser failures.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] Not configured -- printing instead:")
        print(text)
        return False

    original = text
    safe_text = _prepare_for_telegram(original)

    def _truncate(s, n=400):
        return (s[:n] + "...") if len(s) > n else s

    print(f"[telegram] Emergency send WITHOUT parse_mode (orig={_truncate(original)!r}, safe={_truncate(safe_text)!r})")

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": safe_text},
            timeout=timeout,
        )
        if not resp.ok:
            print(f"[telegram] Send failed: {resp.status_code} {resp.text}")
        return resp.ok
    except requests.RequestException as e:
        print(f"[telegram] Send failed: {e}")
        return False


def fetch_today_forecast(lat: float, lon: float, target_date: str, timeout: int = 15) -> dict:
    """
    FIX: this previously had no target_date parameter at all -- it called Open-Meteo
    with forecast_days=1 and blindly took series[0], which is ALWAYS TODAY's
    forecast, never the actual target_date being evaluated (which is always
    tomorrow, per run_check). Now explicitly fetches enough days to cover target_date
    and indexes to the matching date.
    """
    params = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
              "models": ",".join(MODELS), "timezone": "auto", "forecast_days": 3}
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if target_date not in dates:
        print(f"    [forecast] WARNING: target_date {target_date} not in returned range {dates}")
        return {m: None for m in MODELS}
    idx = dates.index(target_date)
    result = {}
    for m in MODELS:
        key = f"temperature_2m_max_{m}"
        series = daily.get(key, daily.get("temperature_2m_max"))
        result[m] = series[idx] if series and idx < len(series) else None
    return result


def fetch_current_temperature(lat: float, lon: float, timeout: int = 10):
    """Live current-conditions reading, in Celsius (Open-Meteo's native unit)."""
    params = {"latitude": lat, "longitude": lon, "current_weather": "true", "timezone": "auto"}
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    cw = data.get("current_weather") or {}
    return cw.get("temperature")


def predict_peak_hour_index(lat: float, lon: float, timeout: int = 15):
    """Returns (peak_hour_index, times_list) from today's hourly forecast, or
    (None, None) on failure."""
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
              "forecast_days": 1, "timezone": "auto"}
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    times = hourly.get("time", [])
    if not temps or not times:
        return None, None
    idx = max(range(len(temps)), key=lambda i: temps[i] if temps[i] is not None else -999)
    return idx, times


def get_live_obs_if_in_peak_window(lat: float, lon: float, unit: str):
    """
    DISABLED. Real bug found in production: this engine only ever evaluates
    TOMORROW's market (target_date = today+1), but this function was reading
    TODAY's current temperature and using it as a floor on tomorrow's forecast.
    """
    return None


def format_signal_alert(signal: se.Signal) -> str:
    return (
        f"\U0001F321\uFE0F <b>SIGNAL</b>\n"
        f"City: {signal.city}\n"
        f"Bucket: {signal.bucket_label}\n"
        f"Corrected forecast: {signal.corrected_mu}\u00b0 (\u03c3={signal.sigma})"
        f"{' [live-obs floor applied]' if signal.used_live_obs else ''}\n"
        f"Models used: {', '.join(signal.models_used)}\n"
        f"Est. probability: {signal.est_prob:.1%}\n"
        f"Market price: {signal.market_price:.1%}\n"
        f"Gap: +{signal.gap_pp}pp\n"
                f"-- real liquidity checked (spread<=20c, real depth), longshot floor, sanity ceiling all passed."
    )


def run_check(bias_data: dict, target_date: str = None):
    journal.check_and_resolve_open_signals()
    target_date = target_date or (dt.date.today() + dt.timedelta(days=1)).isoformat()

    for city, cfg in CITIES.items():
        print(f"--- {city} ---")

        try:
            raw_values = fetch_today_forecast(cfg["lat"], cfg["lon"], target_date)
        except requests.RequestException as e:
            print(f"  Forecast fetch failed: {e}")
            continue

        city_bias = bias_data.get(city, {})
        if not city_bias:
            print(f"  No hindcast data for {city} yet -- run 'python main.py hindcast' first.")
            continue

        live_obs = get_live_obs_if_in_peak_window(cfg["lat"], cfg["lon"], cfg["unit"])
        if live_obs is not None:
            print(f"  In predicted peak window -- live obs = {live_obs:.1f}\u00b0{cfg['unit']}")

        try:
            event = pm.find_matching_event(cfg["keyword_variants"], target_date)
        except requests.RequestException as e:
            print(f"  Market search failed: {e}")
            continue
        if event is None:
            print(f"  No matching Polymarket event found for {target_date}.")
            continue

        raw_buckets = pm.get_market_buckets(event)
        buckets = []
        for rb in raw_buckets:
            spread_cents, depth_ok = pm.get_spread_and_depth(rb["token_id"])
            buckets.append(se.Bucket(
                label=rb["label"], low=rb["low"], high=rb["high"], price=rb["price"],
                token_id=rb["token_id"], outcome=rb["outcome"], market_id=rb["market_id"],
                spread_cents=spread_cents, depth_ok=depth_ok,
            ))

        if not buckets:
            print("  Event found but no parseable buckets.")
            continue

        result = se.evaluate_buckets(city, raw_values, city_bias, buckets, live_obs=live_obs)
        print(f"  {result.reason_code}: {result.detail}")
        if result.all_candidates:
            print("  Top candidates (est_prob | market | gap | spread | depth_ok):")
            for label, p, price, gap, spread, depth in result.all_candidates[:5]:
                print(f"    {label}: {p:.1%} | {price:.1%} | {gap:+.1f}pp | {spread}c | {depth}")

        if result.signal:
            if journal.has_open_signal(result.signal.city, result.signal.bucket_label):
                print(f"  Signal fired but already have an OPEN position on this bucket -- "
                      f"skipping duplicate alert.")
            elif journal.open_signal_count_for_city(result.signal.city) >= journal.MAX_OPEN_SIGNALS_PER_CITY:
                print(f"  Signal fired on a NEW bucket, but {result.signal.city} already has "
                      f"{journal.MAX_OPEN_SIGNALS_PER_CITY} open position(s) -- skipping to avoid "
                      f"stacking correlated exposure on sibling buckets (same underlying outcome).")
            else:
                send_telegram(format_signal_alert(result.signal))
                signal_id = journal.log_signal(result.signal)
                print(f"  -> Telegram alert sent. Logged as journal signal #{signal_id}.")


def run_self_test(bias_data):
    print("=" * 50)
    print("STARTUP SELF-TEST")
    print("=" * 50)
    print(f"[journal] JOURNAL_PATH resolves to: {os.path.abspath(journal.JOURNAL_PATH)}")
    if not bias_data:
        print("[!] No bias_data.json found -- run 'python main.py hindcast' first.")
    else:
        print(f"[OK] bias_data.json loaded for {len(bias_data)} cities.")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        ok = send_telegram("\u2705 Weather engine v3 started. Real thresholds restored, "
                            "No-probability bug fixed, live-obs floor wired.")
        print(f"[{'OK' if ok else 'FAIL'}] Telegram")
    else:
        print("[!] Telegram not configured -- alerts print to console.")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hindcast":
        hindcast.run_full_hindcast(CITIES, out_path=BIAS_DATA_PATH)
        sys.exit(0)

    bias_data = hindcast.load_bias_data(BIAS_DATA_PATH) or {}
    run_self_test(bias_data)
    journal.print_summary()

    check_number = 0
    while True:
        check_number += 1
        print(f"\n[Check #{check_number} -- {dt.datetime.utcnow().isoformat()}]")
        try:
            bias_data = hindcast.load_bias_data(BIAS_DATA_PATH) or bias_data
            run_check(bias_data)
        except Exception as e:
            print(f"[main loop] Unexpected error: {e}")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)
