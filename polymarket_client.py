"""
polymarket_client.py -- consolidated, final version.

FIXES (both confirmed from real production evidence, not guesses):
1. DROPPED scale normalization. Copilot added this as a hypothesis for the 98c
   spread bug, but every single [pm-debug] line in the real logs showed scale=1.0
   -- the hypothesis was tested against real data and ruled out. Keeping unused
   defensive code around is its own risk; removed.
2. FIXED the real bug: each market (one temperature threshold) has a "Yes" and
   "No" outcome, each with its own price and token_id. The old code took
   prices[0]/token_ids[0] only (Yes), or later, an index-based rewrite created
   both but with duplicated probability. This version explicitly creates one
   Bucket per outcome (Yes AND No) with the correct token_id/price pairing, and
   tags which outcome it is so signal_engine can compute No = 1 - Yes correctly.
"""

import json
import re
import datetime as dt
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_OR_BELOW_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*[CF]?\s*or below", re.IGNORECASE)
_OR_HIGHER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*[CF]?\s*or higher", re.IGNORECASE)
_RANGE_RE = re.compile(r"between\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*°\s*([CF])", re.IGNORECASE)
_SINGLE_RE = re.compile(r"(?<![\d-])(-?\d+(?:\.\d+)?)\s*°\s*([CF])(?!\w)", re.IGNORECASE)


def find_events_by_keywords(keywords: list, timeout: int = 15) -> list:
    for kw in keywords:
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/public-search", params={"q": kw},
                headers={"User-Agent": "weather-v3/1.0"}, timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if events:
                return events
        except requests.RequestException as e:
            print(f"  [find_events_by_keywords] search failed for '{kw}': {e}")
    return []


def event_matches_date(event: dict, target_date: str) -> bool:
    end_date_raw = event.get("endDate") or event.get("end_date")
    if end_date_raw:
        try:
            parsed = dt.datetime.fromisoformat(end_date_raw.replace("Z", "+00:00"))
            if parsed.date().isoformat() == target_date:
                return True
        except (ValueError, AttributeError):
            pass
    title = (event.get("title") or event.get("question") or "").lower()
    target = dt.date.fromisoformat(target_date)
    candidates = [target.strftime("%B %-d").lower(), target.strftime("%b %-d").lower(), str(target.day)]
    return any(c in title for c in candidates)


def find_matching_event(keywords: list, target_date: str) -> dict:
    events = find_events_by_keywords(keywords)
    for ev in events:
        if event_matches_date(ev, target_date):
            return ev
    return None


def parse_bucket_label(label: str):
    """
    Returns (low, high). None/None edges for open-ended tails.

    FIX: "between 84-85°F" style range buckets were being misparsed by the
    single-value regex, which matched "-85" as a NEGATIVE 85 (reading the range
    dash as a minus sign) instead of recognizing "84-85" as a range. This
    produced impossible bucket bounds (~-85°F on a July day), which silently
    broke every probability calculation for NYC/Chicago (Yes computed near 0%
    against the bogus negative range -> No came out near 100% on EVERYTHING).
    Range format is now checked FIRST and explicitly, before any single-value
    fallback can misfire on it.
    """
    m = _RANGE_RE.search(label)
    if m:
        low_val, high_val = float(m.group(1)), float(m.group(2))
        return low_val - 0.5, high_val + 0.5  # e.g. "84-85" -> [83.5, 85.5)

    m = _OR_BELOW_RE.search(label)
    if m:
        val = float(m.group(1))
        return None, val + 0.5

    m = _OR_HIGHER_RE.search(label)
    if m:
        val = float(m.group(1))
        return val - 0.5, None

    m = _SINGLE_RE.search(label)
    if m:
        val = float(m.group(1))
        return val - 0.5, val + 0.5

    return None


def get_market_buckets(event: dict) -> list:
    """
    Returns one dict per OUTCOME (Yes and No, separately) per market/temperature
    threshold. Each dict: label, price, token_id, low, high, outcome ('Yes'/'No').
    This is the corrected version -- explicit index pairing, no assumptions about
    array order beyond "outcomes[i] corresponds to prices[i] corresponds to
    token_ids[i]", which is what Gamma's API contract actually guarantees.
    """
    buckets = []
    for market in event.get("markets", []):
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            prices = json.loads(market.get("outcomePrices", "[]"))
            token_ids = json.loads(market.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        if not (len(outcomes) == len(prices) == len(token_ids)):
            print(f"  [pm] array length mismatch for market {market.get('id', '?')}: "
                  f"outcomes={len(outcomes)} prices={len(prices)} token_ids={len(token_ids)} -- skipping")
            continue

        question_label = market.get("question", market.get("groupItemTitle", ""))
        parsed = parse_bucket_label(question_label)
        if parsed is None:
            continue
        low, high = parsed

        for i, outcome_name in enumerate(outcomes):
            try:
                price = float(prices[i])
            except (ValueError, IndexError):
                continue
            token_id = token_ids[i] if i < len(token_ids) else None
            buckets.append({
                "label": f"{question_label} [{outcome_name}]",
                "price": price, "token_id": token_id,
                "low": low, "high": high,
                "outcome": outcome_name,  # 'Yes' or 'No' -- signal_engine uses this
            })
    return buckets


def get_order_book(token_id: str, timeout: int = 10) -> dict:
    resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_spread_and_depth(token_id: str, needed_usd: float = 30.0, max_slippage_pct: float = 3.0):
    """
    Returns (spread_cents, depth_ok). (None, False) on empty/missing book.

    FIX (confirmed via Polymarket/polymarket-cli GitHub issue #75, not a guess):
    the CLOB /book endpoint returns bids in ASCENDING price order (lowest first)
    and asks in DESCENDING price order (highest first) -- the opposite of the
    intuitive assumption. Best bid and best ask are both at the END of their
    respective lists, not the start. Using bids[0]/asks[0] (the old code, in
    every prior version of this project) computes (worst_ask - worst_bid), which
    lands near 98-99c on almost any real book -- exactly the pattern seen in
    production across every city for weeks. This was never a liquidity problem.
    """
    if not token_id:
        return None, False
    try:
        book = get_order_book(token_id)
    except requests.RequestException:
        return None, False
    bids, asks = book.get("bids", []), book.get("asks", [])
    if not bids or not asks:
        return None, False

    best_bid = float(bids[-1]["price"])   # ascending order -> best (highest) bid is LAST
    best_ask = float(asks[-1]["price"])   # descending order -> best (lowest) ask is LAST
    spread_cents = round((best_ask - best_bid) * 100, 2)

    # Walk the ask side from the BEST price outward -- asks arrive worst-first,
    # so we must reverse to accumulate depth starting from the best price.
    max_price = best_ask * (1 + max_slippage_pct / 100)
    filled = 0.0
    for level in reversed(asks):
        price = float(level["price"])
        if price > max_price:
            break
        filled += price * float(level["size"])
        if filled >= needed_usd:
            break
    depth_ok = filled >= needed_usd
    return spread_cents, depth_ok
