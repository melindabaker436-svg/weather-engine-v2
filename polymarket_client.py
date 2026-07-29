"""
polymarket_client.py -- standalone, self-contained.

FIX: bucket parsing now matches the REAL structure confirmed by a live fetch
during this conversation: single whole-degree buckets like "27C", plus two
open-ended tails like "26C or below" / "36C or higher".
"""

import json
import re
import datetime as dt
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_OR_BELOW_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*[CF]?\s*or below", re.IGNORECASE)
_OR_HIGHER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*[CF]?\s*or higher", re.IGNORECASE)
_SINGLE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([CF])(?!\w)", re.IGNORECASE)


def find_events_by_keywords(keywords: list, timeout: int = 15) -> list:
    for kw in keywords:
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/public-search", params={"q": kw},
                headers={"User-Agent": "weather-v2/1.0"}, timeout=timeout,
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
    Returns (low, high). low/high are None for open-ended tails. For a
    single-degree bucket "N", returns (N-0.5, N+0.5).
    """
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
    buckets = []
    for market in event.get("markets", []):
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            prices = json.loads(market.get("outcomePrices", "[]"))
            token_ids = json.loads(market.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not outcomes or not prices:
            continue
        label = market.get("question", market.get("groupItemTitle", ""))
        try:
            price = float(prices[0])
        except (ValueError, IndexError):
            continue
        parsed = parse_bucket_label(label)
        if parsed is None:
            continue
        low, high = parsed
        token_id = token_ids[0] if token_ids else None
        buckets.append({"label": label, "price": price, "token_id": token_id, "low": low, "high": high})
    return buckets


def get_order_book(token_id: str, timeout: int = 10) -> dict:
    resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_spread_and_depth(token_id: str, needed_usd: float = 30.0, max_slippage_pct: float = 3.0):
    """Returns (spread_cents, depth_ok). Returns (None, False) on empty/missing book --
    never silently defaults to a fake wide spread like the old buggy version did."""
    if not token_id:
        return None, False
    try:
        book = get_order_book(token_id)
    except requests.RequestException:
        return None, False
    bids, asks = book.get("bids", []), book.get("asks", [])
    if not bids or not asks:
        return None, False

    best_bid, best_ask = float(bids[0]["price"]), float(asks[0]["price"])
    spread_cents = round((best_ask - best_bid) * 100, 2)

    max_price = best_ask * (1 + max_slippage_pct / 100)
    filled = 0.0
    for level in asks:
        price = float(level["price"])
        if price > max_price:
            break
        filled += price * float(level["size"])
        if filled >= needed_usd:
            break
    depth_ok = filled >= needed_usd
    return spread_cents, depth_ok
