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
import time

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
    if not label:
        return None
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
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [pm] skipped market: json decode error: {e}")
            continue
        if not outcomes or not prices:
            print(f"  [pm] skipped market: no outcomes/prices, groupTitle={market.get('groupItemTitle')!r}")
            continue
        label = market.get("question", market.get("groupItemTitle", "")).strip()
        try:
            price = float(prices[0])
        except (ValueError, IndexError) as e:
            print(f"  [pm] skipped market: price parse failed for label={label!r}, prices={prices!r}: {e}")
            continue
        parsed = parse_bucket_label(label)
        if parsed is None:
            print(f"  [pm] skipped market: label parse failed: {label!r}")
            continue
        low, high = parsed
        token_id = token_ids[0] if token_ids else None
        if not token_id:
            print(f"  [pm] market has no token_id (will have no book): label={label!r}")
        # clamp price to [0,1]
        price = max(0.0, min(1.0, price))
        buckets.append({
            "label": label,
            "price": price,
            "token_id": token_id,
            "low": low,
            "high": high,
            "raw_market": market,
        })
    return buckets


def get_order_book(token_id: str, timeout: int = 10) -> dict:
    resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_spread_and_depth(token_id: str, needed_usd: float = 30.0, max_slippage_pct: float = 3.0):
    """Returns (spread_cents, depth_ok). Returns (None, False) on empty/missing book --
    never silently defaults to a fake wide spread like the old buggy version did.

    Adds defensive normalization: if the order-book prices appear to be on a 0..100
    scale instead of 0..1, we normalize them. Also prints raw best_bid/best_ask for
    suspicious spreads to aid diagnosis.
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

    try:
        raw_best_bid = float(bids[0]["price"])
        raw_best_ask = float(asks[0]["price"])
    except Exception:
        return None, False

    # Detect likely scale mismatch: if the ask price is > 2, assume book uses 0..100 scale
    scale = 1.0
    if raw_best_ask > 2.0:
        scale = 100.0

    # Normalize bids/asks
    bids_norm = []
    asks_norm = []
    for level in bids:
        try:
            bids_norm.append({"price": float(level["price"]) / scale, "size": float(level.get("size", 0.0))})
        except Exception:
            continue
    for level in asks:
        try:
            asks_norm.append({"price": float(level["price"]) / scale, "size": float(level.get("size", 0.0))})
        except Exception:
            continue

    if not bids_norm or not asks_norm:
        return None, False

    best_bid = bids_norm[0]["price"]
    best_ask = asks_norm[0]["price"]
    spread_cents = round((best_ask - best_bid) * 100, 2)

    # If spread is suspiciously large, print the raw and normalized best bid/ask to help debug
    if spread_cents > 50.0:
        print(f"  [pm-debug] token_id={token_id} raw_best_bid={raw_best_bid} raw_best_ask={raw_best_ask} scale={scale} norm_best_bid={best_bid} norm_best_ask={best_ask} spread_cents={spread_cents}")

    max_price = best_ask * (1 + max_slippage_pct / 100)
    filled = 0.0
    for level in asks_norm:
        price = level["price"]
        if price > max_price:
            break
        filled += price * level.get("size", 0.0)
        if filled >= needed_usd:
            break
    depth_ok = filled >= needed_usd
    return spread_cents, depth_ok
