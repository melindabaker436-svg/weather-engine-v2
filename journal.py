"""
journal.py -- lightweight paper-trade tracking for weather_v3.

Logs every fired signal, then each cycle checks whether that SPECIFIC market has
actually resolved (via Gamma's GET /markets/{id}, checking the real "closed"
field -- not just guessing from price proximity to 0/1, since a near-certain
outcome can trade at 0.98 hours before the day even ends without being resolved
yet). Computes real P&L against a fixed nominal stake once resolved.

No real order is ever placed by this engine -- this answers "would this have
made money," not "did I actually buy anything."

PERSISTENCE WARNING: Railway's default filesystem is ephemeral -- wiped on every
redeploy. Given how often this project has been redeployed, this is a real risk,
not theoretical. Attach a Railway Volume and set JOURNAL_PATH to its mount path,
or this P&L history will vanish on the next push.
"""

import csv
import os
import datetime as dt
import polymarket_client as pm

JOURNAL_PATH = os.environ.get("JOURNAL_PATH", "./journal.csv")
NOMINAL_STAKE_USD = 30.0

FIELDNAMES = [
    "signal_id", "logged_at_utc", "city", "bucket_label", "outcome",
    "token_id", "market_id", "entry_price", "est_prob", "gap_pp",
    "status", "resolved_at_utc", "won", "pnl_usd",
]


def _ensure_file():
    dirpath = os.path.dirname(JOURNAL_PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if not os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print(f"  [journal] Created new journal at: {os.path.abspath(JOURNAL_PATH)}")


def has_open_signal(city: str, bucket_label: str) -> bool:
    """
    FIX: prevents the exact spam seen in production -- the same signal firing
    (and sending a new Telegram alert) on every single check cycle. Before
    logging a new signal, check whether an OPEN entry already exists for this
    same city+bucket -- if so, it's not a new opportunity, it's the same
    still-open position being re-detected.
    """
    _ensure_file()
    try:
        with open(JOURNAL_PATH, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"  [journal] has_open_signal check failed, assuming no open signal: {e}")
        return False
    return any(r["status"] == "open" and r["city"] == city and r["bucket_label"] == bucket_label
               for r in rows)


def _next_id() -> str:
    _ensure_file()
    with open(JOURNAL_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    return f"{len(rows) + 1:04d}"


def log_signal(signal) -> str:
    """Call this right after a signal fires. Returns the signal_id, or 'FALLBACK'
    if the write failed (in which case the full row is printed to stdout so the
    signal isn't silently lost -- per the diagnostic gap Copilot correctly flagged)."""
    _ensure_file()
    signal_id = _next_id()
    row = {
        "signal_id": signal_id,
        "logged_at_utc": dt.datetime.utcnow().isoformat(),
        "city": signal.city,
        "bucket_label": signal.bucket_label,
        "outcome": signal.outcome,
        "token_id": signal.token_id or "",
        "market_id": signal.market_id or "",
        "entry_price": signal.market_price,
        "est_prob": signal.est_prob,
        "gap_pp": signal.gap_pp,
        "status": "open",
        "resolved_at_utc": "",
        "won": "",
        "pnl_usd": "",
    }
    try:
        with open(JOURNAL_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        return signal_id
    except Exception as e:
        print(f"  [journal] FAILED to write to {JOURNAL_PATH}: {e}")
        print(f"  [journal] FALLBACK -- signal data (not persisted): {row}")
        return "FALLBACK"


def _resolve_one(row: dict) -> dict:
    """Checks one open row against the real market. Returns the updated row,
    unchanged if not yet resolved or if the market_id is missing/lookup fails."""
    market_id = row.get("market_id")
    if not market_id:
        return row  # can't check without a market_id (older/malformed row)

    try:
        market = pm.get_market_by_id(market_id)
    except Exception as e:
        print(f"  [journal] resolution check failed for signal {row['signal_id']}: {e}")
        return row

    if not market.get("closed"):
        return row  # still open, nothing to do yet

    import json
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        prices = json.loads(market.get("outcomePrices", "[]"))
    except (ValueError, TypeError):
        return row

    our_outcome = row["outcome"]
    final_price = None
    for name, price in zip(outcomes, prices):
        if name == our_outcome:
            final_price = float(price)
            break
    if final_price is None:
        return row

    won = final_price >= 0.5  # resolved outcomes settle to ~1.0 (won) or ~0.0 (lost)
    entry_price = float(row["entry_price"])
    stake = NOMINAL_STAKE_USD

    if won:
        shares = stake / entry_price if entry_price > 0 else 0
        pnl = round(shares * 1.0 - stake, 2)  # payout is $1/share when won
    else:
        pnl = round(-stake, 2)

    row["status"] = "resolved"
    row["resolved_at_utc"] = dt.datetime.utcnow().isoformat()
    row["won"] = "Yes" if won else "No"
    row["pnl_usd"] = pnl
    print(f"  [journal] signal {row['signal_id']} ({row['city']}) RESOLVED: "
          f"{'WON' if won else 'LOST'}, P&L ${pnl}")
    return row


def check_and_resolve_open_signals():
    """Call once per check cycle. Updates any open row whose market has resolved."""
    _ensure_file()
    with open(JOURNAL_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    open_rows = [r for r in rows if r["status"] == "open"]
    if not open_rows:
        return

    updated = {r["signal_id"]: _resolve_one(r) for r in open_rows}
    for i, row in enumerate(rows):
        if row["signal_id"] in updated:
            rows[i] = updated[row["signal_id"]]

    with open(JOURNAL_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def compute_summary() -> dict:
    _ensure_file()
    with open(JOURNAL_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    resolved = [r for r in rows if r["status"] == "resolved"]
    open_count = sum(1 for r in rows if r["status"] == "open")
    wins = [r for r in resolved if r["won"] == "Yes"]
    total_pnl = sum(float(r["pnl_usd"]) for r in resolved) if resolved else 0.0
    win_rate = len(wins) / len(resolved) if resolved else 0.0

    return {
        "total_signals": len(rows),
        "open": open_count,
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(resolved) - len(wins),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_pnl_usd": round(total_pnl, 2),
    }


def print_summary():
    s = compute_summary()
    print("=" * 50)
    print("P&L SUMMARY (nominal $%.0f stake per signal, no real orders placed)" % NOMINAL_STAKE_USD)
    print("=" * 50)
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("=" * 50)
