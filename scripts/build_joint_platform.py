#!/usr/bin/env python3
"""Fetch live data from the 3 Alpaca paper accounts and rebuild the joint
portfolio dashboard, for publishing on GitHub Pages via a scheduled Action.

Credentials come from environment variables (GitHub Actions secrets), never
from a committed file:

    ALPACA_TRADER_KEY_ID / ALPACA_TRADER_SECRET
    FINTECH_TRADER_KEY_ID / FINTECH_TRADER_SECRET
    AGGRESSIVE_CHALLENGER_TRADER_KEY_ID / AGGRESSIVE_CHALLENGER_TRADER_SECRET

For a local test run, export the same 6 vars in your shell, or create a
`.env` file next to this script (KEY=VALUE per line, gitignored) — either
works.

Writes docs/trading/joint_portfolio_platform.html by default (override with
--out). Same data model and the same live-vs-daily-close fix as the version
that runs against /Users/anon/Documents/traders locally — see that folder's
joint_platform/README.md for the full write-up of that bug.

Every call is a read-only GET: /v2/account, /v2/positions, /v2/orders,
/v2/account/portfolio/history. Never places, amends, or cancels an order.
"""
from __future__ import annotations
import argparse, json, os, sys, datetime as dt, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sector_map as SM

BOOKS = ["alpaca_trader", "fintech_trader", "aggressive_challenger_trader"]
LABELS = {
    "alpaca_trader": "Alpaca Trader",
    "fintech_trader": "Fintech Trader",
    "aggressive_challenger_trader": "Aggressive Challenger",
}
ENV_PREFIX = {
    "alpaca_trader": "ALPACA_TRADER",
    "fintech_trader": "FINTECH_TRADER",
    "aggressive_challenger_trader": "AGGRESSIVE_CHALLENGER_TRADER",
}


def load_dotenv_fallback():
    """Optional local convenience: scripts/.env, KEY=VALUE, never committed."""
    p = os.path.join(HERE, ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def creds_for(book):
    prefix = ENV_PREFIX[book]
    return os.environ.get(f"{prefix}_KEY_ID"), os.environ.get(f"{prefix}_SECRET")


def get(url, kid, sec):
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def fetch_book(book, cache_dir):
    kid, sec = creds_for(book)
    if not kid or not sec:
        print(f"  {book}: no {ENV_PREFIX[book]}_KEY_ID/_SECRET in the environment — skipped")
        return False
    base = "https://paper-api.alpaca.markets"
    try:
        acct = get(f"{base}/v2/account", kid, sec)
    except urllib.error.HTTPError as e:
        print(f"  {book}: /v2/account HTTP {e.code} — key not authenticating, skipped")
        return False
    except Exception as e:
        print(f"  {book}: /v2/account failed ({e}) — skipped")
        return False

    positions = get(f"{base}/v2/positions", kid, sec)
    all_orders, after = [], None
    for _ in range(50):
        url = f"{base}/v2/orders?status=all&limit=500&direction=asc"
        if after:
            url += f"&after={after}"
        batch = get(url, kid, sec)
        if not batch:
            break
        all_orders.extend(batch)
        after = batch[-1]["submitted_at"]
        if len(batch) < 500:
            break
    history = get(f"{base}/v2/account/portfolio/history?period=all&timeframe=1D", kid, sec)

    json.dump(acct, open(os.path.join(cache_dir, f"account_{book}.json"), "w"))
    json.dump(positions, open(os.path.join(cache_dir, f"pos_{book}.json"), "w"))
    json.dump(all_orders, open(os.path.join(cache_dir, f"orders_{book}.json"), "w"))
    json.dump(history, open(os.path.join(cache_dir, f"history_{book}.json"), "w"))
    print(f"  {book}: equity ${float(acct['equity']):,.2f}, {len(positions)} positions, "
          f"{len(all_orders)} orders, {len(history.get('timestamp', []))} history points")
    return True


def iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def build_data(cache_dir):
    books_out, combined_equity_by_date, all_trades = {}, {}, []
    combined_sector_value, combined_unclassified = {}, set()

    for book in BOOKS:
        pos_p = os.path.join(cache_dir, f"pos_{book}.json")
        if not os.path.exists(pos_p):
            print(f"  {book}: no cached data — omitted entirely from this build")
            continue
        positions = json.load(open(pos_p))
        orders = json.load(open(os.path.join(cache_dir, f"orders_{book}.json")))
        history = json.load(open(os.path.join(cache_dir, f"history_{book}.json")))
        acct_p = os.path.join(cache_dir, f"account_{book}.json")
        acct = json.load(open(acct_p)) if os.path.exists(acct_p) else None
        filled = [o for o in orders if o["status"] == "filled"]

        curve = [{"date": iso(t), "equity": round(e, 2)}
                 for t, e in zip(history["timestamp"], history["equity"])]

        live_equity = float(acct["equity"]) if acct else None
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        if live_equity is not None:
            if curve and curve[-1]["date"] == today:
                curve[-1]["equity"] = round(live_equity, 2)
            else:
                curve.append({"date": today, "equity": round(live_equity, 2)})

        for pt in curve:
            combined_equity_by_date[pt["date"]] = combined_equity_by_date.get(pt["date"], 0.0) + pt["equity"]

        base_value = history.get("base_value") or (curve[0]["equity"] if curve else None)
        last_equity = live_equity if live_equity is not None else (curve[-1]["equity"] if curve else None)
        total_pl = round(last_equity - base_value, 2) if (last_equity is not None and base_value) else None
        total_pl_pct = round(total_pl / base_value, 6) if (total_pl is not None and base_value) else None

        peak, max_dd, max_dd_date = None, 0.0, None
        for pt in curve:
            if peak is None or pt["equity"] > peak:
                peak = pt["equity"]
            dd = (pt["equity"] - peak) / peak if peak else 0.0
            if dd < max_dd:
                max_dd, max_dd_date = dd, pt["date"]

        pos_out, sector_value, total_mv = [], {}, 0.0
        for p in positions:
            mv = float(p["market_value"])
            total_mv += mv
            label, confident = SM.classify(p["symbol"])
            sector_value[label] = sector_value.get(label, 0.0) + mv
            combined_sector_value[label] = combined_sector_value.get(label, 0.0) + mv
            if not confident:
                combined_unclassified.add(p["symbol"])
            pos_out.append({
                "symbol": p["symbol"], "qty": float(p["qty"]), "market_value": round(mv, 2),
                "avg_entry_price": float(p["avg_entry_price"]), "current_price": float(p["current_price"]),
                "unrealized_pl": round(float(p["unrealized_pl"]), 2),
                "unrealized_plpc": round(float(p["unrealized_plpc"]), 6), "sector": label,
            })
        pos_out.sort(key=lambda x: -x["market_value"])
        sector_list = sorted(
            [{"sector": k, "value": round(v, 2), "pct": round(v / total_mv, 6) if total_mv else 0}
             for k, v in sector_value.items()], key=lambda x: -x["value"])

        trades = []
        for o in filled:
            price = float(o["filled_avg_price"]) if o["filled_avg_price"] else 0.0
            qty = float(o["filled_qty"]) if o["filled_qty"] else 0.0
            row = {"book": book, "book_label": LABELS[book], "symbol": o["symbol"], "side": o["side"],
                   "qty": qty, "price": round(price, 4), "notional": round(price * qty, 2),
                   "filled_at": o["filled_at"], "order_type": o.get("order_type", "")}
            trades.append(row); all_trades.append(row)
        trades.sort(key=lambda x: x["filled_at"], reverse=True)

        books_out[book] = {
            "label": LABELS[book], "equity_curve": curve, "base_value": base_value,
            "current_equity": last_equity, "total_pl": total_pl, "total_pl_pct": total_pl_pct,
            "max_drawdown_pct": round(max_dd, 6), "max_drawdown_date": max_dd_date,
            "n_positions": len(pos_out), "invested_market_value": round(total_mv, 2),
            "unrealized_pl_total": round(sum(p["unrealized_pl"] for p in pos_out), 2),
            "positions": pos_out, "sector_breakdown": sector_list, "trades": trades,
            "n_filled_orders": len(filled),
            "first_trade_at": min((t["filled_at"] for t in trades), default=None),
            "last_trade_at": max((t["filled_at"] for t in trades), default=None),
        }

    combined_dates = sorted(combined_equity_by_date.keys())
    combined_curve = [{"date": d, "equity": round(combined_equity_by_date[d], 2)} for d in combined_dates]
    combined_total_mv = sum(b["invested_market_value"] for b in books_out.values())
    combined_sector_list = sorted(
        [{"sector": k, "value": round(v, 2), "pct": round(v / combined_total_mv, 6) if combined_total_mv else 0}
         for k, v in combined_sector_value.items()], key=lambda x: -x["value"])
    combined_base = sum(b["base_value"] for b in books_out.values() if b["base_value"])
    combined_equity_now = combined_curve[-1]["equity"] if combined_curve else None
    combined_pl = round(combined_equity_now - combined_base, 2) if combined_equity_now else None
    all_trades.sort(key=lambda x: x["filled_at"], reverse=True)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "books": books_out, "book_order": [b for b in BOOKS if b in books_out],
        "combined": {
            "equity_curve": combined_curve, "base_value": combined_base,
            "current_equity": combined_equity_now, "total_pl": combined_pl,
            "total_pl_pct": round(combined_pl / combined_base, 6) if combined_pl is not None and combined_base else None,
            "sector_breakdown": combined_sector_list, "n_filled_orders": len(all_trades),
        },
        "trades_all": all_trades,
        "unclassified_symbols": sorted(combined_unclassified),
        "excluded_book_note": (
            "assigned_asset_trader is excluded from this platform by explicit decision "
            "(2026-08-20): its Alpaca API key was failing authentication (401) at the time "
            "this scope was set. It is not represented in any total on this page."
        ),
        "inception_note": (
            "The books were not funded on the same day, so the combined equity curve "
            "step-changes upward each time a new book's first data point enters the sum "
            "— that is a real funding event, not a return. Check each book's own curve "
            "for its true starting date."
        ),
        "live_vs_history_note": (
            "Today's point on every equity curve is the live intraday mark from "
            "/v2/account, not the daily-close series Alpaca's portfolio-history endpoint "
            "reports (that endpoint only finalizes a day after the close, so mid-session "
            "it still shows yesterday's number). All KPI totals on this page use the live "
            "mark. Every prior day is the finalized daily close."
        ),
        "publish_note": (
            "Published automatically by a scheduled GitHub Action. Positions, sector "
            "exposure, and trade timing shown here are real paper-trading data, refreshed "
            "on the schedule set in .github/workflows/refresh_joint_dashboard.yml."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "trading", "joint_portfolio_platform.html"))
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "_cache"))
    a = ap.parse_args()

    load_dotenv_fallback()
    os.makedirs(a.cache_dir, exist_ok=True)

    print("Fetching live data (read-only) …")
    any_ok = False
    for b in BOOKS:
        any_ok = fetch_book(b, a.cache_dir) or any_ok
    if not any_ok:
        raise SystemExit("No book could be fetched — check secrets/env vars and network")

    print("Building joint_data.json …")
    data = build_data(a.cache_dir)
    json.dump(data, open(os.path.join(a.cache_dir, "joint_data.json"), "w"), indent=1)

    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    out = tpl.replace("__DATA__", json.dumps(data))
    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(out)
    print(f"Wrote {out_path} ({len(out):,} bytes)")
    for b in data["book_order"]:
        v = data["books"][b]
        print(f"  {b}: equity ${v['current_equity']:,.2f}  pl ${v['total_pl']:,.2f} ({v['total_pl_pct']*100:.2f}%)")
    c = data["combined"]
    print(f"  COMBINED: equity ${c['current_equity']:,.2f}  pl ${c['total_pl']:,.2f}")
    if data["unclassified_symbols"]:
        print("  unclassified symbols:", ", ".join(data["unclassified_symbols"]))


if __name__ == "__main__":
    main()
