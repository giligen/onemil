#!/usr/bin/env python3
"""
Phase 2 of the news-classifier A/B — analyse the precomputed verdicts.

The bull-flag cache is RAW (news-kill is applied inside BacktestRunner.run(),
not at cache-build — verified: kill-eligible trades are present in the cache).
So the 4-arm A/B can be computed directly on the raw cache by applying the
news-kill rules per classifier:

  Arm 1  no news filter        — keep every trade
  Arm 2  regex   + news-kill   — current production
  Arm 3  haiku   + news-kill
  Arm 4  haiku_revised + news-kill

news-kill rules (backtest.py::_check_news_kill) — has_catalyst => always keep;
otherwise kill if: avg_vol>=3M (R1) | entry<$3 (R2) | float>=30M (R3) |
$5-12 & pole 8-15% (R4). R4 needs pole_gain (not in the cache) — reported as a
bounded upper estimate, not applied.

Reads data/bull_flag_cache_e50_x30.csv, data/news_ab.db, data/cache.db.
No API calls.
"""
import csv
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

CACHE_CSV = "data/bull_flag_cache_e50_x30.csv"
NEWS_AB_DB = "data/news_ab.db"
CACHE_DB = "data/cache.db"


def news_kill_keeps(has_catalyst, avg_vol, entry_price, float_shares):
    """Replicate backtest.py::_check_news_kill Rules 1-3 (R4 omitted)."""
    if has_catalyst:
        return True, 'has_catalyst'
    if avg_vol >= 3_000_000:
        return False, 'R1_vol>=3M'
    if entry_price < 3.0:
        return False, 'R2_price<$3'
    if float_shares >= 30_000_000:
        return False, 'R3_float>=30M'
    return True, 'survives'


def fmt(label, trades):
    """Summarise a trade subset: n, win-rate, total pnl."""
    n = len(trades)
    pnl = sum(t['pnl'] for t in trades)
    wr = (sum(1 for t in trades if t['pnl'] > 0) / n * 100) if n else 0.0
    return f"  {label:<28} n={n:>4}  WR={wr:>5.1f}%  pnl=${pnl:>11,.0f}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="News-classifier A/B analysis")
    ap.add_argument('--cache', default=CACHE_CSV, help="Stage-1 cache CSV path")
    ap.add_argument('--db', default=NEWS_AB_DB, help="news_ab.db path")
    args = ap.parse_args()
    cache_csv, news_ab_db = args.cache, args.db

    verdict, floats = {}, {}
    c = sqlite3.connect(news_ab_db)
    for s, d, rx, hk, hkr in c.execute(
            "SELECT symbol,trade_date,regex_catalyst,haiku_catalyst,"
            "haiku_revised_catalyst FROM news_ab"):
        verdict[(s, d)] = {'regex': rx, 'haiku': hk, 'haiku_revised': hkr}
    c.close()
    cc = sqlite3.connect(CACHE_DB)
    floats = dict(cc.execute(
        "SELECT symbol,float_shares FROM universe WHERE float_shares IS NOT NULL"))
    cc.close()

    trades = []
    with open(cache_csv) as f:
        for r in csv.DictReader(f):
            key = (r['symbol'], r['date'])
            if key not in verdict:
                continue
            trades.append({
                'symbol': r['symbol'], 'date': r['date'],
                'pnl': float(r['pnl'] or 0),
                'entry_price': float(r['entry_price'] or 0),
                'avg_vol': float(r['avg_volume_20d'] or 0),
                'float_shares': float(floats.get(r['symbol'], 0) or 0),
                'v': verdict[key],
            })

    print("=" * 72)
    print(f"NEWS-CLASSIFIER A/B — {len(trades)} candidate trades (raw cache)")
    print("=" * 72)

    # ── 4-arm comparison ────────────────────────────────────────────────────
    print("\n[1] 4-ARM NEWS-KILL COMPARISON (raw Stage-1 pnl)\n")
    print(fmt("Arm 1  no news filter", trades))
    arms = {}
    for arm, col in [('Arm 2  regex', 'regex'),
                     ('Arm 3  haiku', 'haiku'),
                     ('Arm 4  haiku_revised', 'haiku_revised')]:
        kept, killed = [], []
        for t in trades:
            keep, reason = news_kill_keeps(
                t['v'][col], t['avg_vol'], t['entry_price'], t['float_shares'])
            (kept if keep else killed).append((t, reason))
        arms[col] = {'kept': [t for t, _ in kept], 'killed': killed}
        print(fmt(f"{arm} + news-kill", [t for t, _ in kept]))
        delta = sum(t['pnl'] for t, _ in kept) - sum(t['pnl'] for t in trades)
        print(f"      killed {len(killed):>2} trades  "
              f"=> pnl delta vs no-filter: ${delta:+,.0f}")

    # ── killed-trade detail ─────────────────────────────────────────────────
    print("\n[2] TRADES KILLED BY EACH ARM\n")
    for col in ('regex', 'haiku', 'haiku_revised'):
        killed = arms[col]['killed']
        print(f"  {col}: {len(killed)} killed")
        for t, reason in killed:
            print(f"    {t['symbol']:6} {t['date']}  pnl=${t['pnl']:>9,.0f}  "
                  f"[{reason}]")

    # ── classifier disagreement P&L ─────────────────────────────────────────
    print("\n[3] CLASSIFIER DISAGREEMENT — are the disputed trades winners?\n")
    rx_only = [t for t in trades if t['v']['regex'] and not t['v']['haiku']]
    hk_only = [t for t in trades if t['v']['haiku'] and not t['v']['regex']]
    print(fmt("regex=catalyst, haiku=NO", rx_only))
    print("      ^ regex's extra 'catalysts' that Haiku rejects (false positives?)")
    print(fmt("haiku=catalyst, regex=NO", hk_only))
    hkr_diff = [t for t in trades
                if t['v']['haiku'] != t['v']['haiku_revised']]
    print(fmt("haiku != haiku_revised", hkr_diff))
    for t in hkr_diff:
        print(f"    {t['symbol']:6} {t['date']}  pnl=${t['pnl']:>9,.0f}  "
              f"haiku={t['v']['haiku']} revised={t['v']['haiku_revised']}")

    # ── Rule 4 bound ────────────────────────────────────────────────────────
    r4 = [t for t in trades
          if 5 <= t['entry_price'] < 12 and not t['v']['haiku']]
    print(f"\n[4] Rule 4 (pole-based, omitted): <= {len(r4)} haiku-no-catalyst "
          f"trades have $5-12 entry — upper bound on un-modelled R4 kills.")

    # ── emit per-arm sub-caches for a rigorous Stage-2 backtest ─────────────
    # arms 3 & 4 are identical (haiku_revised kills the same 29) → 3 distinct sets.
    print("\n[5] Writing per-arm sub-cache CSVs (full schema) for Stage-2...")
    kept_keys = {
        'arm1_all':   set((t['symbol'], t['date']) for t in trades),
        'arm2_regex': set((t['symbol'], t['date']) for t in arms['regex']['kept']),
        'arm3_haiku': set((t['symbol'], t['date']) for t in arms['haiku']['kept']),
    }
    with open(cache_csv) as f:
        rdr = csv.DictReader(f)
        hdr = rdr.fieldnames
        allrows = list(rdr)
    for name, keys in kept_keys.items():
        out = f"data/news_ab_{name}.csv"
        rows = [r for r in allrows if (r['symbol'], r['date']) in keys]
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        print(f"    {out}: {len(rows)} rows")
    print("=" * 72)


if __name__ == '__main__':
    main()
