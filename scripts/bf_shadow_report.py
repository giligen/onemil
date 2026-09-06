#!/usr/bin/env python3
"""BF consistency-profile SHADOW report — one session or a window.

Joins the live engine's shadow log lines (journalctl) to the day's bull-flag
trades (trades.db) and prints, per symbol, what profile P1 WOULD have done:

  VWAP GATE [SHADOW] would skip …          -> gate decision
  CONSISTENCY RULES [SHADOW] would skip …  -> pole>=5 / price<=20 decision
  PROFIT PARTIAL [SHADOW] would sell …      -> +2R partial would have fired
  REGIME X mult=M -> shares A -> B          -> regime multiplier (P1 = OFF)

The P1 counterfactual P&L = 0 for would-skips, else actual pnl / regime mult
(the partial's P&L effect is NOT modeled here — it needs bars; the flag is
what the shadow window proves). Pass criteria: docs/bf_p1_runbook.md §2.

Usage:
  python scripts/bf_shadow_report.py --day 2026-09-08
  python scripts/bf_shadow_report.py --days 2026-09-08 2026-09-18   # window
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, Iterable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

RX = {
    'setup': re.compile(r" (?P<sym>[A-Z.\-]+): (?:Conviction |CONVICTION SKIP|VWAP GATE)"),
    'gate': re.compile(r" (?P<sym>[A-Z.\-]+): VWAP GATE \[SHADOW\] would skip — (?P<why>.*)$"),
    'gate_live': re.compile(r" (?P<sym>[A-Z.\-]+): VWAP GATE skip — (?P<why>.*)$"),
    'rules': re.compile(r" (?P<sym>[A-Z.\-]+): CONSISTENCY RULES \[SHADOW\] would skip — (?P<why>.*)$"),
    'partial': re.compile(r" (?P<sym>[A-Z.\-]+): PROFIT PARTIAL \[SHADOW\] would sell (?P<frac>\d+%)"),
    'partial_live': re.compile(r" (?P<sym>[A-Z.\-]+): PROFIT PARTIAL fired"),
    'regime': re.compile(r" (?P<sym>[A-Z.\-]+): REGIME (?P<reg>\w+) mult=(?P<mult>[\d.]+)"),
    'error': re.compile(r"(ERROR|Traceback).*(bf_vwap_gate|bf_profit_partial|bf_risk_cap|PROFIT PARTIAL|VWAP GATE)"),
}


def parse_shadow_lines(lines: Iterable[str]) -> Dict[str, Dict]:
    """Pure parser: journal lines -> {symbol: decisions}. Testable without journald."""
    out: Dict[str, Dict] = defaultdict(lambda: {
        'seen': False, 'gate_skip': None, 'rules_skip': None,
        'partial': False, 'regime_mult': 1.0, 'regime': None})
    errors: List[str] = []
    for ln in lines:
        m = RX['error'].search(ln)
        if m:
            errors.append(ln.strip())
        m = RX['setup'].search(ln)
        if m:
            out[m.group('sym')]['seen'] = True
        for key in ('gate', 'gate_live'):
            m = RX[key].search(ln)
            if m:
                out[m.group('sym')]['gate_skip'] = m.group('why')
        m = RX['rules'].search(ln)
        if m:
            out[m.group('sym')]['rules_skip'] = m.group('why')
        for key in ('partial', 'partial_live'):
            m = RX[key].search(ln)
            if m:
                out[m.group('sym')]['partial'] = True
        m = RX['regime'].search(ln)
        if m:
            out[m.group('sym')]['regime'] = m.group('reg')
            out[m.group('sym')]['regime_mult'] = float(m.group('mult'))
    out['__errors__'] = {'lines': errors}  # type: ignore[assignment]
    return dict(out)


def p1_counterfactual(decisions: Dict[str, Dict], trades: List[Dict]) -> List[Dict]:
    """Join decisions to the day's BF trades -> per-trade rows with P1 pnl."""
    rows = []
    for t in trades:
        if (t.get('strategy') or 'bull_flag') != 'bull_flag':
            continue
        sym = t['symbol']
        d = decisions.get(sym, {})
        pnl = float(t.get('pnl') or 0.0)
        skipped = bool(d.get('gate_skip') or d.get('rules_skip'))
        mult = float(d.get('regime_mult') or 1.0) or 1.0
        rows.append({
            'symbol': sym, 'pnl': pnl,
            'p1_skip': skipped,
            'why': d.get('gate_skip') or d.get('rules_skip') or '',
            'partial_would_fire': bool(d.get('partial')),
            'regime': d.get('regime'), 'regime_mult': mult,
            'p1_pnl': 0.0 if skipped else pnl / mult,
            'seen': bool(d.get('seen')),
        })
    return rows


def report_day(day: str) -> Dict:
    from report_common import journal_grep  # journald-side grep, 60s ceiling
    from persistence.database import Database
    lines = journal_grep(
        r"VWAP GATE|CONSISTENCY RULES|PROFIT PARTIAL|REGIME |Conviction |CONVICTION SKIP|bf_vwap_gate|bf_profit_partial|bf_risk_cap",
        day)
    decisions = parse_shadow_lines(lines)
    errors = decisions.pop('__errors__', {}).get('lines', [])
    db = Database()
    trades = db.get_trades_by_date(day)
    rows = p1_counterfactual(decisions, trades)
    seen = [s for s, d in decisions.items() if d.get('seen')]
    covered = [s for s in seen if decisions[s].get('gate_skip') is not None
               or decisions[s].get('rules_skip') is not None or s in {r['symbol'] for r in rows}]
    print(f"\n=== BF P1 shadow — {day} ===")
    print(f"setups reaching conviction stage: {len(seen)} | shadow-decided: {len(covered)} "
          f"| would-skip (gate): {sum(1 for d in decisions.values() if d.get('gate_skip'))} "
          f"| would-skip (pole/price): {sum(1 for d in decisions.values() if d.get('rules_skip'))} "
          f"| would-partial: {sum(1 for d in decisions.values() if d.get('partial'))} "
          f"| new-code errors: {len(errors)}")
    if rows:
        print(f"{'symbol':8s} {'pnl':>9s} {'P1 pnl':>9s} {'skip':>5s} {'partial':>7s} {'regime':>8s}  why")
        for r in rows:
            reg = f"{r['regime'] or '-'}x{r['regime_mult']:.2f}"
            print(f"{r['symbol']:8s} {r['pnl']:9.0f} {r['p1_pnl']:9.0f} {str(r['p1_skip']):>5s} "
                  f"{str(r['partial_would_fire']):>7s} {reg:>8s}  {r['why']}")
        print(f"day: actual {sum(r['pnl'] for r in rows):,.0f} | P1 {sum(r['p1_pnl'] for r in rows):,.0f} (partial not modeled)")
    else:
        print("no bull-flag trades in trades.db for this day")
    for e in errors[:5]:
        print("  ERROR:", e[:160])
    return {'day': day, 'seen': len(seen), 'rows': rows, 'errors': len(errors)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--day', help='one session (YYYY-MM-DD)')
    ap.add_argument('--days', nargs=2, metavar=('FROM', 'TO'), help='inclusive window')
    a = ap.parse_args()
    if a.days:
        d0, d1 = (date.fromisoformat(x) for x in a.days)
        tot = {'actual': 0.0, 'p1': 0.0, 'n': 0, 'errors': 0}
        d = d0
        while d <= d1:
            if d.weekday() < 5:
                r = report_day(d.isoformat())
                tot['actual'] += sum(x['pnl'] for x in r['rows']); tot['p1'] += sum(x['p1_pnl'] for x in r['rows'])
                tot['n'] += len(r['rows']); tot['errors'] += r['errors']
            d += timedelta(days=1)
        print(f"\nWINDOW {a.days[0]}..{a.days[1]}: {tot['n']} BF trades | actual {tot['actual']:,.0f} | "
              f"P1 {tot['p1']:,.0f} (partial not modeled) | new-code errors {tot['errors']}")
        return 0
    report_day(a.day or date.today().isoformat())
    return 0


if __name__ == '__main__':
    sys.exit(main())
