"""BT replay of any live day using the SHARED rules (trading/ignition_rules)
— the Ignition equivalent of ORB's decision_parity. Universe + catalyst
data come from the day's shadow journal; bars from Alpaca.

Usage: python3 research/scripts/ignition_bt_replay.py 2026-07-22 [...]
Prints the book-parity trade list + P&L and the shadow diff.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import ignition_shadow_report as rep
from trading import ignition_rules as R
from collections import Counter


def replay_day(day: str, verbose: bool = True):
    path = ROOT / 'logs' / f'ignition_shadow_{day}.jsonl'
    if not path.exists():
        print(f'{day}: no shadow journal'); return None
    recs = [json.loads(l) for l in path.read_text().splitlines() if l]
    seen, shadow_verdict = {}, {}
    for r in recs:
        s = r['symbol']
        info = seen.setdefault(s, {'news': None, 'anchor': None,
                                   'gap': r.get('gap_pct'),
                                   'price': r.get('price')})
        # MERGE across a symbol's records — a parked first sighting has
        # has_news=None; resolution lands on a later record (8/5 ZTG:
        # first-record-wins wrongly catalyst-dropped a news trigger)
        if r.get('has_news') is not None:
            info['news'] = r.get('has_news')
        if r.get('anchor') is not None:
            info['anchor'] = r.get('anchor')
        shadow_verdict[s] = r.get('verdict', shadow_verdict.get(s))
    trades, gate_log = [], {}
    for sym, info in seen.items():
        b = rep.day_bars(sym, day)
        if b is None or len(b) < 20: gate_log[sym] = 'no_bars_api'; continue
        g = b[(b['m'] >= 570) & (b['m'] < 960)].sort_values('m').reset_index(drop=True)
        if len(g) < 20: gate_log[sym] = 'thin_day'; continue
        o = float(g.iloc[0]['open'])
        prev_close = (info['price'] / (1 + info['gap'] / 100.0)
                      if info.get('gap') is not None and info.get('price') else None)
        urej = R.universe_reject(o, prev_close)
        if urej: gate_log[sym] = urej; continue
        day_dollar = float((g['volume'] * g['close']).sum())
        if day_dollar < R.DAY_DOLLAR_MIN:
            gate_log[sym] = 'u_dollar_2M'; continue
        lvl = R.level(o)
        trig = g[(g['high'] >= lvl) & (g['m'] >= R.TRIGGER_MIN_START)
                 & (g['m'] <= R.TRIGGER_MIN_END)]
        if trig.empty: gate_log[sym] = 'no_trigger_in_window'; continue
        ti = trig.index[0]
        nxt = g[g.index > ti]
        if nxt.empty: gate_log[sym] = 'no_next_bar'; continue
        nb = nxt.iloc[0]
        entry = float(nb['open']) * R.ENTRY_SLIP
        if R.chase_reject(entry, o): gate_log[sym] = 'chase_guard'; continue
        pre = g[(g['m'] >= g.loc[ti, 'm'] - 30) & (g['m'] < g.loc[ti, 'm'])]
        if len(pre) < R.PRE_BARS_MIN: gate_log[sym] = 'pre_lt5_bars'; continue
        stop = R.stop_from_pre_lows(float(pre['low'].min()), entry)
        rp = R.r_pct_from_stop(entry, stop)
        if rp < R.R_MIN_PCT: gate_log[sym] = f'r_lt5({rp:.1f})'; continue
        rr, reason = R.resim_exit(g[g.index > nb.name].assign(
            m=g[g.index > nb.name]['m']), entry, stop, int(nb['m']) - 1)
        pos = R.position_usd(rp, float(nb['volume']) * entry)
        if R.position_reject(pos): gate_log[sym] = 'pos_lt_2k'; continue
        part = pos / max(float(nb['volume']) * entry, 1)
        pnl = pos * (rr * rp / 100.0) - pos * R.FRICTION_BPS * min(part / R.PARTICIPATION, 1.0)
        trades.append({'sym': sym, 'trig_m': int(g.loc[ti, 'm']),
                       'rr': round(rr, 2), 'reason': reason,
                       'pnl': round(pnl), 'news': info['news'],
                       'anchor': info['anchor']})
    coh = Counter(t['anchor'] for t in trades if t['anchor'])
    kept = [t for t in trades
            if R.catalyst_confirmed(t['news'], t['anchor'],
                                    coh.get(t['anchor'], 0))]
    tot = sum(t['pnl'] for t in kept)
    sh = {s for s, v in shadow_verdict.items() if v == 'SHADOW_TRIGGER'}
    bt = {t['sym'] for t in kept}
    if verbose:
        print(f"===== {day}  BT-kept={len(kept)}  P&L=${tot:+,.0f}  "
              f"shadow-trigs={len(sh)}  agree={len(bt & sh)}")
        for t in sorted(kept, key=lambda x: x['trig_m']):
            tag = 'BOTH   ' if t['sym'] in sh else 'BT-ONLY'
            print(f"  {tag} {t['sym']:6} @{t['trig_m']} {t['rr']:+.2f}R "
                  f"${t['pnl']:+,} ({t['reason']})"
                  + ('' if t['sym'] in sh else
                     f" <- shadow: {shadow_verdict.get(t['sym'], 'unsighted')}"))
        for s in sorted(sh - bt):
            why = gate_log.get(s, 'catalyst-dropped'
                               if s in {t['sym'] for t in trades} else '?')
            print(f"  SH-ONLY {s:6} <- BT gate: {why}")
    return {'day': day, 'kept': kept, 'pnl': tot,
            'both': len(bt & sh), 'bt_only': sorted(bt - sh),
            'sh_only': sorted(sh - bt)}


if __name__ == '__main__':
    days = sys.argv[1:]
    total = 0.0
    for d in days:
        out = replay_day(d)
        if out: total += out['pnl']
    if len(days) > 1:
        print(f"\nTOTAL over {len(days)} day(s): ${total:+,.0f}")
