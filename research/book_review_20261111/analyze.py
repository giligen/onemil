"""HOD + ORB paper-book review 2026-09-30 -> 2026-11-10 (PREREG_HOD_ORB_REVIEW.md). READ-ONLY on every source.

Inputs: data/trades.db (ro), logs/hod_live_parity_ledger.csv, raw/{hod,orb}_account.json (fetch_accounts.py),
research/orb_gates2/book_G3_meas.csv. Outputs (this dir only): NUMBERS.csv, weekly.csv, orb_symday_recon.csv,
hod_day_recon.csv. Prints a compact summary.
"""
import collections
import csv
import json
import math
import sqlite3
import statistics as st
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/ec2-user/onemil')
HERE = ROOT / 'research/book_review_20261111'
sys.path.insert(0, str(HERE))
import fetch_accounts as fa  # noqa: E402

ET = ZoneInfo('America/New_York')
W0, W1 = '2026-09-30', '2026-11-10'
NUM = []
SRC_DB = "sqlite3 file:data/trades.db?mode=ro"


def put(metric, book, value, window=f'{W0}..{W1}', source=SRC_DB, query=''):
    """Append one row to NUMBERS.csv (and echo it)."""
    if isinstance(value, float):
        value = round(value, 4)
    NUM.append({'metric': metric, 'book': book, 'value': value, 'window': window, 'source': source, 'query': query})
    print(f'{book:10s} {metric:42s} {value}', flush=True)


def tstats(rows, key):
    """iid t and day-clustered t of mean(key) over rows (each row has 'day')."""
    x = [r[key] for r in rows]
    n = len(x)
    if n < 2:
        return float('nan'), float('nan')
    m = st.mean(x); sd = st.stdev(x)
    t_iid = m / (sd / math.sqrt(n)) if sd else float('nan')
    by = collections.defaultdict(float)
    for r in rows:
        by[r['day']] += r[key] - m
    g = len(by)
    se = math.sqrt(sum(v * v for v in by.values()) * g / max(g - 1, 1)) / n
    return t_iid, (m / se if se else float('nan'))


def book_stats(book, rows, label, q):
    """Q1 stats on a list of fill rows with pnl, R, day, sym."""
    n = len(rows)
    pnl = [r['pnl'] for r in rows]
    days = sorted({r['day'] for r in rows}); symd = {(r['day'], r['sym']) for r in rows}
    put(f'fills{label}', book, n, query=q)
    put(f'symbol_days{label}', book, len(symd), query=q)
    put(f'sessions_traded{label}', book, len(days), query=q)
    put(f'fills_per_week{label}', book, n / 6.0, query=q + ' ; /6 weeks (30 sessions)')
    put(f'realized_usd{label}', book, sum(pnl), query=q)
    put(f'mean_R{label}', book, st.mean(r['R'] for r in rows), query=q + ' ; R = pnl / trades.total_risk')
    put(f'median_R{label}', book, st.median(r['R'] for r in rows), query=q)
    put(f'win_rate{label}', book, sum(p > 0 for p in pnl) / n, query=q)
    put(f'ex_top1_fill_usd{label}', book, sum(pnl) - max(pnl), query=q)
    k = max(1, math.ceil(0.05 * n))
    put(f'ex_top5pct_usd{label}', book, sum(sorted(pnl)[:-k]), query=q + f' ; drop top {k} fills by $')
    ti, tc = tstats(rows, 'R')
    put(f't_iid_R{label}', book, ti, query=q); put(f't_dayclustered_R{label}', book, tc, query=q)
    sd = st.stdev(r['R'] for r in rows)
    put(f'MDE_meanR_80pct{label}', book, 2.8 * sd / math.sqrt(n), query='2.8*sd(R)/sqrt(n), iid')
    put(f'avg_risk_usd{label}', book, st.mean(r['risk'] for r in rows), query=q)


def week_of(d):
    """Monday of the ISO week of date string d."""
    x = date.fromisoformat(d)
    return (x - timedelta(days=x.weekday())).isoformat()


def weekly(book, rows):
    """Weekly Mon-Fri table rows + cadence numbers."""
    weeks = [week_of(W0)]
    while weeks[-1] < week_of(W1):
        weeks.append((date.fromisoformat(weeks[-1]) + timedelta(days=7)).isoformat())
    out = []
    for w in weeks:
        rr = [r for r in rows if week_of(r['day']) == w]
        s, R = sum(r['pnl'] for r in rr), sum(r['R'] for r in rr)
        out.append({'book': book, 'week': w, 'fills': len(rr), 'usd': round(s, 2), 'R': round(R, 2),
                    'green': 'G' if s > 0 else ('R' if s < 0 else '-')})
    Rs = [o['R'] for o in out]
    put('weeks_strong_ge5R', book, sum(r >= 5 for r in Rs), query='weekly sum R >= 5 (7 calendar weeks, first/last partial)')
    put('weeks_green', book, f"{sum(o['usd'] > 0 for o in out)}/{len(out)}", query='weekly $ > 0')
    put('weekly_R_min', book, min(Rs), query='weekly table'); put('weekly_R_P10', book, sorted(Rs)[0], query='n=7 -> P10 = min')
    return out


def ledger_rows(strategy):
    """Closed fills (pnl not null) of a strategy on the paper account in the window."""
    c = sqlite3.connect('file:' + str(ROOT / 'data/trades.db') + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    q = (f"select * from trades where strategy='{strategy}' and account='paper' and pnl is not null "
         f"and trade_date between '{W0}' and '{W1}'")
    rows = [dict(r) for r in c.execute(q)]
    for r in rows:
        r['day'], r['sym'], r['risk'] = r['trade_date'], r['symbol'], r['total_risk'] or float('nan')
        r['R'] = r['pnl'] / r['total_risk'] if r['total_risk'] else 0.0
    allst = c.execute(f"select order_status,count(*),round(sum(pnl),2) from trades where strategy='{strategy}' "
                      f"and trade_date between '{W0}' and '{W1}' group by order_status").fetchall()
    return rows, q, allst


def exit_mix(book, rows, q):
    """Q2 exit-reason mix."""
    cnt = collections.Counter(r['exit_reason'] for r in rows)
    for k, v in cnt.most_common():
        put(f'exit_{k}', book, f'{v} ({v / len(rows):.0%})', query=q + ' group by exit_reason')
    return cnt


def slip(book, rows, q):
    """Q3 entry slip vs trigger and stop slip vs stop, bps (negative = adverse)."""
    ent = [(r['fill_price'] - r['entry_price']) / r['entry_price'] * 1e4 for r in rows if r['fill_price'] and r['entry_price']]
    put('entry_fill_vs_trigger_bps_mean', book, st.mean(ent), query=q + ' ; (fill_price-entry_price)/entry_price, + = paid above trigger')
    stops = [r for r in rows if r['exit_reason'] in ('stop_loss', 'stop')]
    ss = []
    for r in stops:
        s = r['real_stop_loss_price'] or r['stop_loss_price']
        if s and r['exit_price']:
            ss.append((r['exit_price'] - s) / s * 1e4)
    if ss:
        put('stop_exit_vs_stop_bps_mean', book, st.mean(ss), query=q + " and exit_reason stop ; (exit-stop)/stop, - = adverse")
        put('stop_exit_vs_stop_bps_p10', book, sorted(ss)[int(0.1 * len(ss))], query='same')
        usd = sum((r['exit_price'] - (r['real_stop_loss_price'] or r['stop_loss_price'])) * (r['filled_qty'] or r['shares']) for r in stops)
        put('stop_slip_usd_total', book, usd, query='sum (exit-stop)*qty over stop exits')
    if book == 'ORB':
        d = [r['drift_ask_to_fill_bps'] for r in rows if r['drift_ask_to_fill_bps'] is not None]
        put('drift_ask_to_fill_bps_mean', book, st.mean(d), query=q + ' ; trades.drift_ask_to_fill_bps')
        put('drift_ask_to_fill_bps_median', book, st.median(d), query='same')


def broker_symday(book, acct, classify):
    """Per (category, ET day, symbol) broker cash P&L from FILL activities; inventory carried across days."""
    pos = collections.defaultdict(float); cost = collections.defaultdict(float)
    res = collections.defaultdict(float); warn = []
    for a in sorted(acct['activities'], key=lambda a: a['transaction_time']):
        t = datetime.fromisoformat(a['transaction_time'].replace('Z', '+00:00')).astimezone(ET)
        d = t.date().isoformat(); sym = a['symbol']; q = float(a['qty']); px = float(a['price'])
        cat = classify(a, d)
        sgn = 1 if a['side'] == 'buy' else -1
        k = (cat, sym)
        if d < W0:
            pos[k] += sgn * q; cost[k] += sgn * q * px; continue
        if pos[k] == 0 or (pos[k] > 0) == (sgn > 0):
            pos[k] += sgn * q; cost[k] += sgn * q * px
            continue
        close = min(q, abs(pos[k])); avg = cost[k] / pos[k]
        pnl = (px - avg) * close * (1 if pos[k] > 0 else -1)
        res[(cat, d, sym)] += pnl
        pos[k] -= math.copysign(close, pos[k])
        cost[k] = avg * pos[k]
        if q > close:
            rem = q - close; pos[k] += sgn * rem; cost[k] += sgn * rem * px
    openp = {k: (v, cost[k]) for k, v in pos.items() if abs(v) > 1e-9}
    return res, openp, warn


def history_eq(acct):
    """Map ET date -> equity from 1D portfolio history."""
    h = acct['history']; out = {}
    for ts, eq in zip(h['timestamp'], h['equity']):
        out[datetime.fromtimestamp(ts, timezone.utc).astimezone(ET).date().isoformat()] = eq
    return out


def main():
    """Run every PREREG question and write the outputs."""
    hod_acct = json.load(open(HERE / 'raw/hod_account.json'))
    orb_acct = json.load(open(HERE / 'raw/orb_account.json'))
    weekly_rows = []

    # ---------------- ORB ----------------
    orb, q_orb, orb_st = ledger_rows('orb')
    put('ledger_status_mix', 'ORB', '; '.join(f'{s}:{n}:{p}' for s, n, p in orb_st), query="group by order_status (all accounts)")
    book_stats('ORB', orb, '', q_orb)
    out = [r for r in orb if not (r['day'] == '2026-11-09' and r['sym'] == 'BEX')]
    book_stats('ORB', out, '_exBEXrow', q_orb + " and not BEX 11/9")
    ex_top = sorted(orb, key=lambda r: r['pnl'])[:-1]
    put('mean_R_ex_top1', 'ORB', st.mean(r['R'] for r in ex_top), query=q_orb)
    cnt = exit_mix('ORB', orb, q_orb)
    stops = sum(v for k, v in cnt.items() if k == 'stop_loss')
    put('stop_rate', 'ORB', stops / len(orb), query=q_orb)
    bt = list(csv.DictReader(open(ROOT / 'research/orb_gates2/book_G3_meas.csv')))
    ent = [r for r in bt if r['exit_reason'] != 'no_fill']
    put('BT_stop_rate', 'ORB', sum(r['exit_reason'] == 'stop' for r in ent) / len(ent), window='BT 2025-01..',
        source='research/orb_gates2/book_G3_meas.csv', query="exit_reason=='stop' / exit_reason!='no_fill'")
    put('BT_fill_rate', 'ORB', len(ent) / len(bt), window='BT', source='book_G3_meas.csv', query="!= no_fill / all")
    slip('ORB', orb, q_orb)
    weekly_rows += weekly('ORB', orb)

    # ORB account attribution
    oid2cat = {}
    led_oids = set()
    c = sqlite3.connect('file:' + str(ROOT / 'data/trades.db') + '?mode=ro', uri=True)
    for (oid,) in c.execute("select order_id from trades where strategy='orb' and order_id is not null"):
        led_oids.add(oid)
    orb_symdays = {(d, s) for d, s in c.execute("select trade_date,symbol from trades where strategy='orb'")}

    def flat(orders):
        """Flatten nested bracket legs, children inherit the parent's client id."""
        for o in orders:
            yield o, o
            for leg in o.get('legs') or []:
                yield leg, o
    env = fa.load_env()
    k, s = env['ALPACA_ORB_API_KEY'], env['ALPACA_ORB_API_SECRET']
    for o, parent in flat(orb_acct['orders']):
        oid2cat[o['id']] = (parent.get('client_order_id') or '', parent['id'])
    missing = {a['order_id'] for a in orb_acct['activities'] if a['order_id'] not in oid2cat}
    for oid in missing:
        o = fa.get(f'/v2/orders/{oid}', k, s)
        oid2cat[oid] = (o.get('client_order_id') or '', o['id'])
    print(f'ORB: fetched {len(missing)} orders missing from the list', flush=True)

    def orb_classify(a, d):
        """tom- -> TOM; micha- -> EXT; orb- / ledger order id / ledger symbol-day -> ORB; else EXT."""
        cid, pid = oid2cat.get(a['order_id'], ('', ''))
        if cid.startswith('tom-'):
            return 'TOM'
        if cid.startswith('micha'):
            return 'EXT'
        if cid.startswith('orb-') or a['order_id'] in led_oids or pid in led_oids or (d, a['symbol']) in orb_symdays:
            return 'ORB'
        return 'EXT'
    res, openp, _ = broker_symday('ORB', orb_acct, orb_classify)
    eq = history_eq(orb_acct)
    base_d = max(d for d in eq if d < W0); end_d = max(d for d in eq if d <= W1)
    put('account_equity_change', 'ORB-acct', eq[end_d] - eq[base_d], window=f'{base_d} close..{end_d} close',
        source='GET /v2/account/portfolio/history 1D', query='equity[end]-equity[base]')
    cats = collections.defaultdict(float)
    for (cat, d, sym), v in res.items():
        cats[cat] += v
    posmv = {p['symbol']: (float(p['qty']), float(p['avg_entry_price']), float(p.get('lastday_price') or p['current_price'])) for p in orb_acct['positions']}
    unreal = collections.defaultdict(float)
    for (cat, sym), (qty, cst) in openp.items():
        if sym in posmv:
            unreal[cat] += qty * posmv[sym][2] - cst
        else:
            print(f'WARNING: ORB inventory {cat} {sym} {qty} not in positions (pre-window inventory?)', flush=True)
    for cat in ('ORB', 'TOM', 'EXT'):
        put(f'broker_realized_{cat}', 'ORB-acct', cats[cat], source='GET /v2/account/activities FILL (+orders client_order_id)',
            query='avg-cost realized per category; classify tom-/micha-/orb-|ledger')
        put(f'broker_open_unrealized_{cat}', 'ORB-acct', unreal[cat], source='GET /v2/positions lastday_price', query='qty*lastday - cost')
    resid = eq[end_d] - eq[base_d] - sum(cats.values()) - sum(unreal.values())
    put('equity_residual_unexplained', 'ORB-acct', resid, query='equity change - sum(realized+unrealized all cats)')
    # per symbol-day ledger vs broker
    led = collections.defaultdict(float)
    for r in orb:
        led[(r['day'], r['sym'])] += r['pnl']
    keys = sorted({(d, s) for (cat, d, s) in res if cat == 'ORB'} | set(led))
    recon = []
    for d, sname in keys:
        b = res.get(('ORB', d, sname), 0.0); l = led.get((d, sname), 0.0)
        recon.append({'day': d, 'symbol': sname, 'ledger_usd': round(l, 2), 'broker_usd': round(b, 2), 'diff': round(b - l, 2)})
    with open(HERE / 'orb_symday_recon.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recon[0])); w.writeheader(); w.writerows(recon)
    put('ORB_broker_minus_ledger_usd', 'ORB', sum(x['diff'] for x in recon), source='orb_symday_recon.csv', query='sum diff')
    for x in recon:
        if abs(x['diff']) > 5:
            put(f"recon_diff_{x['day']}_{x['symbol']}", 'ORB', x['diff'], source='orb_symday_recon.csv', query='broker - ledger')
    orb_broker_days = sorted({d for (cat, d, s2) in res if cat == 'ORB'})
    put('ORB_broker_symbol_days', 'ORB', len({(d, s2) for (cat, d, s2) in res if cat == 'ORB'}), source='activities', query='ORB-classified closing fills')
    # gate counts
    c2 = sqlite3.connect('file:' + str(ROOT / 'data/trades.db') + '?mode=ro', uri=True)
    for since in ('2026-09-25', '2026-10-30'):
        sd = c2.execute("select count(distinct trade_date||symbol), count(distinct trade_date), round(sum(pnl),2) from trades where strategy='orb' "
                        "and account='paper' and pnl is not null and trade_date>=? and trade_date<=?", (since, W1)).fetchone()
        put(f'gate_symbol_days_since_{since}', 'ORB', sd[0], window=f'{since}..{W1}', query='distinct trade_date||symbol, paper, pnl not null')
        put(f'gate_sessions_with_fill_since_{since}', 'ORB', sd[1], window=f'{since}..{W1}', query='same')
        put(f'gate_realized_since_{since}', 'ORB', sd[2], window=f'{since}..{W1}', query='same')
        best = c2.execute("select max(pnl) from trades where strategy='orb' and account='paper' and pnl is not null and trade_date between ? and ?", (since, W1)).fetchone()[0]
        put(f'gate_ex_monster_since_{since}', 'ORB', sd[2] - best, window=f'{since}..{W1}', query='realized - max(pnl)')
    rl = c2.execute("select account, min(trade_date), max(trade_date), count(*), round(sum(pnl),2) from trades where strategy='orb' and pnl is not null and trade_date>='2026-09-21' group by account").fetchall()
    put('ramp_check_window_composition', 'ORB', '; '.join(f'{a}:{d0}..{d1}:n{n}:${p}' for a, d0, d1, n, p in rl), window='2026-09-21..',
        query="what orb_ramp_check.load_fills counts: strategy='orb' and pnl not null since stage start, ANY account")

    # ---------------- HOD ----------------
    hod, q_hod, hod_st = ledger_rows('hod_break')
    put('ledger_status_mix', 'HOD', '; '.join(f'{s2}:{n}:{p}' for s2, n, p in hod_st), query="group by order_status")
    book_stats('HOD', hod, '', q_hod)
    cnt = exit_mix('HOD', hod, q_hod)
    put('stop_rate', 'HOD', sum(v for k2, v in cnt.items() if 'stop' in (k2 or '')) / len(hod), query=q_hod + " exit_reason like '%stop%'")
    slip('HOD', hod, q_hod)
    weekly_rows += weekly('HOD', hod)
    tot = c2.execute("select count(*) from trades where strategy='hod_break' and account='paper' and pnl is not null").fetchone()[0]
    put('gate_paper_fills_all_time', 'HOD', tot, window='2026-09-29..', query="strategy='hod_break' and account='paper' and pnl not null")
    # parity ledger
    pl = [r for r in csv.DictReader(open(ROOT / 'logs/hod_live_parity_ledger.csv')) if W0 <= r['date'] <= W1]
    put('parity_rows', 'HOD', len(pl), window=f"{min(r['date'] for r in pl)}..{W1}", source='logs/hod_live_parity_ledger.csv', query='rows in window')
    put('parity_broker_status', 'HOD', '; '.join(f'{k2}:{v}' for k2, v in collections.Counter(r['broker_status'] for r in pl).most_common()),
        source='hod_live_parity_ledger.csv', query='Counter(broker_status)')
    put('parity_reason', 'HOD', '; '.join(f'{k2}:{v}' for k2, v in collections.Counter(r['reason'] for r in pl).most_common(8)),
        source='hod_live_parity_ledger.csv', query='Counter(reason) top 8')
    put('parity_tape_expected', 'HOD', '; '.join(f'{k2}:{v}' for k2, v in collections.Counter(r['tape_expected'] for r in pl).most_common()),
        source='hod_live_parity_ledger.csv', query='Counter(tape_expected)')
    sl = [float(r['slippage_vs_tape_bps']) for r in pl if r['slippage_vs_tape_bps'] not in ('', None)]
    if sl:
        put('slippage_vs_tape_bps_mean', 'HOD', st.mean(sl), source='hod_live_parity_ledger.csv', query=f'n={len(sl)}')
        put('slippage_vs_tape_bps_median', 'HOD', st.median(sl), source='hod_live_parity_ledger.csv', query='same')
    filled = [r for r in pl if r['broker_status'] in ('filled', 'partially_filled')]
    crossed = [r for r in pl if r['tape_expected'] not in ('NO_CROSS', '')]
    put('parity_filled_over_tape_crossed', 'HOD', f"{len(filled)}/{len(crossed)}", source='hod_live_parity_ledger.csv',
        query="broker_status filled|partially_filled / tape_expected != NO_CROSS")
    qual = collections.Counter((r['scanner_qualified_at_arm'], r['broker_status'] in ('filled', 'partially_filled')) for r in pl)
    put('parity_scanner_qualified_x_filled', 'HOD', str(dict(qual)), source='hod_live_parity_ledger.csv', query='Counter')
    # HOD account cross-check
    hres, hopen, _ = broker_symday('HOD', hod_acct, lambda a, d: 'HOD')
    heq = history_eq(hod_acct)
    hb = max(d for d in heq if d < W0); he = max(d for d in heq if d <= W1)
    put('account_equity_change', 'HOD-acct', heq[he] - heq[hb], window=f'{hb}..{he}', source='portfolio/history 1D', query='equity diff')
    h4 = max(d for d in heq if d <= '2026-10-14')
    put('account_equity_change_4wk', 'HOD-acct', heq[he] - heq[h4], window=f'{h4}..{he}', source='portfolio/history 1D', query='equity diff')
    put('broker_realized', 'HOD-acct', sum(hres.values()), source='activities FILL', query='avg-cost realized')
    put('broker_open_inventory', 'HOD-acct', str({k2[1]: v for k2, v in hopen.items()}), source='activities', query='non-flat at end')
    put('equity_residual_unexplained', 'HOD-acct', heq[he] - heq[hb] - sum(hres.values()), query='equity change - broker realized')
    hl = collections.defaultdict(float)
    for r in hod:
        hl[r['day']] += r['pnl']
    hb_day = collections.defaultdict(float)
    for (cat, d, sname), v in hres.items():
        hb_day[d] += v
    hrec = [{'day': d, 'ledger_usd': round(hl.get(d, 0), 2), 'broker_usd': round(hb_day.get(d, 0), 2),
             'diff': round(hb_day.get(d, 0) - hl.get(d, 0), 2)} for d in sorted(set(hl) | set(hb_day))]
    with open(HERE / 'hod_day_recon.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hrec[0])); w.writeheader(); w.writerows(hrec)
    put('HOD_broker_minus_ledger_usd', 'HOD', sum(x['diff'] for x in hrec), source='hod_day_recon.csv', query='sum diff')
    put('HOD_days_absdiff_gt_25', 'HOD', '; '.join(f"{x['day']}:{x['diff']}" for x in hrec if abs(x['diff']) > 25), source='hod_day_recon.csv', query='|diff|>25')

    with open(HERE / 'weekly.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(weekly_rows[0])); w.writeheader(); w.writerows(weekly_rows)
    for r in weekly_rows:
        print(r, flush=True)
    with open(HERE / 'NUMBERS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'book', 'value', 'window', 'source', 'query']); w.writeheader(); w.writerows(NUM)


if __name__ == '__main__':
    main()
