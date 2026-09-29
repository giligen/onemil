"""Refuter probe for PREREG_1617 Frame B (cell 1,619/1,620): re-walks the builder's own fills under
(a) the builder's rule, (b) the PREREG's strict-below cover rule, (c) stop filled at the triggering
print (fast-market), and records entry-print obtainability (first-hit print size, volume above the
limit in the entry window) and the easy_to_borrow flag. Read-only on all inputs."""
import os, sys, sqlite3, time
import numpy as np, pandas as pd
REPO = '/home/ec2-user/onemil'
sys.path.insert(0, REPO)
import research.hod_entry.cell_1619 as C

OUT = os.path.join(REPO, 'research/hod_entry/review/refute_1619_probe.csv')


def walk(symbol, day, entry_ts, entry_m, pp, stop, cover, conn, split, strict, stop_at_print):
    """Copy of C.walk_short with two switches: strict (cover needs a print < cover) and
    stop_at_print (tape stop fills at the triggering print, bar stop at max(open, stop))."""
    if pp >= stop - 1e-9:
        return pp, 'stop', entry_ts, 0.0
    m, cur = entry_m, entry_ts
    while m < C.EOD_M:
        tr = C.load_minute_trades(symbol, day, m) if m <= entry_m + C.TAPE_WINDOW_MIN else None
        if tr is not None:
            sub = tr[tr.ts > cur].sort_values('ts', kind='stable')
            sh = sub[sub.price >= stop - 1e-9]
            ch = sub[sub.price < cover - 1e-9] if strict else sub[sub.price <= cover + 1e-9]
            ts_s = int(sh.ts.iloc[0]) if len(sh) else None
            ts_c = int(ch.ts.iloc[0]) if len(ch) else None
            if ts_s is not None and (ts_c is None or ts_s <= ts_c):
                trig = float(sh.price.iloc[0])
                return (trig if stop_at_print else stop), 'stop', ts_s, (trig / stop - 1) * 1e4
            if ts_c is not None:
                return cover, 'cover', ts_c, 0.0
        else:
            b = C.load_bars(symbol, day, conn)
            b = b[b.m == m]
            if len(b):
                b = b.iloc[0]
                if b.h >= stop - 1e-9:
                    px = b.o if b.o >= stop else stop
                    return px, 'stop', C.minute_start_ns(day, m), (px / stop - 1) * 1e4
                touch = (b.l < cover - 1e-9) if strict else (b.l <= cover + 1e-9)
                if touch:
                    return cover, 'cover', C.minute_start_ns(day, m), 0.0
        cur = C.minute_start_ns(day, m + 1) - 1
        m += 1
    bars = C.load_bars(symbol, day, conn)
    br = bars[bars.m == C.EOD_M]
    base = float(br.iloc[0].o) if len(br) else float(bars[bars.m < C.EOD_M].sort_values('m').iloc[-1].c)
    return base * (1 + C.EOD_ASK_BPS[split] / 1e4), 'eod', C.minute_start_ns(day, C.EOD_M), 0.0


def main():
    """Run the probe on the builder's eligible population and write one row per filled entry."""
    _all, pop = C.load_population()
    etb = pd.read_csv(C.BORROW_CSV).set_index('symbol').easy_to_borrow.to_dict()
    conn = sqlite3.connect(f'file:{C.BARS_DB}?mode=ro', uri=True)
    rows, t0 = [], time.time()
    for i, r in enumerate(pop.itertuples()):
        if i % 1000 == 0:
            print(f'probe {i}/{len(pop)} {time.time()-t0:.0f}s', flush=True)
        level = float(r.level); stop = level * (1 + C.STOP_BPS); limit = level * (1 + C.OFFER_BPS)
        e = C.find_short_entry(r.symbol, r.day, level, float(r.fill_min))
        if e['status'] != 'fill':
            continue
        # obtainability: prints strictly above the limit in the entry window, before the stop print
        m0 = int(np.floor(r.fill_min)); fr = []
        for m in range(m0, m0 + C.ENTRY_WINDOW_MIN + 1):
            t = C.load_minute_trades(r.symbol, r.day, m)
            if t is not None:
                fr.append(t)
        tp = pd.concat(fr)
        tp = tp[(tp.ts >= C.et_ns(r.day, float(r.fill_min) * 60)) &
                (tp.ts < C.minute_start_ns(r.day, m0 + C.ENTRY_WINDOW_MIN + 1))].sort_values('ts')
        above = tp[tp.price > limit + 1e-9]
        first = above.iloc[0]
        rec = dict(split=r.split, day=r.day, symbol=r.symbol, level=level, etb=etb.get(r.symbol),
                   half_entry=r.half_entry, outcome_R=r.outcome_R, R_pct=r.R_pct,
                   first_size=float(first['size']), first_px_bps=(first.price / limit - 1) * 1e4,
                   vol_above=float(above['size'].sum()), n_above=len(above),
                   vol_above_rl=float(above[above['size'] >= 100]['size'].sum()),
                   first_rl_px_bps=((above[above['size'] >= 100].price.iloc[0] / limit - 1) * 1e4
                                    if (above['size'] >= 100).any() else np.nan))
        for tag, cov in (('c19', level - C.COVER_ABS_1619), ('c20', level * (1 - C.COVER_PCT_1620))):
            for v, strict, sap in (('b', False, False), ('s', True, False), ('sp', True, True)):
                px, why, xts, sbps = walk(r.symbol, r.day, e['entry_ts'], e['entry_m'], e['print_price'],
                                          stop, cov, conn, r.split, strict, sap)
                Rf, nRf, npct, _, _ = C.cost_and_r(e['entry_price'], px, why, e['entry_ts'], xts, stop, r.split)
                rec[f'{tag}_{v}_Rf'] = nRf; rec[f'{tag}_{v}_pct'] = npct; rec[f'{tag}_{v}_why'] = why
                rec[f'{tag}_{v}_stopx_bps'] = sbps
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'wrote {OUT} {len(rows)} rows {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
