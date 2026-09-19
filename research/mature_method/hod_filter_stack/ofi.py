#!/usr/bin/env python3
"""hod_filter_stack — arm O: CKS order-flow imbalance at the break minute.

Databento EQUS.MINI `bbo-1s` (priced with metadata.get_cost first: mbp-1 would be ~$450 for this
study and is NOT pulled; bbo-1s is ~$33, under the $60 cap declared in PREREG §3).

Per signal (day, symbol, break_m): the Cont-Kukanov-Stoikov order-flow imbalance accumulated over
the 5 minutes ending at the break bar's close, depth-normalised, plus the 1-minute value over the
break bar itself and the update count (the coverage number the arm is judged on BEFORE it is
scored). Raw quotes are aggregated on the fly and never stored.

1-second sampling is a PROXY: intra-second quote events are lost. EQUS.MINI is ONE publisher.
Both are stated wherever this arm is reported.

Resumable per day. Aborts if the running cost estimate exceeds $55.
"""
import os, sys, time
import numpy as np, pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')
import databento as db  # noqa: E402

D = f'{ROOT}/research/mature_method/hod_filter_stack'
OUT, STATE = f'{D}/ofi.csv', f'{D}/ofi_state.csv'
BUDGET = 55.0
PRE_MIN = 5


def utc(day, minute):
    return (pd.Timestamp(day).tz_localize('America/New_York') +
            pd.Timedelta(minutes=int(minute))).tz_convert('UTC')


def cks(bp, bs, ap, asz):
    """CKS order-flow imbalance increments from consecutive best-quote snapshots."""
    e = ((bp[1:] >= bp[:-1]) * bs[1:] - (bp[1:] <= bp[:-1]) * bs[:-1]
         - (ap[1:] <= ap[:-1]) * asz[1:] + (ap[1:] >= ap[:-1]) * asz[:-1])
    return e


def main():
    sig = pd.read_pickle(f'{D}/b2.pkl')[['day', 'symbol', 'break_m', 'split']]
    sig = sig[sig.split.isin(('TRAIN', 'VAL'))].drop_duplicates()
    done = set()
    if os.path.exists(STATE):
        done = set(pd.read_csv(STATE, dtype=str).day)
    days = [d for d in sorted(sig.day.unique()) if d not in done]
    print(f'signals {len(sig)} | days {sig.day.nunique()} | todo {len(days)}', flush=True)
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    spent = float(pd.read_csv(STATE).cost.sum()) if os.path.exists(STATE) else 0.0
    for n, day in enumerate(days):
        g = sig[sig.day == day]
        syms = sorted(g.symbol.unique())
        st = utc(day, int(g.break_m.min()) - PRE_MIN)
        en = utc(day, int(g.break_m.max()) + 1)
        kw = dict(dataset='EQUS.MINI', symbols=syms, schema='bbo-1s',
                  start=st.isoformat(), end=en.isoformat(), stype_in='raw_symbol')
        try:
            cost = float(c.metadata.get_cost(**kw))
        except Exception as e:
            print(f'  COST FAIL {day}: {e}', flush=True); continue
        if spent + cost > BUDGET:
            print(f'BUDGET STOP at {day}: spent ${spent:.2f} + ${cost:.2f} > ${BUDGET}', flush=True)
            break
        try:
            dfq = c.timeseries.get_range(**kw).to_df()
        except Exception as e:
            print(f'  PULL FAIL {day}: {e}', flush=True); time.sleep(2); continue
        spent += cost
        rows = []
        if len(dfq):
            dfq = dfq.reset_index()
            tcol = 'ts_recv' if 'ts_recv' in dfq.columns else 'ts_event'
            t = pd.to_datetime(dfq[tcol], utc=True).dt.tz_convert('America/New_York')
            dfq['m'] = t.dt.hour * 60 + t.dt.minute
            for sym, gg in dfq.groupby('symbol'):
                gg = gg.sort_values(tcol)
                bp = gg['bid_px_00'].values.astype(float); ap = gg['ask_px_00'].values.astype(float)
                bs = gg['bid_sz_00'].values.astype(float); asz = gg['ask_sz_00'].values.astype(float)
                mm = gg['m'].values.astype(int)
                if len(bp) < 3:
                    continue
                e = cks(bp, bs, ap, asz)
                depth = (bs + asz) / 2.0
                me = mm[1:]
                for bm in g[g.symbol == sym].break_m.unique():
                    w5 = (me > bm - PRE_MIN) & (me <= bm)
                    w1 = me == bm
                    dd = float(np.nanmean(depth[1:][w5])) if w5.sum() else np.nan
                    rows.append(dict(day=day, symbol=sym, break_m=int(bm),
                                     n_upd=int(w5.sum()),
                                     ofi_5m=float(e[w5].sum()) / dd if (w5.sum() and dd and dd > 0) else np.nan,
                                     ofi_1m=float(e[w1].sum()) / dd if (w1.sum() and dd and dd > 0) else np.nan))
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        pd.DataFrame([dict(day=day, cost=cost, n=len(rows))]).to_csv(
            STATE, mode='a', header=not os.path.exists(STATE), index=False)
        if n % 10 == 0:
            print(f'{n+1}/{len(days)} {day} +{len(rows)} spent ${spent:.2f}', flush=True)
    print(f'OFI DONE spent ${spent:.2f}', flush=True)


if __name__ == '__main__':
    main()
