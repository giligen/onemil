"""S1 step 2 — turn the compact status rows into LULD halt/resume EVENTS with symbols.

Taxonomy (decoded from databento_dbn.StatusAction / StatusReason):
  LULD volatility halt   = action PAUSE(9)  + reason LULD_PAUSE(50)
  its resume             = the next TRADING(7) row for the same (day, instrument) with is_trading='Y'
  news halt              = action HALT(8) or QUOTING(3) with reason NEWS_PENDING(30) / NEWS_RELEASED(31)
                           / NEWS_AND_RESUMPTION_TIMES(32)
  regulatory / other     = HALT(8)/SUSPEND(10) with any other reason, and PAUSE(9) with reason != 50
  market-wide (LULD-MWCB)= reason 120..124
The 07:04 UTC pre-session state broadcast and the 09:30 open TRADING broadcast are outside the
LULD path by construction (they carry action 1/7/14 and are only kept as the resume candidates of
an instrument that actually paused).

instrument_id -> raw_symbol comes from the FREE symbology.resolve endpoint, resolved per calendar
day (XNAS.ITCH instrument ids are per-day locate codes). The `definition` schema ($35.20) is not
needed; resolve returned 0 not_found on the probe day.
"""
import os, sys, glob, json
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
import databento as db

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
RAW = f'{HERE}/raw'

ACT = {0: 'NONE', 1: 'PRE_OPEN', 2: 'PRE_CROSS', 3: 'QUOTING', 4: 'CROSS', 5: 'ROTATION',
       6: 'NEW_PRICE_INDICATION', 7: 'TRADING', 8: 'HALT', 9: 'PAUSE', 10: 'SUSPEND',
       11: 'PRE_CLOSE', 12: 'CLOSE', 13: 'POST_CLOSE', 14: 'SSR_CHANGE', 15: 'NOT_AVAILABLE'}
NEWS_REASONS = {30, 31, 32, 33}
MWCB_REASONS = {120, 121, 122, 123, 124}


def classify(action, reason):
    if action == 9 and reason == 50:
        return 'luld_pause'
    if reason in MWCB_REASONS:
        return 'market_wide'
    if reason in NEWS_REASONS:
        return 'news'
    if action in (8, 10):
        return 'regulatory_other'
    if action == 9:
        return 'pause_other'
    return 'state'


def main():
    files = sorted(glob.glob(f'{RAW}/halt_rows_*.parquet'))
    assert files, 'no halt_rows parquet — run fetch_status.py first'
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['et'] = pd.to_datetime(df['et'], utc=True).dt.tz_convert('America/New_York')
    df = df.sort_values(['day', 'instrument_id', 'et']).reset_index(drop=True)
    df['kind'] = [classify(a, r) for a, r in zip(df['action'], df['reason'])]

    tax = df.groupby('kind').size().sort_values(ascending=False)
    print('== status-row taxonomy (halted-instrument tape, in session) ==')
    print(tax.to_string())

    events = []
    unresumed = 0
    for (day, iid), g in df.groupby(['day', 'instrument_id'], sort=False):
        g = g.reset_index(drop=True)
        seq = 0
        for i, row in g.iterrows():
            if row['kind'] != 'luld_pause':
                continue
            seq += 1
            nxt = g.iloc[i + 1:]
            res = nxt[(nxt['action'] == 7) & (nxt['is_trading'].astype(str) == 'Y')]
            if res.empty:
                unresumed += 1
                continue
            r0 = res.iloc[0]
            events.append({'day': day, 'instrument_id': int(iid), 'halt_seq': seq,
                           'halt_ts': row['et'], 'resume_ts': r0['et'],
                           'halt_minutes': (r0['et'] - row['et']).total_seconds() / 60.0})
    ev = pd.DataFrame(events)
    print(f"\nLULD halt/resume pairs: {len(ev):,}   unresumed-by-16:05: {unresumed}")
    if ev.empty:
        sys.exit('no events')
    print('halt duration minutes:', ev['halt_minutes'].describe()[['mean', '50%', 'max']].round(2).to_dict())

    # --- symbols, day by day, via the free symbology endpoint --------------------------------
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    cache_path = f'{HERE}/iid_symbol_map.csv'
    known = {}
    if os.path.exists(cache_path):
        m = pd.read_csv(cache_path, keep_default_na=False, na_values=[''])
        known = {(r.day, int(r.instrument_id)): r.symbol for r in m.itertuples()}
    rows, miss = [], 0
    days = sorted(ev['day'].unique())
    for n, day in enumerate(days):
        ids = sorted(ev.loc[ev['day'] == day, 'instrument_id'].unique().tolist())
        need = [i for i in ids if (day, i) not in known]
        if need:
            nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            r = c.symbology.resolve(dataset='XNAS.ITCH', symbols=[str(i) for i in need],
                                    stype_in='instrument_id', stype_out='raw_symbol',
                                    start_date=day, end_date=nxt)
            for k, v in r['result'].items():
                if v:
                    known[(day, int(k))] = v[0]['s']
            miss += len(r.get('not_found') or [])
        if n % 50 == 0:
            print(f'  resolved {n}/{len(days)} days', flush=True)
    pd.DataFrame([{'day': d, 'instrument_id': i, 'symbol': s} for (d, i), s in known.items()]).to_csv(cache_path, index=False)
    ev['symbol'] = [known.get((d, i)) for d, i in zip(ev['day'], ev['instrument_id'])]
    res_share = ev['symbol'].notna().mean()
    print(f"symbol resolution: {res_share:.4%}  (not_found ids {miss})")

    ev.to_parquet(f'{HERE}/luld_events_raw.parquet', index=False)

    # non-LULD halts, counted separately and NEVER mixed in
    other = df[df['kind'].isin(['news', 'regulatory_other', 'pause_other', 'market_wide'])]
    other.groupby(['kind', 'action', 'reason']).size().reset_index(name='n').to_csv(f'{HERE}/nonluld_halts.csv', index=False)
    print('\n== non-LULD halting rows (counted, excluded from the study) ==')
    print(other.groupby('kind').size().to_string())
    json.dump({'luld_pairs': int(len(ev)), 'unresumed': int(unresumed),
               'symbol_resolution': round(float(res_share), 6)},
              open(f'{HERE}/events_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
