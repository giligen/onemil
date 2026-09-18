"""S1 step 4 — the executability check the two surviving cells live or die on.

Both cells that cleared G1 are SHORTS. Reg SHO rule 201 (SSR) forbids a short sale at or below the
national best bid for the rest of the day and the next, once a stock trades 10% below the prior
close — and a LULD halt is by definition a large move. The XNAS.ITCH status record carries the flag
directly (`is_short_sell_restricted`), which the first pass dropped. This re-reads the raw monthly
pulls and stamps each LULD event with the SSR state in force at the RESUME instant (merge_asof
backward, per day and instrument).
"""
import glob, os, gc
import pandas as pd

import databento as db

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
RAW = f'{HERE}/raw'


def main():
    ev = pd.read_parquet(f'{HERE}/luld_events_raw.parquet')
    ev['resume_ts'] = pd.to_datetime(ev['resume_ts'], utc=True).dt.tz_convert('America/New_York')
    ev['instrument_id'] = ev['instrument_id'].astype('int64')
    keys = set(zip(ev['day'], ev['instrument_id']))
    if os.path.exists(f'{HERE}/ssr_rows.parquet'):
        parts = [pd.read_parquet(f'{HERE}/ssr_rows.parquet')]
        print('cached ssr rows', len(parts[0]))
        return _finish(ev, parts)
    parts = []
    for f in sorted(glob.glob(f'{RAW}/status_2*.dbn.zst')):
        if 'status_20260915' in f:
            continue
        d = db.DBNStore.from_file(f).to_df().reset_index(drop=True)
        d['et'] = pd.to_datetime(d['ts_event'], utc=True).dt.tz_convert('America/New_York')
        d['day'] = d['et'].dt.strftime('%Y-%m-%d')
        d['ssr'] = d['is_short_sell_restricted'].astype(str)
        d = d.loc[d['ssr'].isin(['Y', 'N']), ['day', 'et', 'instrument_id', 'ssr']]
        d = d[[(a, b) in keys for a, b in zip(d['day'], d['instrument_id'])]]
        parts.append(d)
        print(os.path.basename(f), len(d), flush=True)
        gc.collect()
    return _finish(ev, parts)


def _finish(ev, parts):
    ssr = pd.concat(parts, ignore_index=True).sort_values('et')
    ssr['instrument_id'] = ssr['instrument_id'].astype('int64')
    ssr.to_parquet(f'{HERE}/ssr_rows.parquet', index=False)
    del parts
    gc.collect()
    ev = ev.sort_values('resume_ts')
    m = pd.merge_asof(ev, ssr.rename(columns={'et': 'ssr_ts'}), left_on='resume_ts', right_on='ssr_ts',
                      by=['day', 'instrument_id'], direction='backward')
    m['ssr'] = m['ssr'].fillna('unknown')
    m[['day', 'instrument_id', 'halt_seq', 'resume_ts', 'symbol', 'ssr']].to_parquet(f'{HERE}/ssr_state.parquet', index=False)
    print(m['ssr'].value_counts(normalize=True).round(4).to_string())


if __name__ == '__main__':
    main()
