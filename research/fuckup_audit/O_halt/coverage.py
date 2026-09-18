"""S1 step 0c — venue coverage and the availability audit.

(1) Listed-venue mix of the symbols that carry an LULD pause in XNAS.ITCH `status`, joined to
    data/cache.db::universe.exchange (read-only). XNAS.ITCH is a single-venue feed; if NYSE/ARCA/
    AMEX-listed names never carry a pause the study is Nasdaq-listed-only and that caps everything.
(2) 20 known NYSE-listed megacaps: do they appear in the feed at all, and with what actions.
(3) Missingness of every field used in a decision, per split.
"""
import sqlite3, pandas as pd, numpy as np

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
CACHE = '/home/ec2-user/onemil/data/cache.db'
SPLITS = [('TRAIN', '2025-01-01', '2025-12-31'), ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-12-31')]


def main():
    ev = pd.read_parquet(f'{HERE}/luld_events_raw.parquet')
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
    uni = pd.read_sql('select symbol, exchange from universe', con)
    dbs = pd.read_sql('select distinct symbol from daily_bars', con)
    con.close()

    ev['in_daily_bars'] = ev['symbol'].isin(set(dbs['symbol']))
    m = ev.merge(uni, on='symbol', how='left')
    print('== listed-venue mix of LULD-halted SYMBOLS (distinct) ==')
    d = m.drop_duplicates('symbol')
    vc = d['exchange'].fillna('NOT_IN_UNIVERSE').value_counts()
    print((vc / vc.sum() * 100).round(2).to_string())
    print(f"\ndistinct halted symbols {d['symbol'].nunique()}   present in daily_bars "
          f"{d['in_daily_bars'].mean():.2%}")
    print('\n== listed-venue mix of LULD halt EVENTS ==')
    vc2 = m['exchange'].fillna('NOT_IN_UNIVERSE').value_counts()
    print((vc2 / vc2.sum() * 100).round(2).to_string())

    # universe-wide venue mix, for the comparison that answers "does the feed carry NYSE listings"
    print('\n== the tradeable universe itself, for reference ==')
    u = uni['exchange'].value_counts()
    print((u / u.sum() * 100).round(2).to_string())

    for nm, a, b in SPLITS:
        s = ev[(ev['day'] >= a) & (ev['day'] <= b)]
        wk = pd.to_datetime(s['day']).dt.to_period('W').nunique() if len(s) else 0
        print(f'{nm}: events {len(s):,}  days {s["day"].nunique()}  weeks {wk}  '
              f'events/week {len(s)/wk if wk else 0:.1f}')

    tr = pd.read_parquet(f'{HERE}/trades.parquet')
    print('\n== availability of every decision field (post-screen trades) ==')
    for c in ['ref', 'fill', 'pre_ret', 'prev_close', 'adv20', 'px_h5', 'px_h30', 'px_eod']:
        print(f'  {c:11s} non-null {tr[c].notna().mean():.4%}')
    print('\n== timestamp ordering (the availability proof) ==')
    print(f"  halt < resume on all events: {(tr['resume_ts'] > tr['halt_ts']).all()}")
    print(f"  entry bar opens AFTER the resume message on all trades: {(tr['entry_t'] > tr['resume_ts']).all()}")
    print(f"  resume->entry-bar-open seconds: min {tr['react_s'].min():.0f} "
          f"median {tr['react_s'].median():.0f} max {tr['react_s'].max():.0f}")
    print('\n== bar source per split (price-scale check: both stores are Alpaca SIP) ==')
    for nm, a, b in SPLITS:
        s = tr[(tr['day'] >= a) & (tr['day'] <= b)]
        print(f'  {nm}: {s["src"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
