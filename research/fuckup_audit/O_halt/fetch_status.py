"""S1 step 1 — pull XNAS.ITCH `status` ALL_SYMBOLS month by month, keep only in-session
halt/resume rows, write one compact parquet per year.

Never materialises the raw feed: each month is streamed to a .dbn.zst on disk, converted to a
frame, filtered to the RTH window (ET 09:25-16:05) and to instruments that had a halt that day,
then the frame is freed. Cost is priced with metadata.get_cost BEFORE every request and the run
aborts if the running total would exceed BUDGET_USD.
"""
import os, sys, gc, json
from datetime import date
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
import databento as db
import pandas as pd

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
RAW = f'{HERE}/raw'
BUDGET_USD = 100.0
SPENT_SO_FAR = 0.0319 + 0.0  # the 2026-09-15 step-0 probe day

START = date(2025, 1, 1)
END = date(2026, 9, 18)          # exclusive-ish; get_range end is exclusive

ACT_PAUSE, ACT_TRADING, ACT_QUOTING, ACT_HALT, ACT_SUSPEND = 9, 7, 3, 8, 10
REASON_LULD = 50


def months(a: date, b: date):
    y, m = a.year, a.month
    while (y, m) < (b.year, b.month) or (y, m) == (b.year, b.month):
        nm = (y + (m == 12), 1 if m == 12 else m + 1)
        yield date(y, m, 1), date(nm[0], nm[1], 1)
        y, m = nm


def main():
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    spent = SPENT_SO_FAR
    ledger = []
    per_year = {}
    taxonomy = []

    for s, e in months(START, END):
        if s >= END:
            break
        e = min(e, END)
        tag = s.strftime('%Y%m')
        path = f'{RAW}/status_{tag}.dbn.zst'
        if not os.path.exists(path):
            cost = c.metadata.get_cost(dataset='XNAS.ITCH', schema='status', symbols='ALL_SYMBOLS',
                                       stype_in='raw_symbol', start=str(s), end=str(e))
            if spent + cost > BUDGET_USD:
                print(f"ABORT: {tag} would cost ${cost:.3f}, running ${spent:.2f} -> over ${BUDGET_USD}")
                break
            d = c.timeseries.get_range(dataset='XNAS.ITCH', schema='status', symbols='ALL_SYMBOLS',
                                       stype_in='raw_symbol', start=str(s), end=str(e), path=path)
            spent += cost
            ledger.append({'month': tag, 'cost': round(cost, 4), 'cum': round(spent, 4)})
            print(f"{tag} pulled  ${cost:.3f}  cum ${spent:.2f}", flush=True)
        else:
            d = db.DBNStore.from_file(path)
            print(f"{tag} cached", flush=True)

        df = d.to_df().reset_index(drop=True)
        et = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert('America/New_York')
        df['et'] = et
        df['day'] = et.dt.strftime('%Y-%m-%d')
        mins = et.dt.hour * 60 + et.dt.minute
        sess = (mins >= 565) & (mins <= 965)                      # 09:25 .. 16:05 ET
        s_df = df.loc[sess, ['day', 'et', 'instrument_id', 'action', 'reason', 'is_trading', 'is_quoting']]
        del df
        gc.collect()

        tx = s_df.groupby(['action', 'reason']).size().reset_index(name='n')
        tx['month'] = tag
        taxonomy.append(tx)

        # instruments that had ANY halting action that day -> keep their whole in-session tape
        halted = s_df[s_df['action'].isin([ACT_PAUSE, ACT_HALT, ACT_SUSPEND])][['day', 'instrument_id']].drop_duplicates()
        keep = s_df.merge(halted, on=['day', 'instrument_id'], how='inner')
        keep = keep[keep['action'] != 14]                          # SSR_CHANGE is not a trading state
        per_year.setdefault(s.year, []).append(keep)
        print(f"   in-session {len(s_df):,} -> halted-instrument rows {len(keep):,}", flush=True)
        del s_df, keep
        gc.collect()

    for y, parts in per_year.items():
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(f'{RAW}/halt_rows_{y}.parquet', index=False)
        print(f"wrote halt_rows_{y}.parquet  {len(out):,} rows")
    pd.concat(taxonomy, ignore_index=True).to_csv(f'{HERE}/status_taxonomy.csv', index=False)
    json.dump({'spent_usd': round(spent, 4), 'ledger': ledger}, open(f'{HERE}/spend.json', 'w'), indent=1)
    print(f"TOTAL SPENT ${spent:.2f}")


if __name__ == '__main__':
    main()
