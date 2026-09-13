#!/usr/bin/env python3
"""H21 data: FINRA consolidated short interest (bi-monthly, settlement-dated).
Public API, no key. Stored with a conservative `usable_from` = settlement +
13 calendar days (FINRA publishes ~T+9 trading days) so a trade may only
see the report already public on its date — point-in-time safe.
Output research/ignition_zero/short_interest.csv (resumable by offset)."""
import json, os, sys, time, requests, pandas as pd
OUT='research/ignition_zero/short_interest.csv'; STATE='research/ignition_zero/finra_state.json'
URL='https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest'
body={"limit":5000,"offset":0,"dateRangeFilters":[{"fieldName":"settlementDate","startDate":"2024-11-01","endDate":"2026-09-11"}],
      "fields":["symbolCode","settlementDate","currentShortPositionQuantity","previousShortPositionQuantity","averageDailyVolumeQuantity","daysToCoverQuantity","marketClassCode"]}
off=json.load(open(STATE))['offset'] if os.path.exists(STATE) else 0
rows=[]
if os.path.exists(OUT) and off>0: rows=pd.read_csv(OUT).to_dict('records')
while True:
    body['offset']=off
    r=None
    for att in range(4):
        try:
            r=requests.post(URL, headers={'Accept':'application/json','Content-Type':'application/json'}, data=json.dumps(body), timeout=60)
            if r.status_code==429: time.sleep(10); continue
            r.raise_for_status(); break
        except Exception as e:
            print('retry', att, str(e)[:80], flush=True); time.sleep(5); r=None
    if r is None: print('FAILED at offset', off, flush=True); break
    d=r.json()
    if not d: break
    rows.extend(d); off+=len(d)
    pd.DataFrame(rows).to_csv(OUT, index=False); json.dump({'offset':off}, open(STATE,'w'))
    print(f'{off} rows, latest {max(x["settlementDate"] for x in d)}', flush=True)
    if len(d)<5000: break
    time.sleep(0.5)
df=pd.DataFrame(rows); df['usable_from']=(pd.to_datetime(df.settlementDate)+pd.Timedelta(days=13)).dt.strftime('%Y-%m-%d'); df.to_csv(OUT, index=False)
print('DONE', len(df), 'rows, settlement', df.settlementDate.min(), '→', df.settlementDate.max(), '| symbols', df.symbolCode.nunique(), flush=True)
