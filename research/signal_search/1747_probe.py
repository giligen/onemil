"""Probe Alpaca corporate-actions depth for cash dividends (keys never printed)."""
import os, requests, json
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
for url in ('https://data.alpaca.markets/v1/corporate-actions', 'https://paper-api.alpaca.markets/v2/corporate_actions/announcements'):
    for (s, e) in (('2016-01-04', '2016-03-31'), ('2020-01-02', '2020-03-31'), ('2024-01-02', '2024-03-31')):
        if 'data.' in url:
            p = {'types': 'cash_dividend', 'start': s, 'end': e, 'limit': 1000}
        else:
            p = {'ca_types': 'dividend', 'since': s, 'until': e, 'date_type': 'ex_dividend'}
        r = requests.get(url, headers=H, params=p, timeout=60)
        txt = r.text
        print(url.split('/')[2], s, r.status_code, len(txt), txt[:400].replace('\n', ' '), flush=True)
