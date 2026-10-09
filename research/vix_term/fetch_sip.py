"""PREREG_2 adequacy: consolidated (SIP) daily bars for VXX/SVXY (cell 1 used IEX, whose daily close is IEX-only). -> data/<SYM>_daily_sip.csv"""
import logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
c = fetch._client()
for s in ('VXX', 'SVXY'):
    try:
        b = fetch.fetch_daily(c, s, 'sip'); fetch._atomic_csv(b, f'{s}_daily_sip.csv'); logging.info("%s sip %s..%s n=%d vol median %s", s, b.index.min().date(), b.index.max().date(), len(b), b.volume.median())
    except Exception as e:                                   # noqa: BLE001
        logging.error("SIP fetch failed for %s: %s", s, e)
