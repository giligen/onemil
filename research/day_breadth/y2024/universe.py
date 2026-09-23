"""2024H2 holdout universe (PREREG_2024.md) from the Databento EQUS.SUMMARY point-in-time daily bars.

Per day D in 2024-07-02..2024-12-31: open_D >= 1.03 x close_{D-1}, open_D in [$3, $50], volume_{D-1} >= 500,000
(prior row of the SAME symbol, like study_orb_broad's LAG), PIT instrument_class == 'K', not a ^Z[A-Z]ZZT$ test
ticker. The 09:30-09:35 RTH volume floor is applied later, from the fetched minute bars.
Writes candidates.csv (symbol, bar_date).

Usage: python3 research/day_breadth/y2024/universe.py
"""
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
EQUS = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
DEFS = sorted(glob.glob(str(ROOT / 'data/research/databento/pit_definition/def_2024*.parquet')))
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')


def build(daily: pd.DataFrame, class_k: set) -> pd.DataFrame:
    """Pure filter (unit-tested): gappers per the PREREG thresholds from a daily frame."""
    d = daily.sort_values(['symbol', 'bar_date']).copy()
    d['prev_close'] = d.groupby('symbol').close.shift(1)
    d['prev_vol'] = d.groupby('symbol').volume.shift(1)
    m = ((d.bar_date >= '2024-07-02') & (d.prev_close > 0) & (d.open >= 1.03 * d.prev_close)
         & (d.open >= 3.0) & (d.open <= 50.0) & (d.prev_vol >= 500_000)
         & d.symbol.isin(class_k) & ~d.symbol.str.match(TEST_TICKER))
    return d.loc[m, ['symbol', 'bar_date']].reset_index(drop=True)


def alpaca_symbol(sym: str) -> str:
    """Databento/CMS ticker -> the Alpaca format the 2025-26 universe used: warrants 'X+' -> 'X.WS'
    (pm_candidates has BBAI.WS, JOBY.WS, ...); class shares 'BF.B' unchanged."""
    return sym[:-1] + '.WS' if sym.endswith('+') else sym


def main():
    daily = pd.read_parquet(EQUS)
    daily['bar_date'] = daily.bar_date.astype(str)
    # Parity (disclosed in REPORT_2024): preferreds ('-' suffix) never appear in the 2025-26 universe -> excluded.
    daily = daily[~daily.symbol.str.contains('-', regex=False)]
    defs = pd.concat([pd.read_parquet(f, columns=['raw_symbol', 'instrument_class']) for f in DEFS])
    class_k = set(defs.loc[defs.instrument_class == 'K', 'raw_symbol'])
    no_def = set(daily.symbol) - set(defs.raw_symbol)
    out = build(daily, class_k)
    out['symbol'] = out.symbol.map(alpaca_symbol)
    out.to_csv(HERE / 'candidates.csv', index=False)
    print(f'EQUS 2024H2 rows {len(daily):,}, symbols {daily.symbol.nunique():,}, class-K symbols {len(class_k):,}, '
          f'symbols without a PIT definition (excluded) {len(no_def):,}')
    print(f'candidates {len(out):,} symbol-days over {out.bar_date.nunique()} days')
    print(out.groupby(out.bar_date.str[:7]).size().to_string())


if __name__ == '__main__':
    main()
