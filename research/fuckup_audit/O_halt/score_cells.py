"""S1 step 3 — universe screens, entry/exit simulation, the 12 pre-declared cells, gates.

PRE-REGISTRATION (frozen before TEST was read):
  population  every LULD halt that RESUMED, on a symbol present in data/cache.db::daily_bars with
              prev close >= $5 and ADV20 >= 100K (both from daily bars strictly BEFORE the halt
              date), test tickers ^Z[A-Z]ZZT$ removed.
  side        up-halt if the 5-minute return into the halt is > 0, down-halt if < 0 (the tape's own
              direction; the status feed does not carry the band side).
  cells       side {up,down} x rule {continuation, fade} x horizon {+5m, +30m, 15:55} = 12.
  entry       THE ENGINE'S convention: the OPEN of the first 1-minute bar that starts strictly after
              the resume message, accepted only if it is within 0.6% of `ref` (the last trade price
              before the halt) on the side we are buying/selling. No touch fills. Never a level.
  R           declared flat at 2.0% of the fill price for every cell (there is no stop in this stage;
              R is only the scale in which cost and return are expressed). Raw % return reported next
              to it so the two are separable.
  cost        cost_curve.md band x hour median NBBO spread; half = 0.5*spread_pct/r_pct;
              entry 0.25*half (next-open fill, H7 correction), exit 0.875*half for the +5m/+30m
              crossings and 0.412*half for the 15:55 close.
  splits      TRAIN 2025-01-01..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..end of bars.
  gates       G1 TRAIN mean net R > 0 and t >= 2.0 · G2 VAL mean net R > 0 and >= 55% of weeks green
              · G3 TEST read once, after both.
"""
import os, json, re, sqlite3, sys
import numpy as np
import pandas as pd

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
CACHE = '/home/ec2-user/onemil/data/cache.db'
SIP = '/home/ec2-user/onemil/research/bf_zero/bars_sip.db'
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')

R_PCT = 2.0                    # declared R, % of fill price
CAP = 0.006                    # the engine's no-chase cap
MIN_PRICE, MIN_ADV = 5.0, 100_000

SPREAD_BPS = {                 # cost_curve.md, median NBBO spread in the signal minute
    '$5-10':   [35.842, 34.904, 29.806, 26.559, 22.510],
    '$10-20':  [43.795, 42.247, 28.795, 21.708, 28.555],
    '$20-50':  [51.634, 46.045, 45.305, 32.180, 34.876],
    '$50-200': [66.772, 63.442, 39.216, 30.341, 36.355],
    '$200+':   [54.322, 74.316, 76.113, 63.994, 68.163],
}
SPLITS = [('TRAIN', '2025-01-01', '2025-12-31'), ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-12-31')]


def band(p):
    return '$5-10' if p < 10 else '$10-20' if p < 20 else '$20-50' if p < 50 else '$50-200' if p < 200 else '$200+'


def hourbucket(minute_of_day):
    return 0 if minute_of_day < 575 else 1 if minute_of_day < 600 else 2 if minute_of_day < 660 else 3 if minute_of_day < 780 else 4


class Bars:
    """1-minute bars: the SIP tape we own first, Alpaca's cache second. Both store UTC ISO minutes."""

    def __init__(self):
        self.sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=60)
        self.alp = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
        self.src = {}

    def get(self, sym, day):
        rows = self.sip.execute('select t,o,h,l,c,v from bars where symbol=? and day=? order by t',
                                (sym, day)).fetchall()
        src = 'sip'
        if len(rows) < 30:
            lo, hi = f'{day}T00:00:00+00:00', f'{day}T23:59:59+00:00'
            a = self.alp.execute('select timestamp,open,high,low,close,volume from intraday_bars_1min '
                                 'where symbol=? and timestamp>=? and timestamp<=? order by timestamp',
                                 (sym, lo, hi)).fetchall()
            if len(a) > len(rows):
                rows, src = a, 'alpaca'
        if not rows:
            return None, 'none'
        df = pd.DataFrame(rows, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'], utc=True, format='mixed').dt.tz_convert('America/New_York')
        return df.set_index('t'), src

    def daily_prior(self, sym, day):
        """(prev close, ADV20) from daily_bars strictly before `day` — the engine's own definition,
        made point-in-time: the latest <=20 rows inside 45 calendar days, >= 5 rows required."""
        lo = (pd.Timestamp(day) - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
        r = self.alp.execute('select close, volume from daily_bars where symbol=? and bar_date<? and bar_date>=? '
                             'order by bar_date desc limit 20', (sym, day, lo)).fetchall()
        if len(r) < 5:
            return None, None
        return float(r[0][0]), float(np.mean([x[1] for x in r]))


def simulate(ev, bars):
    out, diag = [], {'no_bars': 0, 'no_daily': 0, 'screen_price': 0, 'screen_adv': 0,
                     'no_ref': 0, 'no_entry_bar': 0, 'stale_entry': 0, 'flat_pre': 0, 'test_ticker': 0}
    seen = {}
    for e in ev.itertuples():
        sym, day = e.symbol, e.day
        if not sym or TEST_TICKER.match(sym):
            diag['test_ticker'] += 1
            continue
        if (sym, day) not in seen:
            seen[(sym, day)] = bars.get(sym, day)
        bd, src = seen[(sym, day)]
        pc, adv = bars.daily_prior(sym, day)
        if pc is None:
            diag['no_daily'] += 1
            continue
        if pc < MIN_PRICE:
            diag['screen_price'] += 1
            continue
        if adv < MIN_ADV:
            diag['screen_adv'] += 1
            continue
        if bd is None or bd.empty:
            diag['no_bars'] += 1
            continue
        pre = bd[bd.index < e.halt_ts]
        if len(pre) < 6:
            diag['no_ref'] += 1
            continue
        ref = float(pre['c'].iloc[-1])
        t5 = e.halt_ts - pd.Timedelta(minutes=5)
        base = pre[pre.index <= t5]
        base_px = float(base['c'].iloc[-1]) if len(base) else float(pre['c'].iloc[0])
        pre_ret = ref / base_px - 1.0
        if abs(pre_ret) < 1e-9:
            diag['flat_pre'] += 1
            continue
        side = 'up' if pre_ret > 0 else 'down'

        t_entry = e.resume_ts.floor('min') + pd.Timedelta(minutes=1)
        post = bd[bd.index >= t_entry]
        if post.empty:
            diag['no_entry_bar'] += 1
            continue
        eb_t = post.index[0]
        if (eb_t - t_entry) > pd.Timedelta(minutes=2):    # no print in the first minutes: no order would fill
            diag['stale_entry'] += 1
            continue
        if eb_t.hour * 60 + eb_t.minute > 955:            # past 15:55 — no time to hold
            diag['no_entry_bar'] += 1
            continue
        fill = float(post['o'].iloc[0])
        assert float(post['l'].iloc[0]) <= fill <= float(post['h'].iloc[0])

        row = {'day': day, 'symbol': sym, 'src': src, 'side': side, 'halt_seq': e.halt_seq,
               'halt_ts': e.halt_ts, 'resume_ts': e.resume_ts, 'entry_t': eb_t,
               'react_s': (eb_t - e.resume_ts).total_seconds(),
               'ref': ref, 'fill': fill, 'pre_ret': pre_ret, 'prev_close': pc, 'adv20': adv,
               'gap_pct': fill / ref - 1.0}
        # exits, per horizon: the CLOSE of the bar `h` minutes after the entry bar, truncated at 15:55
        for name, mins in (('h5', 5), ('h30', 30)):
            tgt = eb_t + pd.Timedelta(minutes=mins)
            w = post[post.index <= min(tgt, eb_t.normalize() + pd.Timedelta(hours=15, minutes=55))]
            row[f'px_{name}'] = float(w['c'].iloc[-1]) if len(w) else fill
        w = post[post.index <= eb_t.normalize() + pd.Timedelta(hours=15, minutes=55)]
        row['px_eod'] = float(w['c'].iloc[-1]) if len(w) else fill
        out.append(row)
    return pd.DataFrame(out), diag


def score(tr, rule, horizon):
    """Net R and raw % for one (rule, horizon) on the trades frame `tr` (already one side)."""
    direction = np.where((tr['side'] == 'up') == (rule == 'continuation'), 1.0, -1.0)
    # cap: never chase. long -> fill <= ref*(1+CAP); short -> fill >= ref*(1-CAP)
    ok = np.where(direction > 0, tr['fill'] <= tr['ref'] * (1 + CAP), tr['fill'] >= tr['ref'] * (1 - CAP))
    px = {'h5': tr['px_h5'], 'h30': tr['px_h30'], 'close': tr['px_eod']}[horizon]
    raw = direction * (px.values / tr['fill'].values - 1.0) * 100.0
    mod = tr['entry_t'].dt.hour * 60 + tr['entry_t'].dt.minute
    sbps = np.array([SPREAD_BPS[band(p)][hourbucket(m)] for p, m in zip(tr['fill'], mod)])
    half = 0.5 * (sbps / 100.0) / R_PCT
    exit_mult = 0.412 if horizon == 'close' else 0.875
    net = raw / R_PCT - 0.25 * half - exit_mult * half
    return pd.DataFrame({'day': tr['day'].values, 'symbol': tr['symbol'].values, 'ok': ok,
                         'raw_pct': raw, 'net_R': net, 'gross_R': raw / R_PCT})


def stats(x):
    n = len(x)
    if n < 3:
        return dict(n=n, mean=np.nan, t=np.nan, mde=np.nan, sd=np.nan)
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    return dict(n=n, mean=float(np.mean(x)), t=float(np.mean(x) / se) if se else np.nan,
                mde=float(2.8 * se), sd=sd)


def weeks_green(df):
    if df.empty:
        return np.nan, 0
    wk = pd.to_datetime(df['day']).dt.to_period('W')
    g = df.groupby(wk)['net_R'].sum()
    return float((g > 0).mean()), len(g)


def main():
    ev = pd.read_parquet(f'{HERE}/luld_events_raw.parquet')
    ev['halt_ts'] = pd.to_datetime(ev['halt_ts'], utc=True).dt.tz_convert('America/New_York')
    ev['resume_ts'] = pd.to_datetime(ev['resume_ts'], utc=True).dt.tz_convert('America/New_York')
    assert (ev['resume_ts'] > ev['halt_ts']).all(), 'resume before halt — timestamp ordering broken'
    print(f'events in  {len(ev):,}')
    bars = Bars()
    tr, diag = simulate(ev, bars)
    for c in ('halt_ts', 'resume_ts', 'entry_t'):
        tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_convert('America/New_York')
    print('attrition:', diag)
    print(f'tradeable events {len(tr):,}')
    tr.to_parquet(f'{HERE}/trades.parquet', index=False)
    print('bar source mix:', tr['src'].value_counts().to_dict())
    print('reaction lag resume->entry-bar-open (s):', tr['react_s'].describe()[['min', '50%', 'max']].round(1).to_dict())

    rows = []
    for split, a, b in SPLITS:
        s = tr[(tr['day'] >= a) & (tr['day'] <= b)]
        for side in ('up', 'down'):
            ss = s[s['side'] == side]
            for rule in ('continuation', 'fade'):
                for hz in ('h5', 'h30', 'close'):
                    sc = score(ss, rule, hz)
                    kept = sc[sc['ok']]
                    st = stats(kept['net_R'].values)
                    wg, nw = weeks_green(kept)
                    gst = stats(kept['gross_R'].values)
                    x = np.sort(kept['net_R'].values)
                    ex5 = stats(x[:int(len(x) * 0.95)]) if len(x) > 20 else dict(mean=np.nan)
                    cap3 = stats(np.minimum(kept['net_R'].values, 3.0)) if len(x) > 2 else dict(mean=np.nan)
                    rows.append(dict(split=split, side=side, rule=rule, horizon=hz,
                                     events=len(sc), filled=int(sc['ok'].sum()),
                                     fill_rate=round(float(sc['ok'].mean()), 4) if len(sc) else np.nan,
                                     n=st['n'], mean_netR=st['mean'], t=st['t'], mde=st['mde'],
                                     mean_grossR=gst['mean'], mean_raw_pct=float(kept['raw_pct'].mean()) if len(kept) else np.nan,
                                     weeks_green=wg, weeks=nw,
                                     ex_top5_netR=ex5['mean'], cap3R_netR=cap3['mean']))
    res = pd.DataFrame(rows)
    res.to_csv(f'{HERE}/cells.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        for split, _, _ in SPLITS:
            print(f'\n===== {split} =====')
            print(res[res.split == split].drop(columns=['split']).round(4).to_string(index=False))

    # search-adjusted permutation on TRAIN: symmetric sign-flip null, max |t| over the 12 cells
    rng = np.random.default_rng(7)
    s = tr[(tr['day'] >= SPLITS[0][1]) & (tr['day'] <= SPLITS[0][2])]
    cells = []
    for side in ('up', 'down'):
        ss = s[s['side'] == side]
        for rule in ('continuation', 'fade'):
            for hz in ('h5', 'h30', 'close'):
                sc = score(ss, rule, hz)
                cells.append(sc[sc['ok']]['net_R'].values)
    obs = max(abs(stats(c)['t']) for c in cells if len(c) > 2)
    B, hits = 2000, 0
    for _ in range(B):
        mx = 0.0
        for c in cells:
            if len(c) < 3:
                continue
            f = rng.choice([-1.0, 1.0], size=len(c))
            y = c * f
            se = np.std(y, ddof=1) / np.sqrt(len(y))
            mx = max(mx, abs(np.mean(y) / se) if se else 0.0)
        hits += mx >= obs
    perm_p = (hits + 1) / (B + 1)
    print(f'\nTRAIN max|t| = {obs:.2f}   search-adjusted permutation p = {perm_p:.4f}  (12 cells, B={B})')
    json.dump({'train_max_abs_t': round(float(obs), 4), 'perm_p_12cells': round(float(perm_p), 4),
               'attrition': diag, 'n_trades': int(len(tr))},
              open(f'{HERE}/score_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
