"""Independent rebuild of cells 1,709a / 1,709b from PREREG_1709.md prose only.

Re-simulates the ORB live exit rule (stop = max(range_low, entry - 0.25*ATR14), static lock
arm 1.75*range_size -> stop entry+0.5*range_size, touchgo Rule M / Rule D, force close 15:45 ET)
on 1-min bars, from (i) the BT entry, (ii) the live fill (1,709a) or the replay fill (1,709b).

Run:  nice -n 19 python3 research/orb_exit_resim/rebuild_1709.py
Outputs: rebuild_1709a.csv, rebuild_1709b.csv, rebuild_1709_tables.txt (tables for REBUILD_1709.md).
Bar DBs and daily parquets are opened read-only; nothing else is written.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_exit_resim'
EXEC = ROOT / 'research/exec_learnings_oct2026'
SIP = ROOT / 'research/bf_zero/bars_sip.db'
Y24 = ROOT / 'research/day_breadth/y2024/bars.db'
TRADES_DB = ROOT / 'data/trades.db'
DAILY_PARQUETS = [ROOT / 'data/research/databento/equs_daily_2024H2.parquet',
                  ROOT / 'data/research/databento/equs_daily_2025_2026.parquet']

SLIP = 0.0035            # stops fill at stop - 35 bps (cell 1,443)
BUF = 1.003              # BT entry = range_high * 1.003
ATR_K = 0.25
LOCK_ARM, LOCK_STOP = 1.75, 0.5   # in units of range_size
FORCE_MIN = 15 * 60 + 45
RANGE_START, RANGE_END = 9 * 60 + 30, 9 * 60 + 35
RULE_M_THR, RULE_D_REV, RULE_D_EXIT = 0.5, 0.75, -0.5
MAX_BB_AGE = 15
SCALE_FRAC, SCALE_LEVEL = 0.40, 3.0
BOOT = 4000
rng = np.random.default_rng(1709)

_conn = {}


def ro(path):
    """Read-only sqlite connection (cached)."""
    if path not in _conn:
        _conn[path] = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    return _conn[path]


def load_bars(sym, day):
    """1-min bars (regular + premarket) for symbol-day; sip db first, y2024 db second.
    Returns (DataFrame with mod/o/h/l/c, source) or (None, None)."""
    for src, path in (('sip', SIP), ('y2024', Y24)):
        df = pd.read_sql('select t,o,h,l,c from bars where symbol=? and day=? order by t',
                         ro(path), params=(sym, day))
        if len(df):
            ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
            df['mod'] = ts.dt.hour * 60 + ts.dt.minute
            df = df[(df['mod'] >= RANGE_START) & (df['mod'] < 16 * 60)].reset_index(drop=True)
            if len(df):
                return df, src
    return None, None


def opening_range(bars):
    """(range_high, range_low) from the 09:30-09:34 bars, or (None, None)."""
    r = bars[(bars['mod'] >= RANGE_START) & (bars['mod'] < RANGE_END)]
    if r.empty:
        return None, None
    return float(r['h'].max()), float(r['l'].min())


_daily = None


def atr14(sym, day):
    """ATR14 ending T-1 (mean of the last 14 true ranges) from the Databento daily parquets."""
    global _daily
    if _daily is None:
        _daily = pd.concat([pd.read_parquet(p, columns=['bar_date', 'symbol', 'high', 'low', 'close'])
                            for p in DAILY_PARQUETS]).drop_duplicates(['symbol', 'bar_date'])
        _daily['bar_date'] = _daily['bar_date'].astype(str)
        _daily = _daily.sort_values(['symbol', 'bar_date'])
        _daily = {s: g for s, g in _daily.groupby('symbol')}
    g = _daily.get(sym)
    if g is None:
        return None
    g = g[g['bar_date'] < day].tail(15)
    if len(g) < 15:
        return None
    pc = g['close'].shift(1)
    tr = np.maximum(g['high'] - g['low'], np.maximum((g['high'] - pc).abs(), (g['low'] - pc).abs()))
    return float(tr.iloc[1:].mean())


def simulate(bars, entry, entry_mod, rhigh, rlow, stop, entry_can_stop=True):
    """Walk bars after the fill. Returns dict(reason, px, mod, R, R_scale) or None (invalid).
    Order inside a bar: stop first (conservative), then bar-close rules, then lock arming
    (lock stop live from the NEXT bar). Entry bar cannot arm the lock or scale."""
    rs = rhigh - rlow
    risk = entry - stop
    if risk <= 0 or rs <= 0:
        return None
    m = bars['mod'].values
    o, h, l, c = (bars[k].values for k in 'ohlc')
    idx = np.nonzero(m >= entry_mod)[0]
    if len(idx) == 0:
        return None
    e = idx[0]                         # entry bar index
    bb_idx = np.nonzero((m >= RANGE_END) & (h > rhigh))[0]
    bb_mod = int(m[bb_idx[0]]) if len(bb_idx) else None
    dec_mod = None                     # bar-close at which Rule M is decided
    if bb_mod is not None and int(m[e]) - bb_mod <= MAX_BB_AGE:
        dec_mod = max(bb_mod, int(m[e]))
        bbi = np.nonzero(m == bb_mod)[0][0]
        rng_ = h[bbi] - l[bbi]
        rule_m = rng_ > 0 and (c[bbi] - l[bbi]) / rng_ < RULE_M_THR
    else:
        rule_m = False
    b1_mod = (max(bb_mod, int(m[e])) + 1) if bb_mod is not None else None
    cur_stop, locked, scaled = stop, False, False
    arm_px = entry + LOCK_ARM * rs
    lock_px = entry + LOCK_STOP * rs
    scale_px = entry + SCALE_LEVEL * rs
    res = None
    for i in range(e, len(m)):
        if m[i] >= FORCE_MIN:
            res = ('force_close', float(o[i]), int(m[i]))
            break
        if (i > e or entry_can_stop) and l[i] <= cur_stop:
            px = (min(cur_stop, o[i]) if i > e else cur_stop) * (1 - SLIP)
            res = ('lock_stop' if locked else 'stop_loss', px, int(m[i]))
            break
        if dec_mod is not None and rule_m and m[i] == dec_mod:
            res = ('tag_bb', float(c[i]), int(m[i]))
            break
        if b1_mod is not None and m[i] == b1_mod and i > e - 1 and (entry - l[i]) / rs >= RULE_D_REV:
            res = ('tag_bb', entry + RULE_D_EXIT * rs, int(m[i]))
            break
        if i > e:
            if not scaled and h[i] >= scale_px:
                scaled = True
            if not locked and h[i] >= arm_px:
                locked, cur_stop = True, lock_px
    if res is None:
        res = ('force_close', float(c[-1]), int(m[-1]))
    reason, px, xm = res
    R = (px - entry) / risk
    # sensitivity: engine's 40 % scale-out at +3 range-R (touch fill) -- not in the PREREG text
    sc_hit = scaled and not (reason == 'stop_loss' and False)
    if sc_hit:
        R_scale = SCALE_FRAC * (scale_px - entry) / risk + (1 - SCALE_FRAC) * R
    else:
        R_scale = R
    return dict(reason=reason, px=px, mod=xm, R=R, R_scale=R_scale)


def cluster_ci(df, col='R'):
    """Day-clustered bootstrap: mean, lo, hi (95 %), SE. Resamples days with replacement."""
    d = df.dropna(subset=[col])
    if d.empty:
        return (np.nan,) * 4
    g = d.groupby('date')[col].agg(['sum', 'count'])
    s, n = g['sum'].values, g['count'].values
    k = len(g)
    ix = rng.integers(0, k, size=(BOOT, k))
    means = s[ix].sum(1) / n[ix].sum(1)
    return float(d[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.std())


def exTop(x, q=0.05):
    """Mean after dropping the top q share of rows by value."""
    x = np.sort(np.asarray(x, float))
    k = int(np.ceil(len(x) * q))
    return float(x[:len(x) - k].mean()) if len(x) > k else np.nan


def cls(r):
    """Map exit reason to {stop, lock, tag, force}."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return None
    r = str(r)
    if r.startswith('stop'):
        return 'stop'
    if r.startswith('lock') or r == 'scale_lock':
        return 'lock'
    if r.startswith('tag'):
        return 'tag'
    return 'force'


def build_a(log):
    """Cell 1,709a on the live ledger."""
    led = pd.read_csv(EXEC / 'fills_1702.csv')
    pd_ = pd.read_sql("select order_id, pattern_data from trades where strategy like 'orb%'", ro(TRADES_DB))
    pdm = {r.order_id: json.loads(r.pattern_data) for r in pd_.itertuples() if r.pattern_data}
    rows = []
    for r in led.itertuples():
        p = pdm.get(r.order_id, {})
        rec = dict(date=r.date, symbol=r.symbol, ledger_reason=r.exit_reason, ledger_R=r.R,
                   ledger_exit=r.exit_price, spread_bps=r.spread_bps, entry_live=r.entry, stop_ledger=r.stop,
                   pd_rh=p.get('range_high'), pd_rl=p.get('range_low'), pd_atr=p.get('atr14'),
                   status='ok', bars_ok_ledger=bool(r.bars_ok))
        if not r.bars_ok:
            rec['status'] = 'after_bar_db_end'
            rows.append(rec)
            continue
        bars, src = load_bars(r.symbol, r.date)
        if bars is None:
            rec['status'] = 'no_bars'
            rows.append(rec)
            continue
        rh, rl = opening_range(bars)
        rec.update(src=src, rh_bars=rh, rl_bars=rl)
        if p.get('range_high') is None:
            rec['status'] = 'no_pattern_data'
            rows.append(rec)
            continue
        rh, rl = p['range_high'], p['range_low']      # pattern_data is authoritative
        at = p.get('atr14')
        rec['atr_used'] = at
        fill_ts = pd.to_datetime(r.filled_at, utc=True).tz_convert('America/New_York')
        fmod = fill_ts.hour * 60 + fill_ts.minute
        ex_ts = pd.to_datetime(r.exited_at, utc=True).tz_convert('America/New_York')
        rec['ledger_exit_mod'] = ex_ts.hour * 60 + ex_ts.minute
        for tag, ent in (('i', rh * BUF), ('ii', r.entry)):
            stop = max(rl, ent - ATR_K * at) if at else rl
            if tag == 'i':
                # BT entry: first bar from 09:35 whose high reaches the entry
                hit = bars[(bars['mod'] >= RANGE_END) & (bars['h'] >= ent)]
                if hit.empty:
                    rec['reason_i'] = 'no_fill'
                    continue
                emod = int(hit['mod'].iloc[0])
            else:
                emod = fmod
            rec[f'stop_{tag}'] = stop
            for k, can in (('', True), ('_nes', False)):
                s = simulate(bars, ent, emod, rh, rl, stop, can)
                if s is None:
                    continue
                rec[f'reason_{tag}{k}'] = s['reason']
                rec[f'R_{tag}{k}'] = s['R']
                rec[f'Rscale_{tag}{k}'] = s['R_scale']
                rec[f'exitpx_{tag}{k}'] = s['px']
                rec[f'exitmod_{tag}{k}'] = s['mod']
                if k == '':
                    # decision-time half-spread charged on non-stop exits (row quote, no band)
                    adj = 0 if s['reason'] in ('stop_loss', 'lock_stop') else (r.spread_bps / 2e4) * s['px']
                    rec[f'Rnet_{tag}'] = (s['px'] - adj - ent) / (ent - stop)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rebuild_1709a.csv', index=False)
    log(f'1709a rows {len(df)}; status {df.status.value_counts().to_dict()}')
    return df


def table_a(df, log):
    """Tables for 1,709a."""
    d = df[df.status == 'ok'].copy()
    d = d[d['reason_ii'].notna() & d['reason_i'].notna() & (d['reason_i'] != 'no_fill')]
    n = len(d)
    log(f'1709a: n(live, bars ok, both sims) = {n}  of {len(df)} ledger rows')
    log('validation: pattern_data range vs 09:30-09:34 bar range, share equal within 1c: '
        f"{np.mean((abs(d.pd_rh - d.rh_bars) < 0.011) & (abs(d.pd_rl - d.rl_bars) < 0.011)):.3f}")
    ok_atr = d[d.pd_atr.notna()]
    log(f'  rows with pattern atr14: {len(ok_atr)}; ledger stop equals max(rl, entry-.25atr)? '
        f"{np.mean(abs(ok_atr.stop_ledger - ok_atr.stop_ii) < 0.02):.3f}; atr parquet vs pattern atr (median ratio) "
        f"{np.nanmedian([atr14(r.symbol, r.date) / r.pd_atr for r in ok_atr.itertuples() if atr14(r.symbol, r.date)]):.3f}")
    share = lambda col: (d[col].map(cls) == 'stop').mean()
    led_s, i_s, ii_s = share('ledger_reason'), share('reason_i'), share('reason_ii')
    log(f'stop share  ledger {led_s:.3f}  re-sim(i) {i_s:.3f}  re-sim(ii) {ii_s:.3f}  '
        f'(ii nes {(d.reason_ii_nes.map(cls) == "stop").mean():.3f})')
    for lab, base in (('vs fixed BT 28.4 %', 0.284), ('vs own (i)', i_s)):
        exc = led_s - base
        log(f'  excess {exc * 100:.1f} pp {lab}; (ii)-(i) = {(ii_s - i_s) * 100:.1f} pp -> explained '
            f'{(ii_s - i_s) / exc * 100:.0f} %')
    for c in ('stop', 'lock', 'tag', 'force'):
        log(f'  share {c}: ledger {(d.ledger_reason.map(cls) == c).mean():.3f} '
            f'(i) {(d.reason_i.map(cls) == c).mean():.3f} (ii) {(d.reason_ii.map(cls) == c).mean():.3f}')
    for k in ('R_i', 'R_ii', 'R_ii_nes', 'Rnet_ii', 'Rscale_ii', 'ledger_R'):
        m, lo, hi, se = cluster_ci(d, k)
        log(f'  meanR {k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  ex-top5 {exTop(d[k].dropna()):+.3f}')
    d['led'], d['sim'] = d.ledger_reason.map(cls), d.reason_ii.map(cls)
    cm = pd.crosstab(d.led, d.sim)
    log('confusion (rows=ledger, cols=re-sim ii):\n' + cm.to_string())
    agree = (d.led == d.sim).mean()
    log(f'exit-class agreement (ii vs ledger): {agree:.3f}  (stop vs not-stop agreement '
        f'{((d.led == "stop") == (d.sim == "stop")).mean():.3f})')
    log(f'  agreement (i vs ledger): {(d.led == d.reason_i.map(cls)).mean():.3f}')
    fc = d[d.led == 'force']
    log(f'  force_close ledger minute median {fc.ledger_exit_mod.median()} (=15:45); sim uses open of first bar >= 15:45; '
        f'median |px diff| bps {np.median(abs(fc.exitpx_ii / fc.ledger_exit - 1)[fc.sim == "force"]) * 1e4:.1f}')
    return d


def build_b(log):
    """Cell 1,709b on the replay population (BT pop + replay fills at 0 s and 30 s)."""
    pop = pd.read_csv(ROOT / 'research/orb_latency_bt/population.csv')
    rep = pd.read_csv(EXEC / 'fills_1708.csv')
    rep = rep[rep.delay_s.isin([0, 30])]
    rows = []
    for p in pop.itertuples():
        bars, src = load_bars(p.symbol, p.date)
        base = dict(date=p.date, symbol=p.symbol, period=p.period, bt_reason=p.exit_reason, bt_win=p.win,
                    src=src, covered=bars is not None)
        calc = {}
        if bars is not None:
            rh, rl = opening_range(bars)
            at = atr14(p.symbol, p.date)
            base.update(rh=rh, rl=rl, atr=at)
            # BT-entry re-sim (entry = pop entry price, bar = first bar from 09:35 with high >= trigger)
            hit = bars[(bars['mod'] >= RANGE_END) & (bars['h'] >= p.trigger)]
            if rh and rl and not hit.empty:
                for tg, use in (('', False), ('_fl', True)):
                    stop = max(rl, p.entry_price - ATR_K * at) if (at and use) else rl
                    s = simulate(bars, p.entry_price, int(hit['mod'].iloc[0]), rh, rl, stop)
                    if s:
                        base.update({f'bt_resim_reason{tg}': s['reason'], f'bt_resim_R{tg}': s['R']})
        for d in (0, 30):
            r = rep[(rep.symbol == p.symbol) & (rep.date == p.date) & (rep.delay_s == d)]
            rec = dict(base, delay_s=d, status=r.status.iloc[0] if len(r) else 'absent',
                       fill_price=r.fill_price.iloc[0] if len(r) else np.nan)
            if len(r) and r.status.iloc[0] == 'filled' and bars is not None and rec.get('rh'):
                fmod = RANGE_END + int(max(r.t_star.iloc[0], d) // 60)
                ent = float(rec['fill_price'])
                at = rec['atr']
                # base: stop = range_low (PREREG: no stored atr14 -> range_low; live ledger stop == range_low on
                # 96 % of rows). '_fl' = sensitivity with the ATR floor from the Databento daily parquet.
                for tg, use in (('', False), ('_fl', True)):
                    stop = max(rec['rl'], ent - ATR_K * at) if (at and use) else rec['rl']
                    for k, can in (('', True), ('_nes', False)):
                        s = simulate(bars, ent, fmod, rec['rh'], rec['rl'], stop, can)
                        if s:
                            rec.update({f'reason{tg}{k}': s['reason'], f'R{tg}{k}': s['R'],
                                        f'Rscale{tg}{k}': s['R_scale']})
                rec['atr_used'] = bool(at)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rebuild_1709b.csv', index=False)
    log(f'1709b rows {len(df)}')
    return df


WEEKS = {'2023-01..24H1': 78, '2024H2': 26, '2025H1': 26, '2025H2-2026-09': 61}


def seg(r):
    """Period label with the 2023-24 block split where bar coverage starts (2024-07)."""
    if r.period == '2023-24':
        return '2024H2' if r.date >= '2024-07-01' else '2023-01..24H1'
    return r.period


def table_b(df, log):
    """Tables for 1,709b."""
    df['seg'] = df.apply(seg, axis=1)
    filled = df[df.status == 'filled']
    for d in (0, 30):
        log(f'--- delay {d} s ---')
        f = filled[filled.delay_s == d]
        log('period | n_filled | covered(share) | winner-cov - loser-cov (pp) | n_sim | meanR [CI] | ex-top5 | cap3R | '
            'stop sh | fills/wk(covered)')
        for s in WEEKS:
            x = f[f.seg == s]
            if x.empty:
                continue
            cov = x[x.R.notna()]
            w = x[x.bt_win == 1].R.notna().mean() if (x.bt_win == 1).any() else np.nan
            lz = x[x.bt_win != 1].R.notna().mean() if (x.bt_win != 1).any() else np.nan
            m, lo, hi, _ = cluster_ci(cov, 'R') if len(cov) else (np.nan,) * 4
            log(f'{s} | {len(x)} | {len(cov) / len(x):.2f} | {(w - lz) * 100:+.1f} | {len(cov)} | {m:+.3f} '
                f'[{lo:+.3f},{hi:+.3f}] | {exTop(cov.R) if len(cov) > 3 else np.nan:+.3f} | '
                f'{np.minimum(cov.R, 3).mean() if len(cov) else np.nan:+.3f} | '
                f'{(cov.reason.map(cls) == "stop").mean() if len(cov) else np.nan:.3f} | {len(cov) / WEEKS[s]:.2f}')
        # pooled in-regime and R sensitivities
        for lab, segs in (('in-regime 2025-01..26-09', ['2025H1', '2025H2-2026-09']),
                          ('2024H2 (covered out-of-regime)', ['2024H2']),
                          ('all covered', list(WEEKS))):
            cov = f[f.seg.isin(segs) & f.R.notna()]
            if cov.empty:
                continue
            m, lo, hi, _ = cluster_ci(cov, 'R')
            m2, lo2, hi2, _ = cluster_ci(cov, 'R_nes')
            m3, lo3, hi3, _ = cluster_ci(cov, 'Rscale')
            mf, lof, hif, _ = cluster_ci(cov, 'R_fl')
            log(f'{lab} [ATR-floor sens]: R {mf:+.3f} [{lof:+.3f},{hif:+.3f}] ex-top5 {exTop(cov.R_fl):+.3f} '
                f'stop sh {(cov.reason_fl.map(cls) == "stop").mean():.3f}')
            log(f'{lab}: n {len(cov)} R {m:+.3f} [{lo:+.3f},{hi:+.3f}] ex-top5 {exTop(cov.R):+.3f} | entry-bar-no-stop '
                f'{m2:+.3f} [{lo2:+.3f},{hi2:+.3f}] | +40%@3R scale {m3:+.3f} [{lo3:+.3f},{hi3:+.3f}]')
    # availability over ALL replay-filled rows (both delays pooled by row) and BT cross-check
    f0 = filled[filled.delay_s == 0]
    log(f'availability (delay 0): filled {len(f0)}, covered {f0.R.notna().sum()} ({f0.R.notna().mean():.2f}); '
        f'by seg: ' + ', '.join(f'{s} {f0[f0.seg == s].R.notna().mean():.2f}' for s in WEEKS if (f0.seg == s).any()))
    c = df[df.delay_s == 0].dropna(subset=['bt_resim_reason'])
    log(f'BT-entry cross-check: n {len(c)}; class agreement with BT population reason '
        f'{(c.bt_reason.map(cls) == c.bt_resim_reason.map(cls)).mean():.3f}; stop-vs-not agreement '
        f'{((c.bt_reason.map(cls) == "stop") == (c.bt_resim_reason.map(cls) == "stop")).mean():.3f}; '
        f'stop share BT {(c.bt_reason.map(cls) == "stop").mean():.3f} vs resim {(c.bt_resim_reason.map(cls) == "stop").mean():.3f}')
    c2 = df[df.delay_s == 0].dropna(subset=['bt_resim_reason_fl'])
    log(f'BT-entry cross-check WITH ATR floor: class agreement '
        f'{(c2.bt_reason.map(cls) == c2.bt_resim_reason_fl.map(cls)).mean():.3f}; stop share resim '
        f'{(c2.bt_resim_reason_fl.map(cls) == "stop").mean():.3f}')
    log(f'parquet ATR available on {filled[filled.R.notna()].atr_used.mean():.2f} of simulated replay fills')


def main():
    """Entry point; tables go to stdout and rebuild_1709_tables.txt."""
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    da = build_a(log)
    table_a(da, log)
    db = build_b(log)
    table_b(db, log)
    (OUT / 'rebuild_1709_tables.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.exit(main())
