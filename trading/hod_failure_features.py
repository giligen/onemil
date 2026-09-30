"""Pure feature functions for the FF-k (fast-failure) short-overlay model (cells 1,669 / 1,673 / 1,675).

Bar arrays follow the project convention: a dict of numpy arrays 'o','h','l','c','v','minarr'
(ET minute-of-day per bar, RTH bars only, ascending, index 0 = first RTH bar of the day).

No I/O, no DB, no network -- numpy/pandas bar arrays plus the arm-time inputs only. The research
scorer (research/hod_entry/1675_forward.py) and the live engine MUST both call `k1_features` here
for the FF10-at-k=1 short-overlay rule -- parity by construction (CLAUDE.md 'ONE spec' rule).

Ported verbatim (same arithmetic, same column names) from:
  research/hod_entry/1668_failure.py: find_fill_index, find_break_bar, walk_k, continuous_features,
    shape_features, talib_features, CDL_NAMES
  research/hod_entry/1669_fast_failure.py: bar0_features, talib_bar0, the ARM/BAR0/BAR1 column lists
PREREG_1675.md, cell 1,675.
"""
import numpy as np
import pandas as pd

try:
    import talib
    CDL_NAMES = talib.get_function_groups()['Pattern Recognition']
    TALIB_AVAILABLE = True
except ImportError:                                    # pragma: no cover - environment-dependent
    talib = None
    CDL_NAMES = []
    TALIB_AVAILABLE = False

EOD_M = 955          # 15:55 ET force-flat (matches sip_rebuild.EOD_M / trading.hod_break)
LEVEL_TOL = 0.01      # $ tolerance matching a bar's high to the break level

# Canonical ordered column groups (feat_cols_for_k(per_fill, k=1) in 1669_fast_failure.py).
# ARM_FEATS come from the arm-time context (the caller), not from bar arrays.
ARM_FEATS = ['r_pct', 'atr14_pct', 'F11', 'F12', 'F13', 'F14', 'F15', 'minutes_since_open']
BAR0_FEATS = ['clv0', 'body0', 'rangeATR0', 'volratio0', 'closeR0']
BAR1_CONT = ['cS1_dist_level_1', 'cS2_mfe_1', 'cS3_ret_1', 'cS4_volratio_1', 'cS5_spyret_1',
             'cS6_dist_vwap_1', 'cS7_mae_1', 'cA1_progvol_1']
BAR1_SHAPE = ['clv_last_1', 'wick_last_1', 'body_last_1', 'clv_mean_1', 'red_share_1']


def find_fill_index(bars, fill_min):
    """Index of the LAST bar at or before the (possibly fractional) fill-instant minute.

    Never nearest-neighbor: a nearest match can jump to the bar AFTER the fill when the fill is
    >0.5 min into its own bar, mislabeling a post-fill bar as the fill bar (the fractional-minute
    look-ahead pitfall, memory note 9/26). Returns None if no bar qualifies."""
    idx = np.where(bars['minarr'] <= fill_min)[0]
    return int(idx[-1]) if len(idx) else None


def find_break_bar(bars, fill_min, level, tol=LEVEL_TOL):
    """First bar at/before fill_min whose high is within `tol` of `level` (the breakout bar)."""
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return None
    for i in np.where(bars['minarr'] <= fill_min)[0]:
        if abs(bars['h'][i] - level) <= tol:
            return int(i)
    return None


def walk_k(bars, i0, k, stop, target):
    """Bar-by-bar precedence walk over bars i0+1..i0+k (stop wins a same-bar tie with target).

    None if bar i0+k does not exist (not computable). Else a dict with `preempt`
    ('' / 'stop' / 'target' / 'eod') and the k-bar aggregates needed by continuous_features /
    shape_features / the live short-rule (close_k, high_max, low_min, mean_v, sum_v, next_open)."""
    n = len(bars['o'])
    last_idx = i0 + k
    if last_idx >= n:
        return None
    preempt = ''
    for j in range(i0 + 1, i0 + k + 1):
        if bars['minarr'][j] >= EOD_M:
            preempt = 'eod'
            break
        if bars['l'][j] <= stop:
            preempt = 'stop'
            break
        if bars['h'][j] >= target:
            preempt = 'target'
            break
    seg = slice(i0 + 1, i0 + k + 1)
    return dict(
        preempt=preempt, last_idx=last_idx, close_k=bars['c'][last_idx], open_k=bars['o'][last_idx],
        high_max=bars['h'][seg].max(), low_min=bars['l'][seg].min(), mean_v=bars['v'][seg].mean(),
        sum_v=bars['v'][seg].sum(), next_open=(bars['o'][last_idx + 1] if (last_idx + 1) < n else None))


def bar0_features(bars, i0, entry, R, atr14, break_bar_v):
    """k=0 decision-instant features: the fill bar's OWN CLV/body/range-ATR/volume-ratio/
    close-vs-fill, computed only from bars[0..i0] -- no bar after the fill bar is touched."""
    o, h, l, c, v = bars['o'][i0], bars['h'][i0], bars['l'][i0], bars['c'][i0], bars['v'][i0]
    rng = h - l
    return dict(
        close0=c,
        clv0=(c - l) / rng if rng > 0 else np.nan,
        body0=abs(c - o) / rng if rng > 0 else np.nan,
        rangeATR0=(rng / atr14) if (pd.notna(atr14) and atr14 > 0) else np.nan,
        volratio0=(v / break_bar_v) if (pd.notna(break_bar_v) and break_bar_v > 0) else np.nan,
        closeR0=(c - entry) / R,
    )


def continuous_features(walk, level, entry, stop, break_bar_v, spy_ret, vwap_k):
    """Continuous S1-S7 + progress-per-volume at horizon k (walk = walk_k(...) output, must have
    preempt == '')."""
    R = entry - stop
    close_k, high_max, low_min, mean_v, sum_v = (
        walk['close_k'], walk['high_max'], walk['low_min'], walk['mean_v'], walk['sum_v'])
    have_bb = pd.notna(break_bar_v) and break_bar_v > 0
    f = {}
    f['cS1_dist_level'] = (close_k - level) / R if pd.notna(level) else np.nan
    f['cS2_mfe'] = (high_max - entry) / R
    f['cS3_ret'] = (close_k - entry) / R
    f['cS4_volratio'] = (mean_v / break_bar_v) if have_bb else np.nan
    f['cS5_spyret'] = spy_ret if spy_ret is not None else np.nan
    f['cS6_dist_vwap'] = (close_k - vwap_k) / R if vwap_k is not None else np.nan
    f['cS7_mae'] = (entry - low_min) / R
    volratio_sum = (sum_v / break_bar_v) if have_bb else np.nan
    f['cA1_progvol'] = (f['cS3_ret'] / volratio_sum) if pd.notna(volratio_sum) and volratio_sum != 0 else np.nan
    return f


def shape_features(bars, i0, k):
    """CLV / wick / body (last bar only) + mean CLV / red-share (all k bars), over bars
    fill+1..fill+k -- never the fill bar itself."""
    last = i0 + k
    hi_l, lo_l, cl_l, op_l = bars['h'][last], bars['l'][last], bars['c'][last], bars['o'][last]
    rng_l = hi_l - lo_l
    f = {
        'clv_last': (cl_l - lo_l) / rng_l if rng_l > 0 else np.nan,
        'wick_last': (min(op_l, cl_l) - lo_l) / rng_l if rng_l > 0 else np.nan,
        'body_last': abs(cl_l - op_l) / rng_l if rng_l > 0 else np.nan,
    }
    clvs, reds = [], []
    for j in range(i0 + 1, last + 1):
        rng = bars['h'][j] - bars['l'][j]
        if rng > 0:
            clvs.append((bars['c'][j] - bars['l'][j]) / rng)
        reds.append(bars['c'][j] < bars['o'][j])
    f['clv_mean'] = float(np.mean(clvs)) if clvs else np.nan
    f['red_share'] = float(np.mean(reds)) if reds else np.nan
    return f


def talib_features(bars, i0, k):
    """61 CDL* pattern flags at the last bar of the window (i0+k) + bull/bear fire counts over
    bars i0+1..i0+k. {} if talib is not installed (models trained with it degrade gracefully to
    NaN for these columns; HistGradientBoostingClassifier handles NaN natively)."""
    if not TALIB_AVAILABLE:
        return {}
    last = i0 + k
    o, h, l, c = bars['o'][:last + 1], bars['h'][:last + 1], bars['l'][:last + 1], bars['c'][:last + 1]
    n_bull = n_bear = 0
    last_flags = {}
    for name in CDL_NAMES:
        vals = getattr(talib, name)(o, h, l, c)
        window = vals[i0 + 1:last + 1]
        n_bull += int((window > 0).sum())
        n_bear += int((window < 0).sum())
        last_flags[f'cdl_{name}'] = int(vals[last])
    out = {'talib_nbull': n_bull, 'talib_nbear': n_bear}
    out.update(last_flags)
    return out


def vwap_through(bars, idx):
    """Cumulative typical-price VWAP through bar `idx` inclusive (bars 0..idx). None if no volume."""
    tp = (bars['h'][:idx + 1] + bars['l'][:idx + 1] + bars['c'][:idx + 1]) / 3.0
    v_cum = float(bars['v'][:idx + 1].sum())
    return float((tp * bars['v'][:idx + 1]).sum() / v_cum) if v_cum > 0 else None


def spy_close_at_or_before(spy_bars, target_min):
    if spy_bars is None:
        return None
    idx = np.where(spy_bars['minarr'] <= target_min)[0]
    return spy_bars['c'][idx[-1]] if len(idx) else None


def k1_features(bars, i0, entry, stop, target, level, atr14, break_bar_v, arm_ctx, spy_bars=None, fill_min=None):
    """The full FF10-at-k=1 feature row: ARM_FEATS (from `arm_ctx`, passed straight through) +
    BAR0_FEATS + talib bar0 + BAR1_CONT + BAR1_SHAPE + talib k=1.

    Returns (feats: dict, computable: bool). computable is False when bar i0+1 does not exist, or
    the trade already preempted (stop/target/eod) AT fill+1 -- cutting/shorting at that instant is
    moot, matching 1669's k1_computable gate (walk_k(...)['preempt'] == '').
    """
    R = entry - stop
    feats = {k: arm_ctx.get(k, np.nan) for k in ARM_FEATS}
    feats.update(bar0_features(bars, i0, entry, R, atr14, break_bar_v))
    feats.update({f'{k}_0': v for k, v in talib_features(bars, i0, 0).items()})

    w1 = walk_k(bars, i0, 1, stop, target)
    if w1 is None or w1['preempt'] != '':
        return feats, False

    spy_ret = None
    if spy_bars is not None and fill_min is not None:
        spy_at_fill = spy_close_at_or_before(spy_bars, fill_min)
        if spy_at_fill:
            spy_at_k = spy_close_at_or_before(spy_bars, bars['minarr'][w1['last_idx']])
            if spy_at_k is not None:
                spy_ret = spy_at_k / spy_at_fill - 1.0
    vwap_k = vwap_through(bars, w1['last_idx'])
    cf = continuous_features(w1, level, entry, stop, break_bar_v, spy_ret, vwap_k)
    sf = shape_features(bars, i0, 1)
    feats.update({f'{k}_1': v for k, v in cf.items()})
    feats.update({f'{k}_1': v for k, v in sf.items()})
    feats.update({f'{k}_1': v for k, v in talib_features(bars, i0, 1).items()})
    feats['close_1'] = w1['close_k']
    feats['next_open_1'] = w1['next_open']
    feats['_walk1'] = w1
    return feats, True


def ordered_feature_list(available_columns):
    """The k=1 model's ordered feature list, restricted to columns actually present (matches
    feat_cols_for_k(per_fill, k=1) in 1669_fast_failure.py: ARM + BAR0 + talib_0/cdl_0 (if present)
    + BAR1_CONT + BAR1_SHAPE + talib_1/cdl_1 (if present))."""
    cols = list(ARM_FEATS) + list(BAR0_FEATS)
    cols += [c for c in available_columns if (c.startswith('talib_') or c.startswith('cdl_')) and c.endswith('_0')]
    cols += list(BAR1_CONT) + list(BAR1_SHAPE)
    cols += [c for c in available_columns if (c.startswith('talib_') or c.startswith('cdl_')) and c.endswith('_1')]
    return [c for c in cols if c in available_columns]
