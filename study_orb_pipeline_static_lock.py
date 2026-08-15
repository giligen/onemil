"""CORRECTED monthly + Q1 2026 P&L using shipped static_lock_1R exit.

Previously we used study_orb_100k_defended which sources `pnl` from the
orb_features CSV — but that CSV uses fixed +2R target/-1R stop exits, NOT
the shipped static_lock_1R logic. This script re-simulates exits from
bars (matching show_q1_2025_static_lock.py), then runs the defended
pipeline on the corrected per-trade pnl.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df, OUT_DIR, OrbTrade
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
    ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group
from trading.orb_touchgo_filter import (
    evaluate_rule_m, evaluate_rule_d, find_breakout_bar_ts, load_touchgo_config,
)


# Sizing constants. B+ RESTART 2026-08-15: ACCOUNT/N/RISK are now READ FROM
# orb.yaml (sizing.account_budget_usd / max_concurrent / risk_per_trade_usd) so
# the nightly BT book simulates the SAME config live trades — see load_bt_config()
# (review P1-4). The literals below are legacy fallbacks only (used iff orb.yaml
# lacks the keys). Q_CAPS is consulted ONLY by the mults REFIT fallback, which
# never runs when orb.yaml carries adaptive_mults (B+ ships uniform-1.0).
ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
LOCK_TRIGGER_R = 1.75   # 2026-05-08: BT-validated upgrade from 1.5
LOCK_STOP_R = 0.5       # 2026-05-08: BT-validated upgrade from 1.0
EXIT_SLIP_BPS = 10.0

# Default book path — B+ RESTART 2026-08-15. The pipeline WRITES and
# report_common READS this same path (single source of truth). The legacy
# analysis_results/orb_static_lock_trades.csv is preserved untouched for history.
DEFAULT_BOOK_CSV = 'analysis_results/orb_bplus_book.csv'


def load_bt_config(yaml_path: str = 'orb.yaml') -> dict:
    """Read the B+ sizing / threshold / G1 / book-path constants from orb.yaml.

    Parity by construction (review P1-4): the nightly BT ground-truth book must
    simulate the SAME config live trades. Falls back to the module-level legacy
    literals for any missing key (with a printed WARNING) so old research runs
    without a B+ orb.yaml still work. Env overrides win over yaml:
      ORB_BT_ACCOUNT / ORB_BT_N / ORB_BT_RISK / ORB_BT_THRESHOLD / ORB_BT_BOOK_OUT.
    """
    import yaml as _yaml
    cfg = {}
    try:
        cfg = _yaml.safe_load(open(yaml_path)) or {}
    except Exception as e:
        print(f"WARNING: load_bt_config could not read {yaml_path} ({e}) — "
              f"using legacy literals ACCOUNT={ACCOUNT} N={N} RISK={RISK}")
    sizing = (cfg.get('sizing') or {})
    filt = (cfg.get('filter') or {})
    g1 = (filt.get('g1_veto') or {})
    bt = (cfg.get('backtest') or {})
    pm = (sizing.get('pm_dollar_vol_mult') or {})

    def _f(env, val, default):
        v = os.environ.get(env)
        return float(v) if v not in (None, '') else float(
            val if val is not None else default)

    from study_orb_sizing import FILTER_THRESHOLD as _DEF_THR
    from trading.orb_g1_veto import (DEFAULT_PDR_MIN as _G1_PDR,
                                     DEFAULT_RV20_MIN as _G1_RV)
    from trading.orb_pdr_veto import DEFAULT_MIN_PDR_PCT as _DEF_PDR
    pdr_cfg = (filt.get('prev_day_range_veto') or {})
    out = {
        'account': _f('ORB_BT_ACCOUNT', sizing.get('account_budget_usd'), ACCOUNT),
        'n': int(_f('ORB_BT_N', sizing.get('max_concurrent'), N)),
        'risk': _f('ORB_BT_RISK', sizing.get('risk_per_trade_usd'), RISK),
        'threshold': _f('ORB_BT_THRESHOLD', filt.get('threshold'), _DEF_THR),
        # PDR veto threshold also from orb.yaml (B+ = 11.0). Env
        # ORB_PDR_VETO_MIN_PCT still wins (handled at the veto call site).
        'pdr_min': float(pdr_cfg.get('min_prev_day_range_pct', _DEF_PDR)),
        'g1_rv20_min': float(g1.get('return_volatility_20d_min', _G1_RV)),
        'g1_pdr_min': float(g1.get('prev_day_range_pct_min', _G1_PDR)),
        'g1_enabled': bool(g1.get('enabled', True)),
        # PM/news mult stack: B+ turns it OFF (sizing.pm_dollar_vol_mult.enabled
        # = false). The pipeline must honor the yaml flag so the BT book matches
        # live; env ORB_PM_MULT=0 still forces off at the call site.
        'pm_enabled': bool(pm.get('enabled', True)),
        'book_csv': (os.environ.get('ORB_BT_BOOK_OUT')
                     or bt.get('nightly_book_csv') or DEFAULT_BOOK_CSV),
    }
    print(f"BT config (B+ parity): account=${out['account']:,.0f} N={out['n']} "
          f"risk=${out['risk']:,.0f} threshold={out['threshold']:.12f} "
          f"pdr_veto>={out['pdr_min']} g1({out['g1_enabled']}, "
          f"rv20>={out['g1_rv20_min']}, pdr>={out['g1_pdr_min']}) "
          f"book={out['book_csv']}")
    return out

# Touch-and-go filter (Rule M + Rule D). Default-on; env-var overrides via
# ORB_TOUCHGO_* (see trading/orb_touchgo_filter.py::load_touchgo_config).
# BT/LIVE parity by construction: trading/orb_engine.py imports the same helper.
TOUCHGO_CFG = load_touchgo_config({})


def _session_open_timestamp(bars):
    """ET 9:30 timestamp."""
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


FORCE_CLOSE_ET = os.environ.get('ORB_BT_FORCE_CLOSE_ET', '15:45')


def simulate_static_lock(bars, entry_price, range_high, range_low, entry_time):
    """Simulate ORB static_lock exit with Rule M / Rule D touchgo filter.

    Exit precedence (first to fire wins):
        1. Rule M: at close of entry bar (the breakout bar that triggered our
           buy-stop), if bb_close_pos < TOUCHGO_CFG.rule_m_threshold (default
           0.5) -> exit at bb_close * (1 - 10bps slippage), reason='tag_bb'.
        2. Rule D: at close of bar 1 (first post-entry bar), if bar 1 low
           reverted >= TOUCHGO_CFG.rule_d_revert_R (default 0.75R) below entry
           -> exit at (entry + rule_d_exit_R*range_size) * (1 - 10bps),
           reason='tag_b1'.
        3. Static lock loop (existing): scans remaining bars for lock trigger
           (entry + 1.75R) and stop (range_low) with mid-trade lock_stop ratchet.
        4. EOD (no exit hit): close of the last bar at/before FORCE_CLOSE_ET
           (default 15:45 — LIVE PARITY, fixed 2026-07-04: the old last-bar
           (~15:59) exit understated the book by ~$20K/18mo because the final
           14min into the close are systematically adverse for gapper holds;
           a 15:00/15:30/15:45/15:59 sweep peaks AT 15:45 — live's existing
           force-close time is accidentally optimal. ORB_BT_FORCE_CLOSE_ET
           env restores legacy for old-study reproduction).

    Touchgo rules can be globally disabled via env var ORB_TOUCHGO_ENABLED=0
    (re-imports happen at startup; restart needed).
    """
    range_size = range_high - range_low
    # LIVE PARITY (2026-07-04): truncate at the 15:45 ET force-close, like
    # production. See docstring item 4.
    _et = bars['timestamp'].dt.tz_convert('America/New_York').dt.time
    _fc_h, _fc_m = (int(x) for x in FORCE_CLOSE_ET.split(':'))
    from datetime import time as _dtime
    bars = bars[_et <= _dtime(_fc_h, _fc_m)]
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)

    # Rule M: evaluate at close of entry bar (post.iloc[0]).
    if len(post) >= 1:
        entry_bar = post.iloc[0]
        fire_m, exit_m = evaluate_rule_m(
            float(entry_bar['open']), float(entry_bar['high']),
            float(entry_bar['low']), float(entry_bar['close']),
            TOUCHGO_CFG,
        )
        if fire_m and exit_m is not None:
            return exit_m * (1 - EXIT_SLIP_BPS / 10000), 'tag_bb'

    # Rule D: evaluate at close of bar 1 (post.iloc[1]).
    if len(post) >= 2:
        b1 = post.iloc[1]
        fire_d, exit_d = evaluate_rule_d(
            entry_price, float(b1['low']), range_size, TOUCHGO_CFG,
        )
        if fire_d and exit_d is not None:
            return exit_d * (1 - EXIT_SLIP_BPS / 10000), 'tag_b1'

    # Static lock loop (existing).
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - EXIT_SLIP_BPS/10000), 'lock' if armed else 'stop'
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def main():
    bt_cfg = load_bt_config()
    # Features CSV: env override (ORB_BT_FEATURES_CSV) else latest non-corrmatrix
    # glob. B+ ground-truth regen points this at the frozen
    # orb_features_20260814_1741.csv; the nightly uses the latest.
    csv = os.environ.get('ORB_BT_FEATURES_CSV')
    if not csv:
        csv = sorted(p for p in glob.glob('analysis_results/orb_features_*.csv')
                     if 'corrmatrix' not in p)[-1]
    print(f"Features CSV: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    print("Re-simulating with static_lock_1R exit...")
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        # Shared with live (trading/orb_engine.py) — the market breakout bar is
        # the first bar with high > range_high. `search` is already windowed to
        # [range_end, range_end+60min), so no range_end_ts arg needed.
        entry_ts = find_breakout_bar_ts(search, rh)
        if entry_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        entry_p = float(row['entry_price'])
        exit_p, reason = simulate_static_lock(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)

    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    # Risk-parity sizing (B+ 2026-08-15: account/N/risk from orb.yaml via bt_cfg)
    account = bt_cfg['account']
    n_per_day = bt_cfg['n']
    risk = bt_cfg['risk']
    per_pos_cap = account / n_per_day
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    # Research hook (2026-07-17): dump the FULL resimmed candidate set
    # (all candidates, production exit physics, pre-selection) so selector
    # experiments run on identical physics without re-walking bars.
    _dump = os.environ.get('ORB_BT_DUMP_CANDIDATES')
    if _dump:
        df.to_csv(_dump, index=False)
        print(f"Candidates dumped (post-resim, pre-selection): {_dump} "
              f"({len(df)} rows)")

    # Z-params + quintile cutoffs (2026-07-17 PARITY FIX — same class as
    # the 7/10 mult desync, one layer up): LIVE scores with the frozen
    # April-fit constants in orb.yaml (filter.features + quintile_cutoffs).
    # The per-run TRAIN refit here silently DIVERGED once the band-study
    # full-universe rebuilds added TRAIN-era rows to the features CSV
    # (1,836 TRAIN rows vs the original fit's universe) — identical ASPI
    # features scored 0.410 live vs 0.317 BT on 7/14, flipping quintiles
    # and the selected book, and false-flagging live in the green check.
    # Doctrine: the FROZEN fit (what live trades) is canonical; BT must
    # match it. Refit retained as fallback only when yaml lacks the keys.
    params = None
    cutoffs = None
    try:
        import yaml as _yaml
        _cfg0 = _yaml.safe_load(open('orb.yaml'))
        _feats = (_cfg0.get('filter', {}) or {}).get('features') or {}
        _qcuts = _cfg0.get('quintile_cutoffs') or []
        _fnames = [f for f, _sgn in FILTER_FEATURES]
        if (all(f in _feats for f in _fnames) and len(_qcuts) == 4):
            params = {f: {'mean': float(_feats[f]['mean']),
                          'std': float(_feats[f]['std']),
                          'sign': int(_feats[f]['sign'])}
                      for f in _fnames}
            cutoffs = [float(x) for x in _qcuts]
            print("Z-params + quintile cutoffs: orb.yaml literals (LIVE PARITY)")
    except Exception as _e:
        print(f"Z-params: orb.yaml read failed ({_e}) — falling back to refit")
    # HARD FAIL on missing yaml constants (2026-07-17): BOTH parity bugs
    # (mults 7/10, z-params 7/17) survived because the refit fallback ran
    # silently. A book computed on refit params is NOT ground truth for
    # live and must never be produced by accident. BT_ALLOW_REFIT=1 is
    # the explicit research escape hatch.
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    _allow_refit = os.environ.get('BT_ALLOW_REFIT', '').strip().lower() in (
        '1', 'true', 'yes')
    if params is None:
        if not _allow_refit:
            raise SystemExit(
                "FATAL: orb.yaml lacks filter.features z-params — refusing "
                "to silently refit (that is how the 7/10 and 7/17 parity "
                "bugs happened). Fix orb.yaml or set BT_ALLOW_REFIT=1 for "
                "explicit research use.")
        params = fit_z_params(train, FILTER_FEATURES)
        print("Z-params: TRAIN REFIT (BT_ALLOW_REFIT) — NOT live parity")
    df['_composite'] = composite_score(df, params)
    # B+ 2026-08-15: threshold from orb.yaml (filter.threshold) via bt_cfg, NOT
    # the imported study_orb_sizing.FILTER_THRESHOLD constant (review P1-4).
    threshold = bt_cfg['threshold']
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= threshold].copy()
    if cutoffs is None:
        if not _allow_refit:
            raise SystemExit(
                "FATAL: orb.yaml lacks quintile_cutoffs — refusing to "
                "silently refit. Fix orb.yaml or set BT_ALLOW_REFIT=1.")
        cutoffs = fit_quintile_cutoffs(train_k['_composite'])
        print("Quintile cutoffs: TRAIN REFIT (BT_ALLOW_REFIT) — NOT live parity")
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    # 2026-07-10 PARITY FIX: use orb.yaml's frozen adaptive_mults literals —
    # the same values LIVE trades — instead of refitting per run. The
    # per-run refit silently DIVERGED from live when the 15:45 parity fix
    # changed the features CSV's pnl (BT ran Q2=3.0/Q4=0.25 while live ran
    # Q4=1.842/Q2=0.25 — nobody noticed until the selection re-audit).
    # Refit retained as fallback only when the yaml is absent.
    mults = None
    try:
        import yaml as _yaml
        _cfg = _yaml.safe_load(open('orb.yaml'))
        _am = _cfg.get('adaptive_mults') or {}
        if all(q in _am for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')):
            mults = {q: float(_am[q]) for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')}
            print(f"Mults (orb.yaml literals — LIVE PARITY): {mults}")
    except Exception as _e:
        print(f"Mults: orb.yaml read failed ({_e})")
    if mults is None and os.environ.get('BT_ALLOW_REFIT', '').strip().lower() \
            not in ('1', 'true', 'yes'):
        raise SystemExit(
            "FATAL: orb.yaml lacks adaptive_mults — refusing to silently "
            "refit (7/10 parity-bug class). Fix orb.yaml or set "
            "BT_ALLOW_REFIT=1 for explicit research use.")
    if mults is None:
        mults = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            sub = train_k[train_k['_quintile'] == q]
            mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(sub['_rp_pnl'].mean()) / avg))
        print(f"Mults (REFIT fallback — not live parity): {mults}")

    # Apply to full timeline
    kept = df[df['_composite'] >= threshold].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)

    # Q1 filter (ships on by default; matches trading/orb_engine.py skip_q1).
    # BT-validated lift: +$8,556 OOS across VAL + HOQ1+. Controlled via env var
    # ORB_SKIP_Q1=0/false/no/off to disable for research/diff runs.
    # NOTE: mults above are fit on TRAIN data INCLUDING Q1 trades. Q1 still
    # influences calibration of Q4/Q5 mults via train_k.mean(). Filter only
    # affects which trades are SELECTED, not how mults are FIT. Acceptable
    # because Q1's contribution to avg is small (mult capped at 0.5x).
    _q1_env = os.environ.get('ORB_SKIP_Q1', '1').strip().lower()
    skip_q1 = _q1_env not in ('0', 'false', 'no', 'off', '')
    if skip_q1:
        n_q1 = int((kept['_quintile'] == 'Q1').sum())
        kept = kept[kept['_quintile'] != 'Q1'].copy()
        print(f"Q1 filter: dropped {n_q1} Q1 candidates "
              f"(set ORB_SKIP_Q1=0 to disable)")

    # Top-K + dedup per day
    sel_rows = []
    for day, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        kept_today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            kept_today.append(r)
            if len(kept_today) >= n_per_day: break
        sel_rows.extend(kept_today)
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['month'] = sel['date'].dt.to_period('M').astype(str)

    # PM dollar-volume sizing mult (ships 2026-07-04; matches
    # trading/orb_engine.py via shared trading/orb_pm_mult.py). Upsize-only
    # x1.5 above the TRAIN-frozen cut. PM data: data/research CSV for the
    # historical book + nightly appends; unknown symbol-days fail-open at
    # 1.0 (logged count). Env: ORB_PM_MULT=0 disables.
    from trading.orb_pm_mult import (
        DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT, DEFAULT_HIGH_MULT_NEWS,
        LEGACY_HIGH_MULT, pm_size_multiplier,
    )
    _pm_env = os.environ.get('ORB_PM_MULT', '1').strip().lower()
    # B+ 2026-08-15: honor orb.yaml sizing.pm_dollar_vol_mult.enabled (bt_cfg).
    # B+ ships PM OFF; env ORB_PM_MULT=0 also forces off. When disabled the
    # whole PM/news mult stack is skipped and every mult stays 1.0 (live parity).
    if not bt_cfg['pm_enabled']:
        print("PM sizing mult: DISABLED via orb.yaml "
              "(sizing.pm_dollar_vol_mult.enabled=false) — all mults 1.0")
    if bt_cfg['pm_enabled'] and _pm_env not in ('0', 'false', 'no', 'off', ''):
        import glob as _glob
        _pm_paths = sorted(_glob.glob('data/research/orb_premarket_dollar_vol_*.csv'))
        # News gate on the PM mult (ships 2026-07-10; matches
        # trading/orb_engine.py via the SAME shared helper). has_news comes
        # from the backfill CSVs (research/scripts/orb_news_backfill.py);
        # unknown symbol-days -> None (fail-open, no news boost — same as
        # live's failed-fetch path). ORB_PM_NEWS_GATE=0 restores the legacy
        # ungated x1.5 for old relative comparisons.
        _ng_env = os.environ.get('ORB_PM_NEWS_GATE', '1').strip().lower()
        news_gate_on = _ng_env not in ('0', 'false', 'no', 'off', '')
        _news_map = {}
        if news_gate_on:
            _news_paths = sorted(_glob.glob('data/research/orb_news_catalyst_*.csv'))
            if _news_paths:
                _nw = pd.concat([pd.read_csv(x) for x in _news_paths],
                                ignore_index=True) \
                    .drop_duplicates(subset=['symbol', 'day'], keep='last')
                # Deliberate class rule (2026-07-11, matches live): the
                # news boost requires positive identification as a common
                # stock — wrappers/unknown are structurally ineligible
                # (trading/orb_asset_class.py; the crowding-inversion
                # evidence is in research/orb_news_catalyst_jul2026.md).
                from trading.orb_asset_class import (
                    STOCK as _CLS_STOCK, effective_has_news as _eff_news,
                    load_class_map as _load_cmap)
                _cmap = _load_cmap()
                _news_map = {
                    (r['symbol'], r['day']): _eff_news(
                        (r['n_articles'] or 0) > 0,
                        _cmap.get(r['symbol'], 'unknown'))
                    for _, r in _nw.iterrows()}
                n_class_blocked = sum(
                    1 for k, v in _news_map.items()
                    if v is False and _cmap.get(k[0], 'unknown') != _CLS_STOCK)
                print(f"PM news gate: class rule active — "
                      f"{n_class_blocked} newsy non-stock symbol-days "
                      f"structurally ineligible")
            else:
                print("PM news gate: WARNING no data/research/orb_news_catalyst_*.csv "
                      "— has_news unknown everywhere (fail-open, no news boosts)")
        if _pm_paths:
            _pm = pd.concat([pd.read_csv(x) for x in _pm_paths], ignore_index=True)
            _pm = _pm.dropna(subset=['pm_dollar_vol'])                 .drop_duplicates(subset=['symbol', 'day'], keep='last')
            _pm_map = {(r['symbol'], r['day']): r['pm_dollar_vol']
                       for _, r in _pm.iterrows()}
            sel['_pm_key'] = list(zip(sel['symbol'],
                                      sel['date'].dt.strftime('%Y-%m-%d')))
            if news_gate_on:
                sel['_pm_mult'] = sel['_pm_key'].map(
                    lambda k: pm_size_multiplier(
                        _pm_map.get(k), DEFAULT_HIGH_CUT_USD,
                        DEFAULT_HIGH_MULT,
                        has_news=_news_map.get(k),
                        high_mult_news=DEFAULT_HIGH_MULT_NEWS,
                        news_gate=True))
            else:
                sel['_pm_mult'] = sel['_pm_key'].map(
                    lambda k: pm_size_multiplier(
                        _pm_map.get(k), DEFAULT_HIGH_CUT_USD,
                        LEGACY_HIGH_MULT, news_gate=False))
            n_boost = int((sel['_pm_mult'] > 1.0).sum())
            n_unknown = int(sum(1 for k in sel['_pm_key'] if k not in _pm_map))
            n_above_cut = int(sum(
                1 for k in sel['_pm_key']
                if (_pm_map.get(k) or 0) > DEFAULT_HIGH_CUT_USD))
            sel['_sized_pnl'] = sel['_sized_pnl'] * sel['_pm_mult']
            sel = sel.drop(columns=['_pm_key'])
            if news_gate_on:
                print(f"PM sizing mult (news-gated): {n_boost} picks boosted "
                      f"x{DEFAULT_HIGH_MULT_NEWS} (news+PM$), "
                      f"{n_above_cut - n_boost} above-cut without news at "
                      f"x{DEFAULT_HIGH_MULT}, {n_unknown} PM-unknown->1.0 "
                      f"(ORB_PM_NEWS_GATE=0 for legacy)")
            else:
                print(f"PM sizing mult (LEGACY ungated): {n_boost} picks boosted "
                      f"x{LEGACY_HIGH_MULT}, {n_unknown} unknown->1.0")
        else:
            print("PM sizing mult: WARNING no data/research/orb_premarket_dollar_vol_*.csv "
                  "— all mults 1.0 (fail-open)")

    # PDR veto (ships 2026-07-04; matches trading/orb_engine.py). Applied
    # POST-selection with NO refill — a vetoed pick's slot stays empty,
    # exactly like live (backfill form tested toxic; see
    # trading/orb_pdr_veto.py docstring for evidence + thresholds).
    # Env: ORB_PDR_VETO=0 disables; ORB_PDR_VETO_MIN_PCT overrides threshold.
    from trading.orb_pdr_veto import pdr_veto_applies
    _pdr_env = os.environ.get('ORB_PDR_VETO', '1').strip().lower()
    pdr_veto_on = _pdr_env not in ('0', 'false', 'no', 'off', '')
    if pdr_veto_on and 'prev_day_range_pct' in sel.columns:
        # B+ 2026-08-15: threshold from orb.yaml (bt_cfg, = 11.0) so the BT
        # book matches live; env ORB_PDR_VETO_MIN_PCT still wins.
        pdr_min = float(os.environ.get('ORB_PDR_VETO_MIN_PCT',
                                       str(bt_cfg['pdr_min'])))
        veto_mask = sel['prev_day_range_pct'].apply(
            lambda v: pdr_veto_applies(None if pd.isna(v) else float(v),
                                       pdr_min))
        n_veto = int(veto_mask.sum())
        veto_pnl = float(sel.loc[veto_mask, '_sized_pnl'].sum())
        sel = sel[~veto_mask].copy()
        print(f"PDR veto: dropped {n_veto} pick(s) with prev_day_range_pct "
              f"<= {pdr_min} (their P&L would have been {veto_pnl:+,.0f}; "
              f"set ORB_PDR_VETO=0 to disable)")
    elif pdr_veto_on:
        print("PDR veto: WARNING prev_day_range_pct column missing from "
              "features CSV — veto skipped (fail-open)")

    # G1 volatility-fingerprint veto (B+ RESTART 2026-08-15; matches
    # trading/orb_engine.py via the SAME shared trading/orb_g1_veto.py — parity
    # by construction). Applied POST-selection with NO refill, exactly like PDR.
    # KEEP iff BOTH return_volatility_20d and prev_day_range_pct clear their
    # frozen minimums; fail-open on rv20 NaN/0.0 or pdr NaN. Env: ORB_G1_VETO=0.
    from trading.orb_g1_veto import g1_reject as _g1_reject
    _g1_env = os.environ.get('ORB_G1_VETO', '1').strip().lower()
    g1_veto_on = (bt_cfg['g1_enabled']
                  and _g1_env not in ('0', 'false', 'no', 'off', ''))
    if g1_veto_on and {'return_volatility_20d',
                       'prev_day_range_pct'} <= set(sel.columns):
        g1_mask = pd.Series(
            [_g1_reject(rv, pdr, bt_cfg['g1_rv20_min'], bt_cfg['g1_pdr_min'])
             is not None
             for rv, pdr in zip(sel['return_volatility_20d'],
                                sel['prev_day_range_pct'])],
            index=sel.index)
        n_g1 = int(g1_mask.sum())
        g1_pnl = float(sel.loc[g1_mask, '_sized_pnl'].sum())
        sel = sel[~g1_mask].copy()
        print(f"G1 veto: dropped {n_g1} pick(s) failing the volatility "
              f"fingerprint (rv20>={bt_cfg['g1_rv20_min']} AND "
              f"pdr>={bt_cfg['g1_pdr_min']}); their P&L would have been "
              f"{g1_pnl:+,.0f} (set ORB_G1_VETO=0 to disable)")
    elif g1_veto_on:
        print("G1 veto: WARNING return_volatility_20d/prev_day_range_pct "
              "column missing from features CSV — veto skipped (fail-open)")

    # Catalyst-required veto (ships 2026-07-18; matches orb_engine via
    # trading/orb_catalyst_veto.py). Newsless-and-alone selected picks
    # dropped, slot stays empty (post-selection, like PDR — no refill).
    # News source: orb_news_catalyst_*.csv; anchors from the class map
    # names via the SAME underlying_anchor helper live uses.
    # Env: ORB_CATALYST_VETO=0 disables.
    _cv_env = os.environ.get('ORB_CATALYST_VETO', '1').strip().lower()
    if _cv_env not in ('0', 'false', 'no', 'off', ''):
        from trading.orb_asset_class import (DEFAULT_CLASS_MAP,
                                             underlying_anchor)
        from trading.orb_catalyst_veto import (
            DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies)
        import csv as _csv
        # Self-sufficient glob import: the PM-mult block (which used to define
        # `_glob`) is now SKIPPED when orb.yaml disables PM sizing (B+), so this
        # section must import it itself instead of leaning on that side effect.
        import glob as _glob
        _names = {}
        try:
            with open(DEFAULT_CLASS_MAP, newline='') as _fh:
                for _row in _csv.DictReader(_fh):
                    _names[_row['symbol']] = _row.get('name', '')
        except Exception as _e:
            print(f"Catalyst veto: class-map names unavailable ({_e})")
        _cmap2 = {s: c for s, c in _cmap.items()} if '_cmap' in dir() else None
        if _cmap2 is None:
            from trading.orb_asset_class import load_class_map as _lcm
            _cmap2 = _lcm()
        _anchors = {s: underlying_anchor(s, _names.get(s), _cmap2)
                    for s in set(df['symbol'])}
        sel['_anchor'] = sel['symbol'].map(_anchors)
        # cohort from the FULL candidate universe that morning (df),
        # matching live (engine counts over its candidate dict)
        _day_anchor = df.assign(_a=df['symbol'].map(_anchors))
        _cohorts = {day: anchor_cohort_counts(g['_a'])
                    for day, g in _day_anchor.groupby(
                        _day_anchor['date'].dt.strftime('%Y-%m-%d'))}
        # RAW own-ticker news (tri-state: absent pair -> None -> fail-open),
        # NOT the class-gated effective map used by the sizing gate.
        _raw_news = {}
        for _p in sorted(_glob.glob('data/research/orb_news_catalyst_*.csv')):
            for _, _r in pd.read_csv(_p).iterrows():
                _raw_news[(_r['symbol'], _r['day'])] = (_r['n_articles'] or 0) > 0
        _sel_day = sel['date'].dt.strftime('%Y-%m-%d')
        _has_news_sel = [_raw_news.get((s, d))
                         for s, d in zip(sel['symbol'], _sel_day)]
        cv_mask = [catalyst_veto_applies(hn, a, _cohorts.get(d, {}),
                                         DEFAULT_MIN_COHORT)
                   for hn, a, d in zip(_has_news_sel, sel['_anchor'], _sel_day)]
        cv_mask = pd.Series(cv_mask, index=sel.index)
        n_cv = int(cv_mask.sum())
        cv_pnl = float(sel.loc[cv_mask, '_sized_pnl'].sum())
        sel = sel[~cv_mask].copy()
        print(f"Catalyst veto: dropped {n_cv} newsless-and-alone pick(s) "
              f"(their P&L would have been {cv_pnl:+,.0f}; "
              f"ORB_CATALYST_VETO=0 to disable)")

    # 2026-05-08: fill-rate haircut. Pre-fix, BT assumed every qualified
    # signal filled — no model of buy-stop misses. LIVE Mon-Thu 5/4-5/7
    # observed 9 fills out of 16 buy-stops (56%). The cross_time_min
    # within-10min window in BT does not equal "filled in live": real
    # fills require sustained price + sufficient depth at limit_price.
    # Set ORB_BT_FILL_RATE=0.0 → no haircut (legacy). 0.56 → empirical.
    # 1.0 → full kill (debug). Default 0 = legacy behaviour preserved.
    fill_rate = float(os.environ.get('ORB_BT_FILL_RATE', '0') or 0)
    if 0 < fill_rate < 1:
        import numpy as _np
        rng = _np.random.RandomState(42)  # deterministic
        n_before = len(sel)
        keep_mask = rng.random(n_before) < fill_rate
        sel_pre_haircut_pnl = float(sel['_sized_pnl'].sum())
        sel = sel[keep_mask].copy()
        n_after = len(sel)
        sel_post_haircut_pnl = float(sel['_sized_pnl'].sum())
        print(f"Fill-rate haircut: ORB_BT_FILL_RATE={fill_rate:.2f} "
              f"→ kept {n_after}/{n_before} trades "
              f"(${sel_pre_haircut_pnl:+,.0f} → ${sel_post_haircut_pnl:+,.0f})")

    # Save trade-level CSV for further analysis. B+ 2026-08-15: write to the
    # config-driven book path (bt_cfg['book_csv'] = orb_bplus_book.csv) that
    # report_common reads for the nightly parity gates. The legacy
    # orb_static_lock_trades.csv is left UNTOUCHED for history.
    book_csv = bt_cfg['book_csv']
    os.makedirs(os.path.dirname(book_csv) or '.', exist_ok=True)
    sel.to_csv(book_csv, index=False)

    # Per-day → per-month
    daily = sel.groupby('date').agg(
        pnl=('_sized_pnl', 'sum'),
        picks=('_sized_pnl', 'count'),
    ).reset_index()
    daily['month'] = daily['date'].dt.to_period('M')

    rows = []
    cum = 0.0
    for m, dg in daily.groupby('month'):
        wins = dg[dg['pnl'] > 0]; losses = dg[dg['pnl'] < 0]
        dgs = dg.sort_values('date').reset_index(drop=True)
        dgs['cum'] = dgs['pnl'].cumsum()
        peak = -1e18; mdd = 0.0
        for c in dgs['cum']:
            peak = max(peak, c); mdd = min(mdd, c - peak)
        m_pnl = float(dg['pnl'].sum()); cum += m_pnl
        rows.append({
            'month': str(m),
            'days': len(dg),
            'green': len(wins),
            'red': len(losses),
            'day_wr_pct': len(wins)/len(dg)*100,
            'picks': int(dg['picks'].sum()),
            'best_day': float(dg['pnl'].max()),
            'worst_day': float(dg['pnl'].min()),
            'pnl': m_pnl,
            'cum_pnl': cum,
            'intra_dd': mdd,
        })

    out = pd.DataFrame(rows)
    print("\nORB defended pipeline + STATIC_LOCK_1R — monthly P&L (Jan'25-Apr'26)\n")
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:,.0f}'.format)
    print(out.to_string(index=False))

    print(f"\n{'='*70}")
    print("Negative months")
    print(f"{'='*70}")
    neg = out[out['pnl'] < 0]
    if len(neg) == 0:
        print("  (none)")
    else:
        for _, r in neg.iterrows():
            print(f"  {r['month']}  ${r['pnl']:+,.0f}  "
                  f"({r['days']} days, {r['green']}G/{r['red']}R, worst day ${r['worst_day']:+,.0f})")

    print(f"\n{'='*70}")
    print("Summary (STATIC LOCK)")
    print(f"{'='*70}")
    print(f"Months tracked:   {len(out)}")
    print(f"Negative months:  {(out['pnl']<0).sum()} ({(out['pnl']<0).mean()*100:.1f}%)")
    print(f"Best month:       {out.loc[out['pnl'].idxmax(),'month']}  ${out['pnl'].max():+,.0f}")
    print(f"Worst month:      {out.loc[out['pnl'].idxmin(),'month']}  ${out['pnl'].min():+,.0f}")
    print(f"Median month:     ${out['pnl'].median():+,.0f}")
    print(f"Mean month:       ${out['pnl'].mean():+,.0f}")
    print(f"Cum P&L:          ${out['pnl'].sum():+,.0f}")

    out.to_csv('analysis_results/orb_monthly_static_lock.csv', index=False)
    print(f"\nSaved: analysis_results/orb_monthly_static_lock.csv + {book_csv}")


if __name__ == '__main__':
    main()
