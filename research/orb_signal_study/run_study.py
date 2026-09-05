"""ORB signal study — orchestrator (pre-registered protocol: DESIGN.md).

Runs the whole plan unattended and resumably, appending to REPORT.md as each
result lands and rewriting STATUS.md for the hourly pinger:

  P2 baseline  : production pipeline on features_base.csv (flags off) with a
                 candidate dump -> must equal the production book (identity gate)
  C5 read      : range/ATR14 tier read (L1 + L2 by tier), veto test only if a
                 tier is negative in both OOS eras
  C1 grid      : rvol veto t ∈ {0.5,1,1.5,2} fit on TRAIN (L2), read on OOS;
                 rvol rank form; L1 quintile read
  C2           : green|upper × pre|post
  C3, C4       : exit-mechanics variants (full bar walk, ~35 min each)
  P3 pairs     : survivors per §1c
  P4 report    : summary + proposals

Every variant = ONE env-flag set on top of the production stack. Verdicts are
computed by the pre-committed rules in §1a/§1c — this file does not decide,
it applies the rules.

Usage: python3 research/orb_signal_study/run_study.py [--phase baseline|singles|pairs|report|all]
"""
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402
from trading.orb_experimental_rules import ratr_tier  # noqa: E402

D = f'{ROOT}/research/orb_signal_study'
BASE = f'{D}/features_base.csv'
SIDECARS = f'{D}/sidecar_rvol.csv,{D}/sidecar_ratr.csv'
BOOKS = f'{D}/books'
DUMPS = f'{D}/dumps'
STATE = f'{D}/state.json'
REPORT = f'{D}/REPORT.md'
STATUS = f'{D}/STATUS.md'
PIPELINE = f'{ROOT}/study_orb_pipeline_static_lock.py'
TRAIN_END, OOS1_END = '2025-06-30', '2025-12-31'
VMEM = 6_500_000


def now() -> str:
    return datetime.now(timezone.utc).strftime('%H:%M')


def status(line: str) -> None:
    """Rewrite STATUS.md (the pinger posts the last STATUS: line hourly)."""
    with open(STATUS, 'w') as fh:
        fh.write("# ORB signal study — live status\n\nSTATUS: " + f"{now()} UTC — {line}\n")
    print(f"[{now()}] {line}", flush=True)


def report(md: str) -> None:
    with open(REPORT, 'a') as fh:
        fh.write(md + "\n")


def load_state() -> dict:
    return json.load(open(STATE)) if os.path.exists(STATE) else {'done': {}, 'verdicts': {}}


def save_state(st: dict) -> None:
    json.dump(st, open(STATE, 'w'), indent=1, default=str)


def in_session_pause() -> None:
    """No heavy passes 13:00–20:15 UTC Mon–Fri (node-freeze rule)."""
    while True:
        t = datetime.now(timezone.utc)
        if t.weekday() < 5 and 13 * 60 <= t.hour * 60 + t.minute <= 20 * 60 + 15:
            status("paused for the live session (13:00–20:15 UTC); resumes after close")
            time.sleep(600)
        else:
            return


# ---------------------------------------------------------------------------
# pipeline runner
# ---------------------------------------------------------------------------

def run_variant(name: str, env: dict, use_cache: bool = True) -> str:
    """One pipeline pass; returns the book path. Cached selector-only variants
    replay the baseline candidate dump (seconds); exit variants walk bars."""
    os.makedirs(BOOKS, exist_ok=True); os.makedirs(DUMPS, exist_ok=True)
    book = f'{BOOKS}/book_{name}.csv'
    if os.path.exists(book):
        print(f"  [{name}] cached book"); return book
    in_session_pause()
    e = dict(os.environ)
    e.update({'ORB_BT_FEATURES_CSV': BASE, 'ORB_BT_SIDECAR_CSV': SIDECARS,
              'ORB_BT_BOOK_OUT': book, 'ORB_BT_MONTHLY_OUT': f'{BOOKS}/monthly_{name}.csv',
              'ORB_BT_DUMP_CANDIDATES': f'{DUMPS}/dump_{name}.csv'})
    for k in list(e):
        if k.startswith('ORB_EXP_'):
            del e[k]
    e.update(env)
    if use_cache and os.path.exists(f'{DUMPS}/dump_baseline.csv'):
        e['ORB_BT_RESIM_CACHE'] = f'{DUMPS}/dump_baseline.csv'
    log = f'{ROOT}/logs/orb_study_{name}.log'
    t0 = time.time()
    with open(log, 'w') as lf:
        rc = subprocess.call(['bash', '-c', f'ulimit -v {VMEM} && exec {sys.executable} -u {PIPELINE}'],
                             cwd=ROOT, env=e, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [{name}] pipeline exit {rc} in {time.time() - t0:.0f}s (log {log})", flush=True)
    if rc != 0:
        raise RuntimeError(f"pipeline failed for {name}: see {log}")
    return book


# ---------------------------------------------------------------------------
# L2 book statistics + decision rules
# ---------------------------------------------------------------------------

def load_book(path: str) -> pd.DataFrame:
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    if 'entered' not in b.columns:
        b['entered'] = 1
    return b


def era_of(d: str) -> str:
    return 'TRAIN' if d <= TRAIN_END else ('OOS1' if d <= OOS1_END else 'OOS2')


def mdd(daily: pd.Series) -> float:
    c = daily.cumsum(); return float((c - c.cummax()).min()) if len(c) else 0.0


def book_stats(b: pd.DataFrame, window: str = 'OOS') -> dict:
    """window: 'ALL' | 'TRAIN' | 'OOS' (OOS1+OOS2)."""
    if window == 'TRAIN':
        s = b[b['date'] <= TRAIN_END]
    elif window == 'OOS':
        s = b[b['date'] > TRAIN_END]
    else:
        s = b
    daily = s.groupby('date')['_sized_pnl'].sum().sort_index()
    monthly = s.groupby(s['date'].str[:7])['_sized_pnl'].sum()
    ndays = len(pd.bdate_range(s['date'].min(), s['date'].max())) if len(s) else 1
    out = {'window': window, 'picks': len(s), 'fills': int((s['entered'] == 1).sum()),
           'pnl': round(float(s['_sized_pnl'].sum())), 'mdd': round(mdd(daily)),
           'neg_months': int((monthly < 0).sum()), 'months': len(monthly),
           'worst_month': round(float(monthly.min())) if len(monthly) else 0,
           'picks_per_day': round(len(s) / max(ndays, 1), 3)}
    for era in ('TRAIN', 'OOS1', 'OOS2'):
        out[f'pnl_{era}'] = round(float(b.loc[b['date'].map(era_of) == era, '_sized_pnl'].sum()))
    return out


def giants(b: pd.DataFrame, n: int = 10) -> set:
    return set(zip(*b.nlargest(n, '_sized_pnl')[['symbol', 'date']].T.values.tolist()))


def verdict_single(base: pd.DataFrame, var: pd.DataFrame, l1_ok: bool) -> dict:
    """§1a — all seven must hold for PROPOSE; L1-yes/L2-no = PARK; L1-no = REJECT."""
    sb, sv = book_stats(base, 'OOS'), book_stats(var, 'OOS')
    g_base = giants(base)
    g_kept = len(g_base & set(zip(var['symbol'], var['date'])))
    checks = {
        '1_pnl_+5%': sv['pnl'] >= sb['pnl'] * 1.05,
        '2_mdd': sv['mdd'] >= sb['mdd'] - 100,
        '3_neg_months': sv['neg_months'] <= sb['neg_months'],
        '4_eras': (sv['pnl_OOS1'] >= sb['pnl_OOS1'] - 250) and (sv['pnl_OOS2'] >= sb['pnl_OOS2'] - 250),
        '5_giants_kept>=8': g_kept >= 8,
        '6_picks_not_halved': sv['picks_per_day'] >= 0.5 * sb['picks_per_day'],
        '7_L1_effect': bool(l1_ok),
    }
    if not l1_ok:
        v = 'REJECT'
    elif all(checks.values()):
        v = 'PROPOSE'
    else:
        v = 'PARK'
    return {'verdict': v, 'checks': checks, 'giants_kept': g_kept, 'base': sb, 'var': sv}


def fmt_stats(s: dict) -> str:
    return (f"P&L {s['pnl']:+,} | MDD {s['mdd']:+,} | red months {s['neg_months']}/{s['months']} | "
            f"worst {s['worst_month']:+,} | picks {s['picks']} (fills {s['fills']}, {s['picks_per_day']}/day) | "
            f"eras T {s['pnl_TRAIN']:+,} / O1 {s['pnl_OOS1']:+,} / O2 {s['pnl_OOS2']:+,}")


# ---------------------------------------------------------------------------
# L1 — candidate-level evidence
# ---------------------------------------------------------------------------

def load_dump(name: str) -> pd.DataFrame:
    d = read_orb_csv(f'{DUMPS}/dump_{name}.csv')
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    d['era'] = d['date'].map(era_of)
    return d


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 42):
    """mean(a) − mean(b) with a bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    if len(a) < 5 or len(b) < 5:
        return float('nan'), (float('nan'), float('nan'))
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def l1_threshold(dump: pd.DataFrame, mask_keep: pd.Series, label: str) -> dict:
    """L1 for a keep/drop rule: entered rows only; kept − dropped mean pnl_pct."""
    e = dump[dump['entered'] == 1]
    k = mask_keep.reindex(e.index).fillna(True)
    rows = []
    signs = []
    for era in ('ALL', 'TRAIN', 'OOS1', 'OOS2'):
        s = e if era == 'ALL' else e[e['era'] == era]
        kk = k.reindex(s.index)
        a, b = s.loc[kk, 'pnl_pct'].to_numpy(), s.loc[~kk, 'pnl_pct'].to_numpy()
        diff, (lo, hi) = bootstrap_diff(a, b)
        rows.append({'era': era, 'n_keep': len(a), 'n_drop': len(b), 'mean_keep': round(float(a.mean()), 3) if len(a) else None,
                     'mean_drop': round(float(b.mean()), 3) if len(b) else None, 'diff': round(diff, 3), 'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3)})
        if era != 'ALL' and not np.isnan(diff):
            signs.append(diff > 0)
    pooled = rows[0]
    ok = (not np.isnan(pooled['diff'])) and pooled['ci_lo'] > 0 and sum(signs) >= 2
    return {'label': label, 'rows': rows, 'l1_ok': bool(ok)}


def l1_exit_delta(base_dump: pd.DataFrame, var_dump: pd.DataFrame, label: str) -> dict:
    """L1 for an exit/entry-mechanics variant: per-row pnl delta on fired rows."""
    m = base_dump.merge(var_dump[['symbol', 'date', 'pnl', 'exit_reason']], on=['symbol', 'date'], suffixes=('', '_v'))
    fired = m[m['exit_reason'] != m['exit_reason_v']]
    rows = []
    signs = []
    for era in ('ALL', 'TRAIN', 'OOS1', 'OOS2'):
        s = fired if era == 'ALL' else fired[fired['era'] == era]
        d = (s['pnl_v'] - s['pnl']).to_numpy()
        rng = np.random.RandomState(42)
        ci = (float(np.percentile([rng.choice(d, len(d)).mean() for _ in range(2000)], 2.5)),
              float(np.percentile([rng.choice(d, len(d)).mean() for _ in range(2000)], 97.5))) if len(d) >= 5 else (float('nan'),) * 2
        rows.append({'era': era, 'n_fired': len(d), 'mean_delta_pnl': round(float(d.mean()), 1) if len(d) else None,
                     'sum_delta': round(float(d.sum())) if len(d) else 0, 'ci_lo': round(ci[0], 1), 'ci_hi': round(ci[1], 1)})
        if era != 'ALL' and len(d):
            signs.append(d.mean() > 0)
    pooled = rows[0]
    ok = pooled['n_fired'] >= 5 and pooled['ci_lo'] > 0 and sum(signs) >= 2
    return {'label': label, 'rows': rows, 'l1_ok': bool(ok)}


def md_table(rows: list) -> str:
    if not rows:
        return "(none)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def phase_baseline(st: dict) -> None:
    if 'baseline' in st['done']:
        return
    status("P2 baseline pass on features_base.csv (flags off, candidate dump) — identity gate vs the production book")
    book = run_variant('baseline', {}, use_cache=False)
    b = load_book(book)
    prod = load_book(f'{ROOT}/analysis_results/orb_bplus_book.csv')
    same = (len(b) == len(prod)) and abs(b['_sized_pnl'].sum() - prod['_sized_pnl'].sum()) < 0.01
    s = book_stats(b, 'ALL')
    report(f"\n## Baseline (production stack, flags off)\n\n{fmt_stats(s)}\n\nOOS: {fmt_stats(book_stats(b, 'OOS'))}\n\n"
           f"Identity vs production book: **{'PASS' if same else 'FAIL'}** ({len(b)} vs {len(prod)} picks, "
           f"${b['_sized_pnl'].sum():,.0f} vs ${prod['_sized_pnl'].sum():,.0f})\n")
    if not same:
        raise SystemExit("baseline identity gate FAILED — stop")
    st['done']['baseline'] = book; save_state(st)


def phase_c5_read(st: dict) -> None:
    if 'c5_read' in st['done']:
        return
    status("C5 read — range/ATR14 tiers: L1 by tier and era, L2 book by tier")
    dump = load_dump('baseline')
    sc = read_orb_csv(f'{D}/sidecar_ratr.csv'); sc['date'] = pd.to_datetime(sc['date']).dt.strftime('%Y-%m-%d')
    d = dump.merge(sc, on=['symbol', 'date'], how='left')
    e = d[d['entered'] == 1]
    rows = []
    for tier in ('narrow', 'normal', 'wide', 'unknown'):
        for era in ('TRAIN', 'OOS1', 'OOS2'):
            s = e[(e['ratr_tier'] == tier) & (e['era'] == era)]
            rows.append({'tier': tier, 'era': era, 'n': len(s), 'mean_pnl_pct': round(float(s['pnl_pct'].mean()), 3) if len(s) else None,
                         'wr%': round(float((s['pnl_pct'] > 0).mean() * 100), 1) if len(s) else None})
    b = load_book(st['done']['baseline']).merge(sc, on=['symbol', 'date'], how='left')
    brows = []
    for tier in ('narrow', 'normal', 'wide', 'unknown'):
        for era in ('TRAIN', 'OOS1', 'OOS2'):
            s = b[(b['ratr_tier'] == tier) & (b['date'].map(era_of) == era)]
            brows.append({'tier': tier, 'era': era, 'picks': len(s), 'sized_pnl': round(float(s['_sized_pnl'].sum()))})
    neg_both = [t for t in ('narrow', 'normal', 'wide')
                if all(r['sized_pnl'] < 0 for r in brows if r['tier'] == t and r['era'] in ('OOS1', 'OOS2') and r['picks'] > 0)
                and any(r['picks'] > 0 for r in brows if r['tier'] == t and r['era'] in ('OOS1', 'OOS2'))]
    report(f"\n## C5 — range/ATR14 tier READ\n\nL1 (candidate level, entered rows):\n\n{md_table(rows)}\n\n"
           f"L2 (baseline book by tier):\n\n{md_table(brows)}\n\nTiers negative in BOTH OOS eras: {neg_both or 'none'} "
           f"→ {'veto test queued' if neg_both else 'no veto test (pre-registered condition not met)'}\n")
    st['done']['c5_read'] = neg_both; save_state(st)


def eval_single(st: dict, name: str, env: dict, l1: dict, use_cache: bool = True, note: str = '') -> dict:
    book = run_variant(name, env, use_cache=use_cache)
    base, var = load_book(st['done']['baseline']), load_book(book)
    v = verdict_single(base, var, l1['l1_ok'])
    st['verdicts'][name] = {'verdict': v['verdict'], 'checks': v['checks'], 'env': env, 'oos': v['var'], 'l1_ok': l1['l1_ok']}
    save_state(st)
    report(f"\n### {name} {note}\n\nenv: `{env}`\n\nL1 — {l1['label']}:\n\n{md_table(l1['rows'])}\n\n"
           f"L2 OOS baseline: {fmt_stats(v['base'])}\n\nL2 OOS variant:  {fmt_stats(v['var'])}\n\n"
           f"ALL-window variant: {fmt_stats(book_stats(var, 'ALL'))}\n\nchecks: {v['checks']} | giants kept {v['giants_kept']}/10\n\n"
           f"**VERDICT: {v['verdict']}**\n")
    status(f"{name}: {v['verdict']} (OOS P&L {v['var']['pnl']:+,} vs base {v['base']['pnl']:+,}, MDD {v['var']['mdd']:+,})")
    return v


def train_fit_threshold(st: dict, prefix: str, env_key: str, grid: list, extra_env: dict) -> float:
    """§1b: best TRAIN L2 P&L subject to TRAIN MDD >= base TRAIN MDD − 100."""
    base = load_book(st['done']['baseline']); sb = book_stats(base, 'TRAIN')
    best, best_pnl = None, -1e18
    rows = []
    for t in grid:
        env = dict(extra_env); env[env_key] = str(t)
        b = load_book(run_variant(f'{prefix}_t{t}', env))
        s = book_stats(b, 'TRAIN')
        ok = s['mdd'] >= sb['mdd'] - 100
        rows.append({'t': t, 'train_pnl': s['pnl'], 'train_mdd': s['mdd'], 'mdd_ok': ok, 'picks': s['picks']})
        if ok and s['pnl'] > best_pnl:
            best, best_pnl = t, s['pnl']
    report(f"\nTRAIN grid for `{env_key}` (base TRAIN {fmt_stats(sb)}):\n\n{md_table(rows)}\n\nchosen t = {best}"
           + (" (grid edge — reported, not extended)" if best in (grid[0], grid[-1]) else "") + "\n")
    return best


def phase_singles(st: dict) -> None:
    dump = load_dump('baseline')
    rv = read_orb_csv(f'{D}/sidecar_rvol.csv'); rv['date'] = pd.to_datetime(rv['date']).dt.strftime('%Y-%m-%d')
    dr = dump.merge(rv[['symbol', 'date', 'rvol_open5']], on=['symbol', 'date'], how='left')
    report("\n## Singles\n")
    # --- C1a rvol veto ---
    if 'C1a' not in st['verdicts']:
        status("C1a — rvol_open5 veto: TRAIN grid {0.5,1,1.5,2} then OOS")
        cov = dr['rvol_open5'].notna().mean() * 100
        report(f"\n### C1 data: rvol_open5 coverage {cov:.1f}% of candidates; median "
               f"{dr['rvol_open5'].median():.2f}; entered-row quintile read:\n")
        e = dr[(dr['entered'] == 1) & dr['rvol_open5'].notna()].copy()
        e['q'] = pd.qcut(e['rvol_open5'], 5, labels=['q1', 'q2', 'q3', 'q4', 'q5'], duplicates='drop')
        qrows = [{'rvol_q': q, 'n': len(g), 'mean_pnl_pct': round(float(g['pnl_pct'].mean()), 3), 'wr%': round(float((g['pnl_pct'] > 0).mean() * 100), 1),
                  'rvol_range': f"{g['rvol_open5'].min():.2f}-{g['rvol_open5'].max():.2f}"} for q, g in e.groupby('q', observed=True)]
        report(md_table(qrows) + "\n")
        t = train_fit_threshold(st, 'C1a', 'ORB_EXP_RVOL_VETO', [0.5, 1.0, 1.5, 2.0], {})
        keep = dr['rvol_open5'].isna() | (dr['rvol_open5'] >= t)
        l1 = l1_threshold(dr, keep, f"rvol_open5 >= {t} (kept) vs < {t} (dropped)")
        eval_single(st, 'C1a', {'ORB_EXP_RVOL_VETO': str(t)}, l1, note=f"rvol veto < {t} (TRAIN-fit)")
    # --- C1b rvol rank ---
    if 'C1b' not in st['verdicts']:
        status("C1b — rank the day's candidates by rvol_open5 (paper's top-N form)")
        e = dr[(dr['entered'] == 1) & dr['rvol_open5'].notna()]
        med = float(e['rvol_open5'].median())
        l1 = l1_threshold(dr, dr['rvol_open5'].isna() | (dr['rvol_open5'] >= med), f"above vs below median rvol ({med:.2f}) — rank-form proxy")
        eval_single(st, 'C1b', {'ORB_EXP_RVOL_RANK': '1'}, l1, note="rank by rvol desc, quintile/composite tie-break")
    # --- C2 ---
    for form in ('green', 'upper'):
        for gate in ('pre', 'post'):
            name = f'C2_{form}_{gate}'
            if name in st['verdicts']:
                continue
            status(f"{name} — range-direction gate")
            col = 'range_return_pct' if form == 'green' else 'range_close_position'
            keep = dump[col].isna() | ((dump[col] > 0) if form == 'green' else (dump[col] >= 0.5))
            l1 = l1_threshold(dump, keep, f"{form}: kept vs dropped")
            eval_single(st, name, {'ORB_EXP_RCP_GATE': gate, 'ORB_EXP_RCP_FORM': form}, l1)
    # --- C5 veto (only if the read queued it) ---
    for tier in (st['done'].get('c5_read') or []):
        name = f'C5_veto_{tier}'
        if name in st['verdicts']:
            continue
        status(f"{name} — range/ATR tier veto (pre-registered condition met)")
        lo, hi = {'narrow': (0.3, None), 'normal': (None, None), 'wide': (None, 0.6)}[tier]
        if lo is None and hi is None:
            continue
        env = {}
        if lo is not None: env['ORB_EXP_RATR_MIN'] = str(lo)
        if hi is not None: env['ORB_EXP_RATR_MAX'] = str(hi)
        sc = read_orb_csv(f'{D}/sidecar_ratr.csv'); sc['date'] = pd.to_datetime(sc['date']).dt.strftime('%Y-%m-%d')
        dd = dump.merge(sc, on=['symbol', 'date'], how='left')
        keep = dd['ratr_tier'] != tier
        l1 = l1_threshold(dd, keep, f"drop tier {tier}")
        eval_single(st, name, env, l1)
    # --- C3 mid-kill (full walk) ---
    if 'C3' not in st['verdicts']:
        status("C3 — midpoint-reversal kill (full bar walk, ~35 min)")
        run_variant('C3', {'ORB_EXP_MID_KILL': '1'}, use_cache=False)
        l1 = l1_exit_delta(dump, load_dump('C3'), "rows where the exit changed (mid_kill fired): variant − baseline pnl")
        eval_single(st, 'C3', {'ORB_EXP_MID_KILL': '1'}, l1, use_cache=False)
    # --- C4 re-arm (full walk) ---
    if 'C4' not in st['verdicts']:
        status("C4 — one re-arm after a tag exit (full bar walk, ~35 min)")
        run_variant('C4', {'ORB_EXP_REARM': '1'}, use_cache=False)
        l1 = l1_exit_delta(dump, load_dump('C4'), "rows where a re-arm fired: variant − baseline pnl")
        eval_single(st, 'C4', {'ORB_EXP_REARM': '1'}, l1, use_cache=False)


def phase_pairs(st: dict) -> None:
    ok = {k: v for k, v in st['verdicts'].items() if v['verdict'] in ('PROPOSE', 'PARK') and v['l1_ok']}
    proposed = [k for k, v in ok.items() if v['verdict'] == 'PROPOSE']
    report(f"\n## Pairs\n\nsurvivors (PROPOSE): {proposed} | L1-yes PARK: {[k for k, v in ok.items() if v['verdict'] == 'PARK']}\n")
    cands = [k for k in ok if k in ('C1a', 'C1b', 'C2_green_pre', 'C2_green_post', 'C2_upper_pre', 'C2_upper_post', 'C3', 'C4') or k.startswith('C5')]
    # C1a/C1b are alternatives; C2 forms are alternatives — take the best of each family
    fam = {}
    for k in cands:
        f = 'C1' if k.startswith('C1') else ('C2' if k.startswith('C2') else ('C5' if k.startswith('C5') else k))
        if f not in fam or st['verdicts'][k]['oos']['pnl'] > st['verdicts'][fam[f]]['oos']['pnl']:
            fam[f] = k
    keys = list(fam.values())
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    pairs = [p for p in pairs if any(st['verdicts'][x]['verdict'] == 'PROPOSE' for x in p)]
    base = load_book(st['done']['baseline']); sb = book_stats(base, 'OOS')
    rows = []
    for a, b in pairs:
        name = f'PAIR_{a}+{b}'
        env = {**st['verdicts'][a]['env'], **st['verdicts'][b]['env']}
        needs_walk = any(k in ('ORB_EXP_MID_KILL', 'ORB_EXP_REARM') for k in env)
        status(f"{name} — pair pass")
        bk = load_book(run_variant(name, env, use_cache=not needs_walk))
        sp = book_stats(bk, 'OOS')
        sa, sb_ = st['verdicts'][a]['oos'], st['verdicts'][b]['oos']
        inter = sp['pnl'] - sa['pnl'] - sb_['pnl'] + sb['pnl']
        best_single = max(sa['pnl'], sb_['pnl'])
        beats = sp['pnl'] > best_single and sp['mdd'] >= max(sa['mdd'], sb_['mdd']) and sp['neg_months'] <= min(sa['neg_months'], sb_['neg_months'])
        # interaction must hold in both OOS eras
        inter_eras = {era: sp[f'pnl_{era}'] - sa[f'pnl_{era}'] - sb_[f'pnl_{era}'] + sb[f'pnl_{era}'] for era in ('OOS1', 'OOS2')}
        kind = 'synergistic' if inter > 250 and all(v > 0 for v in inter_eras.values()) else ('redundant' if inter < -250 else 'additive')
        verdict = 'PROPOSE-PAIR' if (beats and kind != 'redundant') else ('SHIP-SINGLES' if kind == 'additive' else 'BEST-SINGLE')
        st['verdicts'][name] = {'verdict': verdict, 'env': env, 'oos': sp, 'interaction': inter, 'kind': kind, 'l1_ok': True}
        save_state(st)
        rows.append({'pair': name, 'oos_pnl': sp['pnl'], 'mdd': sp['mdd'], 'neg_months': sp['neg_months'], 'A_pnl': sa['pnl'], 'B_pnl': sb_['pnl'],
                     'interaction': round(inter), 'inter_OOS1': round(inter_eras['OOS1']), 'inter_OOS2': round(inter_eras['OOS2']), 'kind': kind, 'verdict': verdict})
    report(md_table(rows) + "\n" if rows else "(no eligible pairs)\n")


def phase_report(st: dict) -> None:
    rows = []
    for k, v in st['verdicts'].items():
        o = v['oos']
        rows.append({'variant': k, 'verdict': v['verdict'], 'oos_pnl': o['pnl'], 'mdd': o['mdd'], 'neg_months': o['neg_months'],
                     'picks': o['picks'], 'L1': v.get('l1_ok'), 'env': v['env']})
    base = book_stats(load_book(st['done']['baseline']), 'OOS')
    proposals = [k for k, v in st['verdicts'].items() if v['verdict'].startswith('PROPOSE')]
    parked = [k for k, v in st['verdicts'].items() if v['verdict'] == 'PARK']
    report(f"\n## Summary\n\nbaseline OOS: {fmt_stats(base)}\n\n{md_table(rows)}\n\n"
           f"### Proposals\n\n- PROPOSE (ship-candidates, to shadow live first): {proposals or 'none'}\n"
           f"- PARK (signal exists at L1, no book-level lift yet): {parked or 'none'}\n"
           f"- everything else: REJECT\n\nDecisions are joint: nothing ships from this file. "
           f"Reproduce any row: `ORB_BT_FEATURES_CSV={BASE} ORB_BT_SIDECAR_CSV={SIDECARS} <env> python3 study_orb_pipeline_static_lock.py`\n")
    with open(STATUS, 'w') as fh:
        fh.write(f"# ORB signal study — live status\n\nSTATUS: {now()} UTC — STUDY COMPLETE. Proposals: {proposals or 'none'}; parked: {parked or 'none'}. REPORT.md written.\nDONE\n")


def main() -> None:
    phase = sys.argv[sys.argv.index('--phase') + 1] if '--phase' in sys.argv else 'all'
    st = load_state()
    if not os.path.exists(REPORT):
        with open(REPORT, 'w') as fh:
            fh.write("# ORB entry-signal study — REPORT (auto-appended by run_study.py; protocol in DESIGN.md)\n")
    if phase in ('baseline', 'all'):
        phase_baseline(st)
    if phase in ('singles', 'all'):
        phase_c5_read(st)
        phase_singles(st)
    if phase in ('pairs', 'all'):
        phase_pairs(st)
    if phase in ('report', 'all'):
        phase_report(st)


if __name__ == '__main__':
    main()
