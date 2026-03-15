"""
Cushion V4 — Stack cheap rules to maximize DD reduction per PnL cost.

Key findings from V3:
- Max 5 trades/day: -1.5% PnL, -20.7% DD (almost free)
- 4 total losses/day: -1.9% PnL, -14.9% DD
- 3 consec losses: -7.2% PnL, -21.5% DD
- 80%↔100% scaling: -15.7% PnL, -23.5% DD

This script tests STACKING these rules and explores new combos.
"""

import pandas as pd
import numpy as np
from datetime import date
from collections import defaultdict


def load_trades(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def run(df, start_scale=1.0, scale_down_on_loss=False, max_trades=0,
        consec_stop=0, total_loss_stop=0, daily_loss_cap=0,
        dd_cb=0, prev_day_scale=False, scale_after_n_wins=0):
    """Generalized strategy runner with all rule toggles."""
    tp, dpd = [], defaultdict(float)
    dates = sorted(df['date'].unique())
    prev_day_pnl = 0

    for trade_date in dates:
        day_df = df[df['date'] == trade_date].sort_values('entry_time')

        # Previous day scaling
        if prev_day_scale and prev_day_pnl < 0:
            scale = start_scale
        else:
            scale = start_scale if not prev_day_scale else 1.0

        cum, peak, cl, tl, wins, stopped, count = 0.0, 0.0, 0, 0, 0, False, 0

        for _, r in day_df.iterrows():
            if stopped: continue
            if max_trades > 0 and count >= max_trades: continue

            pnl = r['pnl'] * scale
            tp.append(pnl); dpd[trade_date] += pnl
            cum += pnl; peak = max(peak, cum); count += 1

            if r['pnl'] > 0:
                cl = 0; wins += 1
                if scale_after_n_wins > 0 and wins >= scale_after_n_wins:
                    scale = 1.0
                elif scale_down_on_loss:
                    scale = 1.0  # scale up on win
            else:
                cl += 1; tl += 1
                if scale_down_on_loss: scale = start_scale

                if consec_stop > 0 and cl >= consec_stop: stopped = True
                if total_loss_stop > 0 and tl >= total_loss_stop: stopped = True

            if daily_loss_cap > 0 and cum <= -daily_loss_cap: stopped = True
            if dd_cb > 0 and peak - cum >= dd_cb: stopped = True

        prev_day_pnl = cum

    return tp, dpd


def m(dpd, tp):
    """Compute metrics."""
    daily = [dpd[d] for d in sorted(dpd)]
    cum = np.cumsum(daily)
    total = sum(daily)
    pk = np.maximum.accumulate(cum)
    dd = abs(min(cum - pk)) if len(cum) > 0 else 0

    wins = sum(1 for p in tp if p > 0)
    wr = wins / len(tp) * 100 if tp else 0

    mm_pnls = [v for d, v in sorted(dpd.items()) if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
    mm_pnl = sum(mm_pnls)
    if mm_pnls:
        mc = np.cumsum(mm_pnls)
        mp = np.maximum.accumulate(mc)
        mm_dd = abs(min(mc - mp))
    else:
        mm_dd = 0

    calmar = total / dd if dd > 0 else 0
    sharpe = (np.mean(daily) / np.std(daily)) * np.sqrt(252) if len(daily) > 1 and np.std(daily) > 0 else 0

    return total, dd, mm_pnl, mm_dd, wr, len(tp), calmar, sharpe


def pr(label, total, dd, mm_pnl, mm_dd, wr, trades, calmar, sharpe, bp, bd):
    pd_ = (total / bp - 1) * 100
    dd_ = (dd / bd - 1) * 100
    print(f"  {label:<60} ${total:>9,.0f} ({pd_:>+5.1f}%) ${dd:>7,.0f} ({dd_:>+5.1f}%) ${mm_pnl:>8,.0f} ${mm_dd:>7,.0f} {wr:>5.1f} {trades:>5} {calmar:>6.1f}")


def main():
    df = load_trades('C:/Work/onemil/full_15mo.csv')
    print(f"Loaded {len(df)} trades, {df['date'].nunique()} days\n")

    # Baseline
    bt, bd_ = run(df)
    bp, bdd, bmm, bmm_dd, bwr, bn, bcal, bsh = m(bd_, bt)

    hdr = f"  {'Strategy':<60} {'PnL':>10} {'Δ%':>8} {'MaxDD':>8} {'Δ%':>8} {'MM PnL':>9} {'MM DD':>8} {'WR%':>5} {'#':>5} {'Calmr':>6}"
    sep = "  " + "-" * 140

    # =========================================================================
    print("=" * 155)
    print("STACKING CHEAP RULES (each rule independently costs <10% PnL)")
    print("=" * 155)
    print(hdr); print(sep)

    configs = [
        ("Baseline", {}),
        ("Max 5 trades/day", dict(max_trades=5)),
        ("4 total losses stop", dict(total_loss_stop=4)),
        ("3 consec losses stop", dict(consec_stop=3)),
        ("Max 5 + 4 total losses", dict(max_trades=5, total_loss_stop=4)),
        ("Max 5 + 3 consec", dict(max_trades=5, consec_stop=3)),
        ("Max 4 + 4 total losses", dict(max_trades=4, total_loss_stop=4)),
        ("Max 4 + 3 consec", dict(max_trades=4, consec_stop=3)),
        ("Max 5 + 4 total + 3 consec", dict(max_trades=5, total_loss_stop=4, consec_stop=3)),
        ("Max 4 + 4 total + 3 consec", dict(max_trades=4, total_loss_stop=4, consec_stop=3)),
        ("Max 5 + 3 consec + DLC $1500", dict(max_trades=5, consec_stop=3, daily_loss_cap=1500)),
    ]

    for label, kw in configs:
        t, d = run(df, **kw)
        pr(label, *m(d, t), bp, bdd)

    # =========================================================================
    print()
    print("=" * 155)
    print("MILD SCALING (85-90%) + CHEAP STOPS")
    print("=" * 155)
    print(hdr); print(sep)

    configs2 = [
        ("Baseline", {}),
        ("90%↔100%", dict(start_scale=0.90, scale_down_on_loss=True)),
        ("90%↔100% + max 5", dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5)),
        ("90%↔100% + 3 consec", dict(start_scale=0.90, scale_down_on_loss=True, consec_stop=3)),
        ("90%↔100% + max 5 + 3 consec", dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5, consec_stop=3)),
        ("90%↔100% + max 5 + 4 total", dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5, total_loss_stop=4)),
        ("85%↔100%", dict(start_scale=0.85, scale_down_on_loss=True)),
        ("85%↔100% + max 5", dict(start_scale=0.85, scale_down_on_loss=True, max_trades=5)),
        ("85%↔100% + 3 consec", dict(start_scale=0.85, scale_down_on_loss=True, consec_stop=3)),
        ("85%↔100% + max 5 + 3 consec", dict(start_scale=0.85, scale_down_on_loss=True, max_trades=5, consec_stop=3)),
        ("85%↔100% + max 4 + 3 consec", dict(start_scale=0.85, scale_down_on_loss=True, max_trades=4, consec_stop=3)),
        ("85%↔100% + max 5 + 4 total", dict(start_scale=0.85, scale_down_on_loss=True, max_trades=5, total_loss_stop=4)),
        ("80%↔100% + max 5", dict(start_scale=0.80, scale_down_on_loss=True, max_trades=5)),
        ("80%↔100% + max 5 + 3 consec", dict(start_scale=0.80, scale_down_on_loss=True, max_trades=5, consec_stop=3)),
    ]

    for label, kw in configs2:
        t, d = run(df, **kw)
        pr(label, *m(d, t), bp, bdd)

    # =========================================================================
    print()
    print("=" * 155)
    print("PREVIOUS DAY SCALING — scale down after losing days")
    print("=" * 155)
    print(hdr); print(sep)

    configs3 = [
        ("Baseline", {}),
        ("90% after losing day", dict(start_scale=0.90, prev_day_scale=True)),
        ("85% after losing day", dict(start_scale=0.85, prev_day_scale=True)),
        ("80% after losing day", dict(start_scale=0.80, prev_day_scale=True)),
        ("75% after losing day", dict(start_scale=0.75, prev_day_scale=True)),
        ("80% after losing day + max 5", dict(start_scale=0.80, prev_day_scale=True, max_trades=5)),
        ("80% after losing day + 3 consec", dict(start_scale=0.80, prev_day_scale=True, consec_stop=3)),
        ("80% after losing day + max 5 + 3 consec", dict(start_scale=0.80, prev_day_scale=True, max_trades=5, consec_stop=3)),
        ("75% after losing day + max 5 + 3 consec", dict(start_scale=0.75, prev_day_scale=True, max_trades=5, consec_stop=3)),
    ]

    for label, kw in configs3:
        t, d = run(df, **kw)
        pr(label, *m(d, t), bp, bdd)

    # =========================================================================
    print()
    print("=" * 155)
    print("FIRST N WINS TO SCALE — don't scale up on first win, require N wins")
    print("=" * 155)
    print(hdr); print(sep)

    for start_s in [0.85, 0.80, 0.75]:
        for n_wins in [1, 2, 3]:
            label = f"{start_s*100:.0f}%→100% after {n_wins} win(s)"
            t, d = run(df, start_scale=start_s, scale_after_n_wins=n_wins)
            pr(label, *m(d, t), bp, bdd)

    # =========================================================================
    print()
    print("=" * 155)
    print("TOP CANDIDATES — best PnL/DD tradeoff with <15% PnL cost")
    print("=" * 155)
    print(hdr); print(sep)

    top = [
        ("BASELINE", {}),
        # Pure stops
        ("★ Max 5 trades/day", dict(max_trades=5)),
        ("★ Max 5 + 3 consec", dict(max_trades=5, consec_stop=3)),
        ("★ Max 5 + 4 total losses", dict(max_trades=5, total_loss_stop=4)),
        ("★ Max 4 trades/day", dict(max_trades=4)),
        # Mild scale + stops
        ("★ 90%↔100% + max 5", dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5)),
        ("★ 90%↔100% + max 5 + 3 consec", dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5, consec_stop=3)),
        ("★ 85%↔100% + max 5", dict(start_scale=0.85, scale_down_on_loss=True, max_trades=5)),
        # Previous day
        ("★ 80% after losing day + max 5", dict(start_scale=0.80, prev_day_scale=True, max_trades=5)),
    ]

    for label, kw in top:
        t, d = run(df, **kw)
        pr(label, *m(d, t), bp, bdd)

    # =========================================================================
    # Monthly breakdown for top 3
    print()
    print("=" * 155)
    print("MONTHLY BREAKDOWN — Top strategies")
    print("=" * 155)

    strats = {
        'Baseline': {},
        'Max5+3C': dict(max_trades=5, consec_stop=3),
        '90%+Max5': dict(start_scale=0.90, scale_down_on_loss=True, max_trades=5),
        '85%+Max5': dict(start_scale=0.85, scale_down_on_loss=True, max_trades=5),
        'Max4': dict(max_trades=4),
    }

    strat_dpd = {}
    for name, kw in strats.items():
        t, d = run(df, **kw)
        strat_dpd[name] = d

    months = sorted(set((d.year, d.month) for d in strat_dpd['Baseline']))
    header = f"{'Month':>10}"
    for n in strats: header += f" {n:>12}"
    print(header)
    print("-" * (10 + 13 * len(strats)))

    for yr, mo in months:
        line = f"  {yr}-{mo:02d}"
        for n in strats:
            pnl = sum(v for d, v in strat_dpd[n].items() if d.year == yr and d.month == mo)
            line += f" ${pnl:>10,.0f}"
        print(line)

    print()
    line = "  TOTAL  "
    for n in strats:
        line += f" ${sum(strat_dpd[n].values()):>10,.0f}"
    print(line)

    line = "  MaxDD  "
    for n in strats:
        daily = [strat_dpd[n][d] for d in sorted(strat_dpd[n])]
        cum = np.cumsum(daily)
        pk = np.maximum.accumulate(cum)
        dd = abs(min(cum - pk))
        line += f" ${dd:>10,.0f}"
    print(line)

    line = "  MM DD  "
    for n in strats:
        mm = [v for d, v in sorted(strat_dpd[n].items()) if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
        if mm:
            mc = np.cumsum(mm); mp = np.maximum.accumulate(mc)
            mm_dd = abs(min(mc - mp))
        else:
            mm_dd = 0
        line += f" ${mm_dd:>10,.0f}"
    print(line)

    line = "  Calmar "
    for n in strats:
        daily = [strat_dpd[n][d] for d in sorted(strat_dpd[n])]
        cum = np.cumsum(daily)
        pk = np.maximum.accumulate(cum)
        dd = abs(min(cum - pk))
        total = sum(daily)
        cal = total / dd if dd > 0 else 0
        line += f" {cal:>12.1f}"
    print(line)


if __name__ == '__main__':
    main()
