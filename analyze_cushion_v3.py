"""
Cushion strategy V3 — Focus on LOW PnL IMPACT strategies.

V2 showed pure cushion (50% start) cuts DD 59% but costs $97K (39%).
The 3-consec-stop costs only $18K (7%) for 22% DD cut.

Goal: Find rules that cut DD 50%+ while costing <15% PnL ($37K max).

Approach:
1. Explore stop-based rules (stop the day after bad sequences)
2. Explore mild scaling (75-90% start, not 50%)
3. Explore combo: mild scale + stop rules
4. Explore time-of-day based rules (are Mar-May losses concentrated?)
5. Explore intraday loss-cap (different from DD-based CB)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict


def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df['entry_minutes'] = df['entry_time'].dt.hour * 60 + df['entry_time'].dt.minute - 570  # mins from 9:30
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def metrics(daily_pnls_by_date, trade_pnls):
    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    cum = np.cumsum(daily_pnls)
    total = sum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    dd = abs(min(cum - peak)) if len(cum) > 0 else 0

    wins = sum(1 for p in trade_pnls if p > 0)
    wr = wins / len(trade_pnls) * 100 if trade_pnls else 0
    gp = sum(p for p in trade_pnls if p > 0)
    gl = abs(sum(p for p in trade_pnls if p <= 0))
    pf = gp / gl if gl > 0 else float('inf')

    mm = sum(v for d, v in daily_pnls_by_date.items() if date(2025, 3, 1) <= d <= date(2025, 5, 31))
    mm_pnls = [v for d, v in sorted(daily_pnls_by_date.items()) if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
    if mm_pnls:
        mc = np.cumsum(mm_pnls)
        mp = np.maximum.accumulate(mc)
        mm_dd = abs(min(mc - mp))
    else:
        mm_dd = 0

    calmar = total / dd if dd > 0 else float('inf')
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if len(daily_pnls) > 1 and np.std(daily_pnls) > 0 else 0

    return {
        'pnl': total, 'dd': dd, 'mm_pnl': mm, 'mm_dd': mm_dd,
        'wr': wr, 'trades': len(trade_pnls), 'pf': pf,
        'calmar': calmar, 'sharpe': sharpe
    }


def print_row(label, m, baseline_pnl, baseline_dd):
    pnl_delta = (m['pnl'] / baseline_pnl - 1) * 100
    dd_delta = (m['dd'] / baseline_dd - 1) * 100
    pf_str = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
    print(f"  {label:<55} ${m['pnl']:>9,.0f} ({pnl_delta:>+5.1f}%) ${m['dd']:>7,.0f} ({dd_delta:>+5.1f}%) ${m['mm_dd']:>7,.0f} {m['wr']:>5.1f} {m['trades']:>5} {m['calmar']:>6.1f} {m['sharpe']:>6.2f}")


def print_header():
    print(f"  {'Strategy':<55} {'PnL':>10} {'Δ%':>8} {'MaxDD':>8} {'Δ%':>8} {'MM DD':>8} {'WR%':>5} {'#':>5} {'Calmr':>6} {'Shrpe':>6}")
    print("  " + "-" * 135)


def main():
    df = load_trades('C:/Work/onemil/full_15mo.csv')
    total = len(df)
    print(f"Loaded {len(df)} trades, {df['date'].nunique()} days")
    print()

    # Baseline
    bl_tp = list(df['pnl'])
    bl_dpd = defaultdict(float)
    for _, r in df.iterrows():
        bl_dpd[r['date']] += r['pnl']
    bl = metrics(bl_dpd, bl_tp)
    BASE_PNL = bl['pnl']
    BASE_DD = bl['dd']

    # =========================================================================
    print("=" * 150)
    print("SECTION A: Stop rules only (no position scaling — all trades at 100%)")
    print("=" * 150)
    print_header()
    print_row("A0: Baseline (no rules)", bl, BASE_PNL, BASE_DD)

    # A1: Stop after N consecutive losses
    for n in [2, 3, 4]:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            cl, stopped = 0, False
            for _, r in dt.iterrows():
                if stopped: continue
                tp.append(r['pnl']); dpd[td] += r['pnl']
                if r['pnl'] > 0: cl = 0
                else:
                    cl += 1
                    if cl >= n: stopped = True
        print_row(f"A1: Stop after {n} consecutive losses", metrics(dpd, tp), BASE_PNL, BASE_DD)

    # A2: Stop after daily PnL < -$X (absolute loss cap)
    for cap in [1000, 1500, 2000, 2500]:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            cum, stopped = 0.0, False
            for _, r in dt.iterrows():
                if stopped: continue
                tp.append(r['pnl']); dpd[td] += r['pnl']
                cum += r['pnl']
                if cum <= -cap: stopped = True
        print_row(f"A2: Stop after daily loss > ${cap}", metrics(dpd, tp), BASE_PNL, BASE_DD)

    # A3: Stop after N total losses (not consecutive)
    for n in [3, 4, 5]:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            losses, stopped = 0, False
            for _, r in dt.iterrows():
                if stopped: continue
                tp.append(r['pnl']); dpd[td] += r['pnl']
                if r['pnl'] <= 0:
                    losses += 1
                    if losses >= n: stopped = True
        print_row(f"A3: Stop after {n} total losses in day", metrics(dpd, tp), BASE_PNL, BASE_DD)

    # A4: DD-based CB (stop when intraday peak-to-trough >= $X)
    for cb in [1000, 1500, 2000, 2500]:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            cum, peak, stopped = 0.0, 0.0, False
            for _, r in dt.iterrows():
                if stopped: continue
                tp.append(r['pnl']); dpd[td] += r['pnl']
                cum += r['pnl']; peak = max(peak, cum)
                if peak - cum >= cb: stopped = True
        print_row(f"A4: CB stop DD >= ${cb}", metrics(dpd, tp), BASE_PNL, BASE_DD)

    # A5: Max trades per day
    for mx in [2, 3, 4, 5]:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            count = 0
            for _, r in dt.iterrows():
                if count >= mx: continue
                tp.append(r['pnl']); dpd[td] += r['pnl']
                count += 1
        print_row(f"A5: Max {mx} trades per day", metrics(dpd, tp), BASE_PNL, BASE_DD)

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION B: Mild scaling (75-90% start, scale to 100% on win)")
    print("=" * 150)
    print_header()

    for start_pct in [0.90, 0.85, 0.80, 0.75, 0.70]:
        # B1: Scale up on first win, no scale down
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            scale = start_pct
            for _, r in dt.iterrows():
                pnl = r['pnl'] * scale
                tp.append(pnl); dpd[td] += pnl
                if r['pnl'] > 0: scale = 1.0
        m = metrics(dpd, tp)
        print_row(f"B1: {start_pct*100:.0f}%→100% on first win", m, BASE_PNL, BASE_DD)

    print()
    for start_pct in [0.90, 0.85, 0.80, 0.75, 0.70]:
        # B2: Scale up on win, scale down on loss
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            scale = start_pct
            for _, r in dt.iterrows():
                pnl = r['pnl'] * scale
                tp.append(pnl); dpd[td] += pnl
                if r['pnl'] > 0: scale = 1.0
                else: scale = start_pct
        m = metrics(dpd, tp)
        print_row(f"B2: {start_pct*100:.0f}%↔100% (up win, down loss)", m, BASE_PNL, BASE_DD)

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION C: COMBOS — mild scaling + stop rules")
    print("=" * 150)
    print_header()

    combos = [
        # (label, start_scale, scale_down_on_loss, consec_stop, daily_loss_cap, dd_cb, max_trades)
        ("C0: Baseline", 1.0, False, 0, 0, 0, 0),
        ("C1: 3 consec stop only", 1.0, False, 3, 0, 0, 0),
        ("C2: 3 consec + CB $2K", 1.0, False, 3, 0, 2000, 0),
        ("C3: 80%↔100% + 3 consec", 0.80, True, 3, 0, 0, 0),
        ("C4: 80%↔100% + CB $2K", 0.80, True, 0, 0, 2000, 0),
        ("C5: 80%↔100% + 3 consec + CB $2K", 0.80, True, 3, 0, 2000, 0),
        ("C6: 75%↔100% + 3 consec", 0.75, True, 3, 0, 0, 0),
        ("C7: 75%↔100% + CB $2K", 0.75, True, 0, 0, 2000, 0),
        ("C8: 75%↔100% + 3 consec + CB $2K", 0.75, True, 3, 0, 2000, 0),
        ("C9: max 3 trades/day", 1.0, False, 0, 0, 0, 3),
        ("C10: max 4 trades/day", 1.0, False, 0, 0, 0, 4),
        ("C11: 80%→100% (no down) + 3 consec", 0.80, False, 3, 0, 0, 0),
        ("C12: 85%↔100% + 3 consec", 0.85, True, 3, 0, 0, 0),
        ("C13: 90%↔100% + 3 consec", 0.90, True, 3, 0, 0, 0),
        ("C14: 80%↔100% + max 4 trades", 0.80, True, 0, 0, 0, 4),
        ("C15: 80%↔100% + 3 consec + max 5", 0.80, True, 3, 0, 0, 5),
        ("C16: 75%↔100% + 3 consec + max 5", 0.75, True, 3, 0, 0, 5),
        ("C17: 80%↔100% + 4 total losses stop", 0.80, True, 0, 0, 0, 0),  # special
        ("C18: daily loss cap $1500", 1.0, False, 0, 1500, 0, 0),
        ("C19: 80%↔100% + daily loss cap $1500", 0.80, True, 0, 1500, 0, 0),
        ("C20: 80%↔100% + daily loss cap $2K", 0.80, True, 0, 2000, 0, 0),
    ]

    for label, start, down_on_loss, cs, dlc, cb, mt in combos:
        tp, dpd = [], defaultdict(float)
        for td, dt in df.groupby('date'):
            dt = dt.sort_values('entry_time')
            scale = start
            cum, peak, cl, total_losses, stopped, count = 0.0, 0.0, 0, 0, False, 0

            for _, r in dt.iterrows():
                if stopped: continue
                if mt > 0 and count >= mt: continue

                pnl = r['pnl'] * scale
                tp.append(pnl); dpd[td] += pnl
                cum += pnl; peak = max(peak, cum); count += 1

                if r['pnl'] > 0:
                    scale = 1.0
                    cl = 0
                else:
                    cl += 1
                    total_losses += 1
                    if down_on_loss: scale = start

                    if cs > 0 and cl >= cs: stopped = True

                    # Special: C17 uses total losses
                    if label == "C17: 80%↔100% + 4 total losses stop" and total_losses >= 4:
                        stopped = True

                if dlc > 0 and cum <= -dlc: stopped = True
                if cb > 0 and peak - cum >= cb: stopped = True

        print_row(label, metrics(dpd, tp), BASE_PNL, BASE_DD)

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION D: Time analysis — where do Mar-May losses come from?")
    print("=" * 150)

    mm_df = df[(df['date'] >= date(2025, 3, 1)) & (df['date'] <= date(2025, 5, 31))]
    rest_df = df[~((df['date'] >= date(2025, 3, 1)) & (df['date'] <= date(2025, 5, 31)))]

    print(f"\n  Mar-May 2025: {len(mm_df)} trades, PnL=${mm_df['pnl'].sum():,.0f}")
    print(f"  Rest of period: {len(rest_df)} trades, PnL=${rest_df['pnl'].sum():,.0f}")

    # Trade order within day
    print(f"\n  Mar-May P&L by trade # within day:")
    for td, dt in mm_df.groupby('date'):
        dt = dt.sort_values('entry_time').reset_index()
        for i, (_, r) in enumerate(dt.iterrows()):
            pass  # just need the grouping

    # Better: compute PnL by trade position
    trade_pos_pnl = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0})
    for period_name, period_df in [("Mar-May", mm_df), ("Rest", rest_df)]:
        pos_data = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0})
        for td, dt in period_df.groupby('date'):
            dt = dt.sort_values('entry_time').reset_index(drop=True)
            for i, (_, r) in enumerate(dt.iterrows()):
                pos = min(i, 5)  # cap at 5+
                pos_data[pos]['pnl'] += r['pnl']
                pos_data[pos]['count'] += 1
                if r['pnl'] > 0: pos_data[pos]['wins'] += 1

        print(f"\n  {period_name} — PnL by trade position in day:")
        for pos in sorted(pos_data.keys()):
            d = pos_data[pos]
            wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
            avg = d['pnl'] / d['count'] if d['count'] > 0 else 0
            label = f"#{pos+1}" if pos < 5 else "#6+"
            print(f"    Trade {label}: {d['count']:>4} trades, WR={wr:.0f}%, PnL=${d['pnl']:>8,.0f}, Avg=${avg:>6,.0f}")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION E: Analysis — what happens AFTER a losing first trade?")
    print("=" * 150)

    # For each day, classify: first trade win vs loss, then look at rest-of-day PnL
    first_win_rest = []
    first_loss_rest = []

    for td, dt in df.groupby('date'):
        dt = dt.sort_values('entry_time').reset_index(drop=True)
        if len(dt) == 0: continue
        first_pnl = dt.iloc[0]['pnl']
        rest_pnl = dt.iloc[1:]['pnl'].sum() if len(dt) > 1 else 0

        if first_pnl > 0:
            first_win_rest.append({'date': td, 'first': first_pnl, 'rest': rest_pnl, 'total': first_pnl + rest_pnl, 'n_rest': len(dt) - 1})
        else:
            first_loss_rest.append({'date': td, 'first': first_pnl, 'rest': rest_pnl, 'total': first_pnl + rest_pnl, 'n_rest': len(dt) - 1})

    fw = pd.DataFrame(first_win_rest)
    fl = pd.DataFrame(first_loss_rest)

    print(f"\n  Days where FIRST trade WINS: {len(fw)}")
    print(f"    Avg first trade PnL: ${fw['first'].mean():,.0f}")
    print(f"    Avg rest-of-day PnL: ${fw['rest'].mean():,.0f}")
    print(f"    Avg total day PnL:   ${fw['total'].mean():,.0f}")
    print(f"    Rest-of-day win rate: {(fw['rest'] > 0).mean()*100:.0f}%")

    print(f"\n  Days where FIRST trade LOSES: {len(fl)}")
    print(f"    Avg first trade PnL: ${fl['first'].mean():,.0f}")
    print(f"    Avg rest-of-day PnL: ${fl['rest'].mean():,.0f}")
    print(f"    Avg total day PnL:   ${fl['total'].mean():,.0f}")
    print(f"    Rest-of-day win rate: {(fl['rest'] > 0).mean()*100:.0f}%")

    # What about after 2 consecutive losses?
    print(f"\n  After 2 consecutive losses at start of day:")
    two_loss_start = []
    for td, dt in df.groupby('date'):
        dt = dt.sort_values('entry_time').reset_index(drop=True)
        if len(dt) < 2: continue
        if dt.iloc[0]['pnl'] <= 0 and dt.iloc[1]['pnl'] <= 0:
            rest_pnl = dt.iloc[2:]['pnl'].sum() if len(dt) > 2 else 0
            n_rest = len(dt) - 2
            two_loss_start.append({'date': td, 'first2': dt.iloc[0]['pnl'] + dt.iloc[1]['pnl'],
                                    'rest': rest_pnl, 'n_rest': n_rest})

    tl = pd.DataFrame(two_loss_start)
    if len(tl) > 0:
        print(f"    Occurs on {len(tl)} days ({len(tl)/df['date'].nunique()*100:.0f}% of days)")
        print(f"    Avg first 2 trades PnL: ${tl['first2'].mean():,.0f}")
        print(f"    Avg rest-of-day PnL:    ${tl['rest'].mean():,.0f}")
        print(f"    Rest positive: {(tl['rest'] > 0).mean()*100:.0f}% of days")
        print(f"    Total rest-of-day PnL:  ${tl['rest'].sum():,.0f}")

    # =========================================================================
    # Section F: What about the losing STREAKS specifically?
    print()
    print("=" * 150)
    print("SECTION F: Worst days deep dive")
    print("=" * 150)

    daily_pnls = defaultdict(float)
    daily_trades = defaultdict(int)
    for _, r in df.iterrows():
        daily_pnls[r['date']] += r['pnl']
        daily_trades[r['date']] += 1

    worst_days = sorted(daily_pnls.items(), key=lambda x: x[1])[:20]
    print(f"\n  20 worst days:")
    print(f"  {'Date':>12} {'PnL':>10} {'Trades':>7}")
    print("  " + "-" * 35)
    for d, pnl in worst_days:
        print(f"  {d}  ${pnl:>9,.0f}  {daily_trades[d]:>5}")

    # How much of total DD comes from worst N days?
    sorted_daily = sorted(daily_pnls.values())
    print(f"\n  Contribution of worst N days to total losses:")
    total_losses_all = sum(p for p in daily_pnls.values() if p < 0)
    for n in [5, 10, 15, 20, 30]:
        worst_n = sum(sorted_daily[:n])
        pct = worst_n / total_losses_all * 100 if total_losses_all != 0 else 0
        print(f"    Worst {n:>2} days: ${worst_n:>10,.0f} ({pct:.0f}% of all losses)")


if __name__ == '__main__':
    main()
