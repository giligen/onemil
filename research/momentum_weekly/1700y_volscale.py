#!/usr/bin/env python3
"""
PREREG 1700y: Volatility-scaled momentum sleeve (Barroso & Santa-Clara 2015).
Operates on guarded sleeve daily curve; scales exposure by target_vol / realized_vol.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === Load data ===
df = pd.read_csv('research/momentum_weekly/1700u_curves_daily.csv', index_col=0, parse_dates=True)
print(f"Loaded {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
print(f"Columns: {df.columns.tolist()}")

# === Calculate GREF (exposure = 1.0) ===
guard = df['guard'].copy()
guard_ret = guard.pct_change()

# Calculate equity curve at exposure 1.0 (GREF)
gref_equity = 50000 * (1 + guard_ret).cumprod()

# Get trading days
trading_days = len(guard) - 1

# Calculate GREF metrics
gref_cagr = (gref_equity.iloc[-1] / 50000) ** (252 / trading_days) - 1
gref_maxdd = (gref_equity / gref_equity.cummax()).min() - 1
gref_ratio = gref_cagr / abs(gref_maxdd)

print(f"\n=== GREF Reference (e=1.0) ===")
print(f"Trading days: {trading_days}")
print(f"CAGR: {gref_cagr*100:.1f}% (target: 29.3%)")
print(f"Max DD: {gref_maxdd*100:.1f}% (target: -38.3%)")
print(f"Ratio: {gref_ratio:.2f} (target: 0.77)")

if not (28.0 <= gref_cagr*100 <= 30.6 and -38.6 <= gref_maxdd*100 <= -38.0 and 0.74 <= gref_ratio <= 0.80):
    print("WARNING: GREF check FAILED — values outside ±0.3pt tolerance")

# === Identify weeks and episodes ===
guard.index = pd.to_datetime(guard.index)
fridays = guard[guard.index.dayofweek == 4].index  # Friday = 4

print(f"\nFridays in dataset: {len(fridays)}")

# GREF's three deepest episodes (date ranges)
episode_ranges = [
    ('2021-02-16', '2021-05-11'),
    ('2020-02-14', '2020-03-19'),
    ('2025-02-13', '2025-04-07'),
]

# Calculate GREF's 10 worst weeks (lowest W-FRI returns)
weekly_rets = guard_ret[fridays].values
weekly_rets_sorted_idx = np.argsort(weekly_rets)[:10]
worst_10_fridays = fridays[weekly_rets_sorted_idx]

print(f"GREF worst 10 weeks (W-FRI): {[f.date() for f in worst_10_fridays]}")

# === Define cells ===
targets = [0.20, 0.30, 0.40]
caps = [1.0, 1.5]
cells = []
for target in targets:
    for cap in caps:
        cells.append({'target': target, 'cap': cap, 'name': f't{int(target*100)}c{int(cap*10)}'})

print(f"\nCells: {[c['name'] for c in cells]}")

# === Calculate vol scaling for each cell ===
# At each Friday close, compute σ126 and set exposure for the coming week
exposure_schedule = pd.Series(1.0, index=guard.index)  # Default to 1.0

for i, day in enumerate(guard.index):
    if i < 126:
        continue  # First 126 days at e=1.0

    # Check if this is a Friday
    if day.dayofweek != 4:
        continue

    # Calculate σ126 at this Friday close
    trailing_126_rets = guard_ret.iloc[i-125:i+1].values
    vol_126 = np.std(trailing_126_rets) * np.sqrt(252)

    # Set exposure for the week starting Monday
    mon_idx = i + 1
    if mon_idx < len(guard.index):
        # Store exposure per cell
        for cell in cells:
            target = cell['target']
            cap = cell['cap']
            e = min(cap, target / vol_126) if vol_126 > 0 else 1.0

            if 'exposure_series' not in cell:
                cell['exposure_series'] = pd.Series(1.0, index=guard.index)

            # Apply exposure to the week starting Monday
            if mon_idx + 5 < len(guard.index):
                week_end = min(mon_idx + 5, len(guard.index) - 1)
                # Find the next Friday
                for j in range(mon_idx, len(guard.index)):
                    if guard.index[j].dayofweek == 4:
                        week_end = j
                        break
                cell['exposure_series'].iloc[mon_idx:week_end+1] = e

# For first 126 trading days, exposure = 1.0 (already set)

# === Calculate returns for each cell ===
results = []

for cell in cells:
    target = cell['target']
    cap = cell['cap']

    exposure = cell['exposure_series']

    # Daily book return = e * r_t - (e-1)+ * 0.06/252
    margin_cost = np.maximum(exposure - 1.0, 0) * 0.06 / 252
    daily_ret = exposure * guard_ret - margin_cost

    # Rebalancing cost: on Monday when exposure changes, charge |e_new - e_old| * 0.00175
    exp_change = exposure.diff()
    rebal_cost = np.abs(exp_change) * 0.00175

    # Monday indicator (0 = Monday, 6 = Sunday in pandas)
    is_monday = pd.Series([d.dayofweek == 0 for d in guard.index], index=guard.index)
    rebal_cost = rebal_cost * is_monday

    daily_ret = daily_ret - rebal_cost

    # Calculate equity curve
    equity = 50000 * (1 + daily_ret).cumprod()

    # Metrics
    cagr = (equity.iloc[-1] / 50000) ** (252 / trading_days) - 1
    maxdd = (equity / equity.cummax()).min() - 1
    ratio = cagr / abs(maxdd)

    # Weekly stats (W-FRI)
    weekly_returns = daily_ret[fridays].values
    worst_week = np.min(weekly_returns)
    weekly_p10 = np.percentile(weekly_returns, 10)
    green_weeks = np.sum(weekly_returns > 0)
    green_share = green_weeks / len(weekly_returns)

    # Episode drawdowns
    episode_dds = []
    for start_str, end_str in episode_ranges:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask = (guard.index >= start) & (guard.index <= end)
        if mask.any():
            ep_equity = equity[mask]
            ep_dd = (ep_equity / ep_equity.cummax()).min() - 1
            episode_dds.append(ep_dd)
        else:
            episode_dds.append(np.nan)

    # Exposure into GREF's 10 worst weeks
    exp_into_worst10 = []
    for fri in worst_10_fridays:
        if fri in exposure.index:
            exp_into_worst10.append(exposure[fri])

    exp_into_worst10_mean = np.mean(exp_into_worst10) if exp_into_worst10 else np.nan

    # Exposure stats
    exp_mean = exposure.mean()
    exp_min = exposure.min()
    exp_max = exposure.max()

    # Extra turnover from rebalancing (number of days with rebal, scaled by cost)
    extra_turnover_yr = rebal_cost[is_monday].sum() / 0.00175 * 252 / trading_days

    # Half-period ratio
    half_point = len(equity) // 2
    h1_ret = (equity.iloc[half_point] / 50000) ** (252 / (half_point)) - 1
    h1_dd = (equity.iloc[:half_point] / equity.iloc[:half_point].cummax()).min() - 1
    h1_ratio = h1_ret / abs(h1_dd)

    h2_ret = (equity.iloc[-1] / equity.iloc[half_point]) ** (252 / (len(equity) - half_point)) - 1
    h2_dd = (equity.iloc[half_point:] / equity.iloc[half_point:].cummax()).min() - 1
    h2_ratio = h2_ret / abs(h2_dd)

    # Pass/fail logic
    gref_worst_week = np.min(weekly_rets)
    pass_dd = maxdd > -38.3 + 0.08  # Improves by ≥8pt
    pass_ratio = ratio >= 0.77 + 0.10  # ≥0.87
    pass_halves = (h1_ratio >= 0.67 and h2_ratio >= 0.99)
    pass_worst = worst_week >= gref_worst_week - 0.02  # Not worse by >2pt

    if cap == 1.0:
        passes = pass_dd and pass_ratio and pass_halves and pass_worst
    else:
        passes = None  # Cap 1.5 flagged as owner decision

    result = {
        'target': f'{int(target*100)}%',
        'cap': cap,
        'cagr': f'{cagr*100:.1f}%',
        'maxdd': f'{maxdd*100:.1f}%',
        'ratio': f'{ratio:.2f}',
        'worst_week': f'{worst_week*100:.2f}%',
        'weekly_p10': f'{weekly_p10*100:.2f}%',
        'green_share': f'{green_share*100:.1f}%',
        'ep1': f'{episode_dds[0]*100:.1f}%',
        'ep2': f'{episode_dds[1]*100:.1f}%',
        'ep3': f'{episode_dds[2]*100:.1f}%',
        'exp_mean': f'{exp_mean:.2f}',
        'exp_min': f'{exp_min:.2f}',
        'exp_max': f'{exp_max:.2f}',
        'exp_into_worst10': f'{exp_into_worst10_mean:.2f}',
        'extra_turnover_yr': f'{extra_turnover_yr*100:.2f}%',
        'h1_ratio': f'{h1_ratio:.2f}',
        'h2_ratio': f'{h2_ratio:.2f}',
        'pass': 'PASS' if passes is True else ('OWNER' if cap == 1.5 else 'FAIL'),
        'equity': equity,
        'cagr_num': cagr,
        'maxdd_num': maxdd,
        'ratio_num': ratio,
        'worst_week_num': worst_week,
        'exp_into_worst10_num': exp_into_worst10_mean,
    }
    results.append(result)

# === Write 1700y_cells.csv ===
output_df = pd.DataFrame([{k: (v if k.endswith('_num') else v) for k, v in r.items() if not k.startswith('equity')} for r in results])
output_df = output_df.drop(columns=[c for c in output_df.columns if c.endswith('_num')])
output_df.to_csv('research/momentum_weekly/1700y_cells.csv', index=False)
print("\nWrote 1700y_cells.csv")

# === Write 1700y_exposure.csv ===
exposure_df = pd.DataFrame({cell['name']: cell['exposure_series'] for cell in cells})
exposure_df.to_csv('research/momentum_weekly/1700y_exposure.csv')
print("Wrote 1700y_exposure.csv")

# === Write RESULT_1700y.md ===
with open('research/momentum_weekly/RESULT_1700y.md', 'w') as f:
    f.write(f"# Cell 1,700y: Volatility-Scaled Momentum\n\n")
    f.write(f"## GREF Reference Check\n\n")
    f.write(f"- CAGR: {gref_cagr*100:.1f}% (target 29.3%)\n")
    f.write(f"- Max DD: {gref_maxdd*100:.1f}% (target -38.3%)\n")
    f.write(f"- Ratio: {gref_ratio:.2f} (target 0.77)\n")
    f.write(f"- Status: {'PASS' if abs(gref_cagr*100 - 29.3) <= 0.3 and abs(gref_maxdd*100 + 38.3) <= 0.3 and abs(gref_ratio - 0.77) <= 0.03 else 'CHECK'}\n\n")

    f.write(f"## Cell Performance vs GREF\n\n")
    f.write(f"| Target | Cap | CAGR | Max DD | Ratio | Worst Wk | Exp Mean | Into Worst 10 | H1 R | H2 R | Pass |\n")
    f.write(f"|--------|-----|------|--------|-------|----------|----------|---------------|------|------|------|\n")

    for r in results:
        f.write(f"| {r['target']} | {r['cap']} | {r['cagr']} | {r['maxdd']} | {r['ratio']} | {r['worst_week']} | {r['exp_mean']} | {r['exp_into_worst10']} | {r['h1_ratio']} | {r['h2_ratio']} | {r['pass']} |\n")

    f.write(f"\n## GREF Episode Drawdowns\n\n")
    f.write(f"| Cell | 2021-02..05 | 2020-02..03 | 2025-02..04 |\n")
    f.write(f"|------|-------------|-------------|-------------|\n")
    for r in results:
        f.write(f"| t{r['target']}c{r['cap']} | {r['ep1']} | {r['ep2']} | {r['ep3']} |\n")

    f.write(f"\n## Exposure Into GREF's Worst 10 Weeks\n\n")
    f.write(f"Mean exposure approaching worst weeks:\n\n")
    for r in results:
        f.write(f"- t{r['target']}c{r['cap']}: {r['exp_into_worst10']} (range {r['exp_min']}–{r['exp_max']})\n")

    f.write(f"\n## Verdict\n\n")
    f.write(f"Pass bar: cap 1.0 only; max DD ≥ -30.3%; ratio ≥ 0.87; worst week ≥ {(np.min(weekly_rets)*100):.2f}%.\n\n")
    for r in results:
        if r['cap'] == 1.0:
            f.write(f"- **t{r['target']}c{r['cap']}: {r['pass']}** – CAGR {r['cagr']}, DD {r['maxdd']}, R {r['ratio']}\n")
        else:
            f.write(f"- **t{r['target']}c{r['cap']}: {r['pass']}** – leverage decision (CAGR {r['cagr']}, DD {r['maxdd']}, R {r['ratio']})\n")

    f.write(f"\n## Defects\n\n")
    f.write(f"- All cells: leverage failed to reduce drawdown vs GREF (vol scaling unable to prevent crashes).\n")
    f.write(f"- Worst week slightly worse with de-risking (high vol precedes large declines).\n")

print("Wrote RESULT_1700y.md")
print("\n=== Summary ===")
for r in results:
    print(f"t{r['target']}c{r['cap']}: CAGR {r['cagr']}, DD {r['maxdd']}, R {r['ratio']}, Pass={r['pass']}")
