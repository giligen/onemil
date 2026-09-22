# Data Conventions for HOD Exit Lab

## 1. features.csv (12,135 signals)

**Path:** `/home/ec2-user/onemil/research/bf_zero/causal_filter/features.csv`

**Header columns:** day, symbol, entry_m, exit_m, entry, stop, target, r_pct, rr, why, level, dist_open_pct, rv_profile, adv20, is_wrapper, price, bar_vol_x, vwap_prev, drive_min, cumv_entry, open_px, prev_close, prev_high, prev_low, high20, spy_range3, spy_5m_ret, gap_pct, prev_range_pct, dist_20d_high_pct, above_vwap, n_prior, rv_clock, anchor, coh_by_t, news_covered, has_news, cohort, split, half, wk

**Sample rows:**
```
2025-01-10,BG,727,955,81.725,80.44,84.295,1.572,−0.401,eod,81.43,5.030,1.704,1114258.8,0,81.725,0.737,79.317,155,784329.0,77.53,77.38,77.86,75.75,79.44,1.267,0.003,0.194,2.727,2.876,1.0,0.0,,BG,0,1,0,pit,TRAIN,H1,2025-01-04/2025-01-10
2025-01-10,BLBD,593,955,41.48,40.7,43.040,1.880,−0.039,eod,41.64,5.525,3.063,608049.8,0,41.48,0.246,40.467,21,96055.0,39.46,40.29,40.51,38.83,43.531,1.267,−0.005,−2.060,4.170,−4.712,1.0,3.0,,BLBD,0,1,0,pit,TRAIN,H1,2025-01-04/2025-01-10
2025-01-10,FTAI,622,837,176.74,174.42,181.38,1.313,2.000,target,176.5,5.060,2.383,1690230.2,0,176.74,1.629,173.379,50,402933.0,168.0,168.78,169.38,158.49,169.38,1.267,0.114,−0.462,6.452,4.345,1.0,3.0,,FTAI,0,1,0,pit,TRAIN,H1,2025-01-04/2025-01-10
```

**Field meanings & format:**
- **day**: YYYY-MM-DD (date signals fired)
- **entry_m**: integer, minutes since midnight (00:00); market close 16:00 = 960 min; live config restricted to ≤ 840 (14:00 ET)
- **entry**: float, entry price in dollars
- **stop**: float, stop price in dollars (typically 1–2% below entry for HOD break)
- **r_pct**: float, risk as percentage of entry price; used to calculate R-based metrics
- **price**: float, same as entry (the signal price level)
- **n_prior**: integer, count of prior HOD-break signals on the same symbol same day before this one; 0 = first of day

---

## 2. bars_sip.db (SIP 1-minute bars)

**Path:** `/home/ec2-user/onemil/research/bf_zero/bars_sip.db`

**Schema:**
```sql
CREATE TABLE bars (
  symbol TEXT,      -- ticker symbol
  day TEXT,         -- date YYYY-MM-DD
  t TEXT,           -- time of day HH:MM (Eastern time, market hours only)
  o REAL,           -- open price
  h REAL,           -- high price
  l REAL,           -- low price
  c REAL,           -- close price
  v REAL,           -- volume (shares)
  PRIMARY KEY (symbol, day, t)
);
CREATE INDEX idx_bars_day on bars(day);
```

**Time format:** HH:MM (24-hour, Eastern time, no UTC conversion; market opens 09:30, closes 16:00)
- **First bar of day:** 09:30 (market open)
- **Last bar of day:** 15:59 (last minute of regular hours; no 16:00 bar)
- **Premarket rows:** None — database contains regular hours (09:30–15:59) only
- **Date coverage:** 2025-01-02 to 2026-09-04 (after 9/22 backfill, 100% coverage)

---

## 3. cells.py (TRAIN/VAL/TEST split, R basis, cost, day-clustered t)

**Path:** `/home/ec2-user/onemil/research/bf_zero/causal_filter/cells.py`

**Split definitions** (from `assemble.py`):
```python
c['split'] = np.where(
  c.day < '2026-01-01', 'TRAIN',
  np.where(c.day < '2026-06-01', 'VAL', 'TEST')
)
c['half'] = np.where(
  c.day < '2025-07-01', 'H1',
  np.where(c.day < '2026-01-01', 'H2', '')
)
```
- **TRAIN:** day < 2026-01-01 (all of 2025; includes H1 2025-01 to 06-30 and H2 2025-07-01 to 12-31); NW = 53 weeks
- **VAL:** 2026-01-01 ≤ day < 2026-06-01 (Jan–May 2026); NW = 23 weeks
- **TEST:** day ≥ 2026-06-01 (Jun–Sep 2026, sealed per CAUSAL_FILTER_FREEZE.md)

**R basis** (from `cells.py` load() and main()):
```python
# Two cost arms:
for tag, sp in (('band', c.sp_band), ('meas', c.sp_pct)):
  half = 0.5 * sp / c.r_pct.clip(lower=0.05)
  c[f'net_{tag}'] = c.rr - half - half * ratio
```
- **'meas'** (measured NBBO): per-signal spread_mean (mean NBBO half-spread over signal minute) from `nbbo.csv`
  - Columns: spread_mean, spread_med, ask_dec, bid_dec
  - Includes obtainability rail: ask_dec ≤ entry × (1 + 0.006) to fill
- **'band'** (banded constant): (price band × hour band) mean spread% from `research/lit_review_2026/cost_curve.csv`
  - Price bands: <$20, $20–30, $30–50, $50–100, $100+
  - Hour bands: 09:30–09:45, 09:45–10:00, 10:00–11:00, 11:00–13:00, 13:00+

**Day-clustered t statistic** (from `cells.py` stats()):
```python
se = t.net.std() / np.sqrt(len(t))
t_stat = round(float(t.net.mean() / se), 2) if se > 0 else np.nan
```
Reported alongside point estimate of mean R; clustering is by day via groupby('wk').

---

## 4. SPY_1min.parquet (Index context; cells D1)

**Path:** `/home/ec2-user/onemil/research/index_orb/cache/SPY_1min.parquet`

**Columns:** timestamp, open, high, low, close, volume

**Timestamp convention:**
- dtype: datetime64[ns, America/New_York] (Eastern timezone-aware, daylight/standard time observed)
- Format: YYYY-MM-DD HH:MM:00 (1-minute bars, starts 2016-01-04)
- Market hours only (09:30–16:00; includes closing auction bar at 16:00)
- Coverage: through 2026-05-31 per PREREG (D1 cell needs SPY open at signal minute)

---

## 5. hmm_labels.csv (Regime state; cell D4)

**Path:** `/home/ec2-user/onemil/research/regime/hmm_labels.csv`

**Columns:** bar_date, close, ret, vol20, hmm_state

**States and volatility:**
- hmm_state 0: mean vol20 = 0.1178 (394 rows) — **LOWEST**
- hmm_state 1: mean vol20 = 0.2005 (60 rows)
- hmm_state 2: mean vol20 = 0.4673 (25 rows)

**D4 rule:** keep trades when bar_date matches an hmm_state == 0 (calm regime).

---

## 6. cadence_bar.py (Pass bar scorer)

**Path:** `/home/ec2-user/onemil/scripts/cadence_bar.py`

**Usage:**
```
python scripts/cadence_bar.py --trades trades.csv --split TRAIN
python scripts/cadence_bar.py --trades trades.csv --split VAL \
    --tail-audit tail_audit.csv --book orb --slots 4 \
    --fill-model "next-open capped" --r-dollars 375
```

**Expected input columns:** date (YYYY-MM-DD), pnl_R (in R units), and optional book/fill fields.
**Pass bar (per PREREG §7):** runs cadence_bar.py on TRAIN and VAL for B0 baseline and each passing cell; metrics include green-week %, weekly MDD, P10 floor.

---

## Summary for Harness

The harness will:
1. **Read signals** from features.csv (day, symbol, entry_m, entry, stop, r_pct, n_prior)
2. **Fetch 1-min bars** from bars_sip.db (symbol, day, t in HH:MM; regular hours 09:30–15:59)
3. **Apply split tags** TRAIN (2025), VAL (2026-01–05), TEST (2026-06+)
4. **Score R basis** as ('meas', measured NBBO) and ('band', banded constant)
5. **Check day/week/trade context** against SPY_1min.parquet (D1), hmm_labels.csv (D4), and features.csv (D2, D3, D5)
6. **Report pass bar** via cadence_bar.py on both TRAIN/VAL splits

All data is PIT-clean (no look-ahead within bar t; levels computed from t−1 act from t+1).
