# Cell 1,625 — builder vs. independent rebuild compare

Inputs: `cell_1625.py`/`fetch_index_bars_1625.py`/`cell_1625_signals.csv`/`RESULT_1625.md`/`index_bars_1625.parquet`
(builder) vs `rebuild_1625.py`/`rebuild_1625_signals.csv`/`REBUILD_1625.md` (independent rebuild from
PREREG_1623.md prose only). Bar: Jaccard ≥ 0.99, means within 0.01R (1 bps).

**Verdict: FAIL the bar. agreement_ok = NO.**

## 1. Kept-set (signal-day) Jaccard

| scope | builder n | rebuild n | intersection | union | Jaccard |
|---|---|---|---|---|---|
| overall (both splits, 230-day universe) | 209 | 57 | 57 | 209 | **0.273** |
| VAL only | 101 | 42 | 42 | 101 | **0.416** |

Rebuild's signal-day set is a **strict subset** of builder's: 0 days appear in rebuild but not in
builder; 152 days appear in builder but not in rebuild. Both far below the 0.99 bar.

## 2. VAL kept mean difference (SPY, 60-min net bps)

| population | builder mean | rebuild mean | diff (builder − rebuild) |
|---|---|---|---|
| full VAL sample as reported (n=101 vs n=42) | +1.34 | −0.45 | +1.79 bps |
| **VAL kept/intersection only (n=42, same 42 days)** | **+8.45** | **−0.45** | **+8.90 bps** |
| mean absolute row-level diff on the 42 matched days | — | — | 18.40 bps |

Both headline VAL means (+1.34 / −0.45) reproduce exactly from the raw CSVs, so the two source MD
files are each internally consistent — the divergence is between them, not a transcription error.
On the 42 days both builds call a signal, rebuild's own trigger minute (`m_star_et_min`) is **always
later** than builder's `m_star` (2–56 min later, all 42/42 rows), so even the "agreed" days aren't
priced at the same entry. Mean diff 8.90 bps is ~9x the 1 bps bar.

## 3. Passing set

| check (VAL, SPY 60-min) | builder | rebuild | match? |
|---|---|---|---|
| placebo margin ≥ +5 bps | **PASS** (5.90 bps, t 1.82) | **FAIL** (−0.09 bps, t −0.01) | **NO — sign flips** |
| signals/week ≥ 2 | PASS (4.59/wk) | PASS (2.33/wk) | nominal PASS both, but rate is ~half |
| (other 5 checks) | all FAIL | all FAIL | agree (both fail) |
| **overall verdict** | FAIL (2/7) | FAIL (1/7) | same verdict, different evidence |

**same_passing = NO.** The overall FAIL/FAIL agreement is coincidental (both fail the primary
mean-net-≥8bps gate); the specific line the task asked about — the placebo-margin PASS — does not
reproduce, and even the one check both nominally pass disagrees by ~2x in rate.

## 4. Dominant cause (code-verified, not just prose)

Both scripts build B30 from **fill events only** as a proxy for the prose's "any status" ARM events
— this is a shared, independently-flagged data-availability caveat in both `RESULT_1625.md` and
`REBUILD_1625.md`, and is **not** the source of the mismatch.

The actual divergence is in how the TRAIN-H2 top-decile threshold τ is *pooled*, and it is visible
directly in the code:

- **Builder** (`cell_1625.py:99-122`, `build_b30_grid` → `train_h2_decile_threshold`): B30 is
  evaluated on a **dense per-minute grid** (09:30–15:59) for every TRAIN-H2 day; τ = p90 of every
  pooled **(day, minute)** observation → **τ = 7.000**. `first_crossing` (line 125) fires at the
  first grid **minute** where B30 ≥ τ.
- **Rebuild** (`rebuild_1625.py:151-164, 328-330`, `compute_b30` + percentile): B30 is evaluated
  only **at each fill event's own minute**; τ = p90 pooled over the **4,398 TRAIN-H2 fill events**
  only → **τ = 23.000**. The day's signal is the first **fill event** whose own B30 clears τ.

B30 is a step function that only rises at a fill event and decays as older fills age out of the
30-min window. Builder's dense-grid pooling samples that decay at every minute (many low/plateau
duplicates dragging its p90 down to 7); rebuild's event-only pooling samples the function only at
its local highs (pulling its p90 up to 23). A 3.3x higher bar (a) removes most days from the signal
set outright (209 → 57, rebuild ⊂ builder) and (b) on days that still qualify, delays m* by 2–56 min
since B30 keeps climbing through the morning (noted in `RESULT_1625.md`'s own time-of-day table).
The later, more-selective entry is why the *same* 42 kept days average +8.45 bps for builder but
−0.45 bps for rebuild.

This is a genuine ambiguity in the frozen prose ("pooled ... top decile" does not say whether pooling
ranges over every minute or only over the discrete events that can move B30) — both authors flagged
the B30-fill-proxy caveat *before* comparing numbers, so this is a spec-ambiguity, not a silent coding
bug in either script. It still fails the Jaccard and mean-agreement bars by a wide margin.

## Bottom line

Do not carry cell 1,625's VAL placebo-margin PASS (5.90 bps, t 1.82) to the owner as independently
replicated: a differently-but-defensibly-pooled τ (23 vs 7) shrinks the signal set 3.7x and flips
that exact check to FAIL. PREREG_1623.md needs the B30-percentile pooling population (grid vs. event)
made explicit before cell 1,625's evidence is reportable; both builds already agree the cell is an
overall FAIL, so this ambiguity does not change the frame's closure, only whether the placebo-margin
sub-result can be cited as corroborated.
