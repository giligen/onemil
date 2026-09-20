# PREREG — ORB wide seed (gap ≥ 3%, open $3–50), selection frozen then per-stratum refit
Written 2026-09-20 before any wide-seed number was read. Owner: "increase frequency; params/filters for
stocks above $30 and for 3% vs 5% gaps are likely different; cost is lower than modeled."

## Data
Features built by `research/orb_seed_wide/build_wide_features.py` (production builder, seed constants
patched: MIN_GAP_PCT 5→3, MAX_OPEN_PRICE 30→50; month-chunked) into `research/orb_seed_wide/out/`.
Production features CSV and cache.db never written. TEST ≥ 2026-06-01 is NOT built.

## Splits
TRAIN = 2025 (H1/H2 reported separately), VAL = 2026-01-01..2026-05-31, TEST sealed.

## Cost
Live-calibrated setting M (research/exec_cost/REPORT_RECAL.md): ORB entry 8.3 bps, exit unchanged.
Quoted (13.5 bps) reported beside it as the pessimistic bound.

## Strata (defined by the widening, fixed here)
S0 = the production seed (gap ≥ 5%, $3–30) — the reproduction gate: must equal the honest book to the cent.
S1 = gap 3–5%, $3–30.   S2 = gap ≥ 5%, $30–50.   S3 = gap 3–5%, $30–50 (reported, expected thin).

## Cells (count continues from 1,299)
1,300 whole wide seed, selection FROZEN (production z-params, cutoffs, vetoes, 8 slots, no refill).
1,301 S1 frozen, 1,302 S2 frozen — each vs its matched control (same day, same stratum, non-signal
      name walked under ORB's exit from 09:36 — the frames13 F41 construction).
1,303 S1 per-stratum REFIT (z-params + quintile cutoffs on 2025 inside S1, `scripts/orb_weekly_refit.py`
      method; mults never refit), scored on VAL.   1,304 S2 per-stratum refit.
1,305 S1 refit + per-stratum era-consistency vetoes (raw-feature buckets worst in BOTH 2025 halves).
1,306 S2 same.
1,307 combined book = S0 + best-passing form of S1 + S2, shared 8 slots (also 12 slots reported).
1,308 combined at quoted cost (pessimistic bound).
1,309 S3 frozen, report only.

## Pass bar (pre-committed)
Per stratum: 2026 VAL net ≥ +0.10 R with day-clustered t ≥ 2; stratum − control ≥ +0.10 R; both 2025 halves
same-signed; ex-top-5 % reported (diagnostic). Refit forms judged on VAL only; a refit that wins TRAIN and
loses VAL is dead. Combined book: cadence bar C1–C5 (docs/cadence_bar.md) on both splits, fills/week
≥ 3, stacked $ up both splits, MDD ≤ 1.25× the honest book's. Availability rail: ≥ 80 % of stratum picks
with bars; missingness gap ≤ 5 pp. Every ≥ +3 R trade obtainability-audited (C6).

## Not allowed
Buckets chosen on results; per-stratum exits; refitting adaptive mults; reading TEST; refill after a veto.
