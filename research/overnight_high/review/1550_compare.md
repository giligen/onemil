# Cell 1550/1551 builder vs. rebuild comparison

Compares `cell_1550_nights.csv` (builder) against `rebuild_1550_nights.csv` (independent
rebuild), keys = (sample, cell, date, symbol).

## Membership Jaccard (nightly book, per sample x cell)

| sample | cell | n_builder | n_rebuild | n_intersection | Jaccard |
|---|---|---|---|---|---|
| EXTENSION | 1550 | 9362 | 9364 | 9340 | 0.9951 |
| EXTENSION | 1551 | 17058 | 17071 | 17044 | 0.9976 |
| PANEL | 1550 | 2878 | 2879 | 2866 | 0.9914 |
| PANEL | 1551 | 6294 | 6299 | 6283 | 0.9957 |

All four groups agree on **>99% of nightly membership** — 22-27 rows differ per group out of
~2,900-17,100. On every (date,symbol) key present in BOTH files, `ret_on_next` is bit-identical
(max abs diff = 0.0 across 9,340 shared EXTENSION/1550 rows) — the return computation itself is
not in dispute, only which nights are in the book.

## Per-sample net-bps difference (builder net_bps_5 vs rebuild net_bps_5bp)

| sample | period | builder net_bps | rebuild net_bps | abs diff |
|---|---|---|---|---|
| EXTENSION | FULL (1550) | 4.75 | 17.22 | 12.47 |
| EXTENSION | FULL-2020 (1550) | 46.43 | 64.78 | 18.36 |
| EXTENSION | **FULL-2021 (1550)** | -8.11 | 22.40 | **30.51** |
| EXTENSION | FULL-2022..2023 | ~match | ~match | <1 |
| EXTENSION | FULL-2024 (1550) | 3.48 | 5.79 | 2.30 |
| PANEL | TRAIN (1550) | 9.85 | 13.04 | 3.19 |
| PANEL | **VAL (1550)** | 13.66 | 57.51 | **43.85** |
| PANEL | TEST (1550) | -2.01 | -20.72 | 18.71 |

**Max abs net-bps diff = 43.85 bps, on PANEL/VAL/cell 1550** — the exact holdout period the
1550/1551 pass bar is scored on. Builder says VAL is +13.7 bps; rebuild says +57.5 bps. This is
not a rounding difference: it changes the VAL read materially and must be resolved before either
number goes to the owner.

## Cause of the membership difference (verified, not inferred)

The ~22-27 differing rows per group are **not random** — they are concentrated in exactly the
kind of names that stress a rolling-window warm-up rule: illiquid/newly-volatile tickers around
2020-2021 COVID/meme squeezes and the 2024/2026 GME reruns. Rows present ONLY in the rebuild
carry outsized single-night returns the builder never sees at all (VIRX +215%, GME +113%
[2024-05-13, the Roaring-Kitty return], USFR +100%, CODX +98%, SPRT +95%, NURO +85%, OCGN +77%,
APT +73%/+69%, BBIG +72%, VXRT +84%/+54%, NVAX +59%) — builder's max is +49% and min -45% with
zero rows beyond 0.5 abs return; rebuild has 24 rows beyond 0.5 (up to 2.15). Because the book is
tail-heavy by construction (a handful of nights carry the mean), a ~0.3% membership swap that
happens to add/drop these specific mega-return nights swings net-bps by tens of bps even though
the Jaccard is >0.99.

Traced to source (`build_panel.py` vs `rebuild_1550.py`, both otherwise identical: same
`HIGH_WINDOW/HIGH_MINP = 252/252`, same `VOL_MULT = 1.5`, same underlying parquet files for both
EXTENSION and PANEL):

```
build_panel.py:44   ADV_MIN_PERIODS = 10   # adv20 = shift(1).rolling(20, min_periods=10)
rebuild_1550.py:49  ADV_WINDOW, ADV_MINP = 20, 20   # adv20 = shift(1).rolling(20, min_periods=20)
```

**Dominant cause: ADV window convention.** The builder accepts an ADV20 average from as few as
10 valid prior trading days (tolerating short warm-ups / occasional missing bars inside the
trailing 20-session window); the rebuild requires the full 20 valid sessions or the row is NaN
and the vol_ratio signal cannot fire. For names with a data gap or short pre-signal history
inside that 20-day window — exactly the volatile/thin names in the diff list above — this flips
whether `vol_ratio >= 1.5` evaluates at all, which flips whether the 252-day-high row enters the
book on that date. High252 convention, universe/dollar-volume filters, test-ticker exclusion and
missing-bar handling are otherwise identical between the two implementations and were ruled out
by direct code diff.

## Verdict

Membership agreement is high (Jaccard 0.991-0.998) but **not tail-safe**: the disagreeing rows
are disproportionately the extreme-return nights that dominate a mean-based net-bps stat, so the
PANEL/VAL net-bps figure — the number the 1550/1551 pass/fail call rests on — differs by 43.85
bps between builder and rebuild (+13.7 vs +57.5). Recommend re-running both under a single,
agreed ADV_MIN_PERIODS convention (20, matching the rebuild's stricter and more defensible
choice, since a 10-day-old average is not really "ADV20") before either number is reported to the
owner as the TEST-disclosed VAL read for 1550/1551.
