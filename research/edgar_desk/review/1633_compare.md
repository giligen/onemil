# Compare — cell_1633_events_pead.csv (builder) vs rebuild_1633_events_pead.csv (rebuild)

Independent-check comparison for PREREG_1633.md's cells 1,633-1,635 (idea 6, `research/IDEAS_20260928.md`).
Builder = `research/edgar_desk/cell_1633.py` -> `cell_1633_events_pead.csv` (131,366 rows, TRAIN/VAL only).
Rebuild = `research/edgar_desk/rebuild_1633.py` -> `rebuild_1633_events_pead.csv` (55,889 rows, TRAIN/VAL only).
Both files confirmed TEST-free (0 TEST rows either side) — TEST was never computed for this comparison, per SEALED.
Builder's per-cell rows carry decile/hold pre-selected (cell `1633`=decile10/h10, `1634`=decile10/h20,
`1635`=decile1/h10 short-eligible-filtered already, `1636_h10`/`1636_h20`=full report-only decile table,
all deciles); rebuild is one row per event across all deciles with both hold outcomes as columns, so cell subsets
were reconstructed from it (decile==10 for 1633/1634; decile==1 & shortable_flag==True for 1635, rebuild's own
"FILTERED" definition) before keying both sides on (symbol, ann_date/filing_date).

## Event-set Jaccard (keyed by symbol + ann_date + cell)

| Scope | n builder | n rebuild | n intersect | n union | Jaccard |
|---|---:|---:|---:|---:|---:|
| Cell 1633 (long top decile, h10) | 6,229 | 5,979 | 5,783 | 6,425 | **0.900** |
| Cell 1634 (long top decile, h20) | 6,222 | 5,979 | 5,776 | 6,425 | **0.899** |
| Cell 1635 (short bottom decile, filtered) | 1,313 | 4,881 | 1,083 | 5,111 | **0.212** |
| **Combined, 3 cells** (symbol,ann_date,cell) | 13,764 | 16,839 | 12,642 | 17,961 | **0.704** |
| Population (all deciles, hold=10 side, diagnostic) | 58,466 | 55,882 | 53,482 | 60,866 | 0.879 |

None of these clear the PREREG's independent-check bar (event set Jaccard ≥ 0.98). The gap is real and is
explained below — it is a population-membership disagreement, not a disagreement in the return computation
itself (see the bps section: shared events agree almost exactly).

## VAL mean bps difference, cell 1,633

| | builder | rebuild | diff |
|---|---:|---:|---:|
| Headline VAL mean net bps (each side's own event set, n_b=2,093 / n_r=2,025) | 9.11 | 9.03 | **0.07 bps** |
| Matched-pair only (n=1,971 events both sides include) | mean(b−r) = 0.143 bps, median\|diff\| = 0.000 bps, max\|diff\| = 281.9 bps (one event) |

The headline diff (0.07 bps) is inside the PREREG's "bps within 2" bar, and the matched-pair diff confirms this
isn't cancellation: on events both pipelines actually include, the return computation itself agrees almost
exactly (median difference is zero). One matched pair differs by 281.9 bps and is worth a manual look, but it
does not move either side's mean. **Reaction-session mapping agreement on matched pairs: 1,971/1,971 = 100%
identical** — see "ruled out" below.

## Passing cells

Builder: **[]** (per RESULT_1633.md, cells 1633/1634/1635 all FAIL the frozen VAL pass bar on every required
criterion — 1634 passes 2 of 6 individual line items (mean_net, TRAIN-same-sign-t) but not all 6, so its overall
verdict is FAIL like the others).
Rebuild: **[]** (per REBUILD_1633.md — same three cells, same FAIL verdict on every cell: 1633 fails all 6
criteria, 1634 "still short of any of the frozen bars" despite a directionally better VAL t=1.10, 1635 is
negative on VAL once shortable-filtered).
**Match: YES.** Both sides independently reach an empty passing set — the headline conclusion (PEAD does not
clear the pre-registered bar on this population) is robust to the event-set disagreement documented above.

## Dominant cause of differences

Checked all four candidates named in the task, using the two scripts' own code plus the data:

1. **Universe-filter "prior session" convention — PRIMARY cause of the population-level gap (not UTC/ET).**
   Builder's `cell_1633.py` (docstring + code, lines ~48-56, 280-283) reads the $3/$1M universe filter
   *uniformly at the session strictly before date_et*, even for after-close filings, by design ("a conservative,
   always-causal choice... it never uses same-day information, even for after-close filings where date_et's own
   close would also be causally valid"). Rebuild's `rebuild_1633.py` (lines 289-330) instead keys `prior_close`/
   `dvol20` off the bucket-dependent `prior_session` column, which for an after-close filing equals *date_et's
   own session* (same day as the filing) — confirmed directly from row 2 of the rebuild CSV (WSBF,
   acceptance_et 2019-01-30 16:01:25, bucket=afterclose, prior_session=2019-01-30=filing_date). This is
   **exactly the ambiguity RESULT_1633.md's own independent-check flag #2 pre-registered as the next thing to
   test** ("'Prior session' for the universe filter was read literally as the session strictly before the
   announcement's own calendar day... An equally defensible reading uses date_et's own close for after-close
   filings — rebuild independently and compare"). Since after-close filings are a large share of item-2.02
   releases and price/dollar-volume can cross the $3/$1M screens from one session to the next around an
   earnings release, this one convention plausibly accounts for most of the population Jaccard's shortfall
   (58,466 vs 55,882, symmetric difference 7,384 / union 60,866). This is a **documented, PREREG-anticipated
   methodology choice, not a coding bug** — both readings are defensible per the PREREG prose, and PREREG_1633.md
   itself did not pin down which one to use.

2. **Reaction-session mapping / UTC vs ET — RULED OUT.** Both scripts convert `acceptance_datetime` (UTC) to
   `America/New_York` via the same mechanism (`cell_1633.py`: `zoneinfo.ZoneInfo("America/New_York")` +
   `tz_convert`; `rebuild_1633.py`: `pd.to_datetime(..., utc=True)` + `.dt.tz_convert("America/New_York")`) —
   i.e. both took the "convert properly" reading the PREREG demanded and that RESULT_1633.md flagged as
   disagreeing with the *older* cell_1552 convention (irrelevant here, since builder and rebuild agree with
   each other). Confirmed empirically: 0/1,971 matched-pair events disagree on `reaction_session` (100% exact
   agreement) — this candidate contributes zero to the observed gap.

3. **Decile edges — largely ruled out.** Empirical TRAIN cutoffs (from each side's own full-population rows)
   agree closely despite the differing TRAIN population sizes: decile-10 lower edge builder=0.0847 vs
   rebuild=0.0849 (rebuild's stated cutoff); decile-1 upper edge builder=-0.0836 vs rebuild=-0.0837. The two
   pipelines converge on nearly identical cut points, so decile-edge disagreement is not a meaningful driver of
   which events land in the top/bottom decile.

4. **Duplicates — minor, secondary contributor.** Builder's population (`1636_h10`) has 326 exact-duplicate
   (symbol, ann_date) rows (0.55%); rebuild's has 7 (0.01%). `rebuild_1633.py` explicitly dedupes
   (`drop_duplicates` on exact (cik, acceptance_datetime, items), then collapses same-symbol/same-ET-date to
   the earliest acceptance); the corresponding drop_duplicates call is present in `cell_1633.py` only for the
   *price panel*, not for events — i.e. builder does not dedupe same-day repeat 8-K/2.02 filings. Real, but at
   326 rows this explains under 5% of the 7,384-row population symmetric difference — not the dominant cause.

5. **Cell-1635-specific cause (separate from 1-4, explains its 0.212 Jaccard): SSR-proxy applied on one side
   only.** Builder's cell-1635 short-eligible filter combines the borrow_flags.csv lookup with an SSR proxy
   (prior_close < $5 OR R0 <= -10%, per RESULT_1633.md's own text). Rebuild's "FILTERED" cell-1635 definition
   uses `shortable_flag==True` only — REBUILD_1633.md's own caveats say so explicitly ("SSR is not represented
   at all in borrow_flags.csv... No SSR exclusion is applied anywhere"). Direct test: of the 3,799 events
   rebuild counts as shortable-bottom-decile that builder's cell 1635 excludes, **98.8% would be excluded by
   builder's SSR proxy** (97.5% via R0 <= -10% alone — mechanically expected, since bottom-decile events are
   R0 <= -10% almost by construction). This single, already-documented filter difference accounts for
   essentially all of cell 1635's gap; it is not a bug on either side, but the two CSVs are answering a
   slightly different question for this cell (short-eligible-by-borrow vs short-eligible-by-borrow-and-SSR).

## Bottom line

The two pipelines **compute the reaction and the forward return the same way** (0.07 bps headline / ~0.14 bps
matched-pair mean difference on VAL cell 1633, 100% reaction-session agreement, near-identical decile cutoffs),
so this is not a coding-error disagreement. They **disagree on population membership**, primarily through one
PREREG-anticipated ambiguity (the universe filter's "prior session" convention for after-close filings — flag
#2 in RESULT_1633.md) plus, for cell 1635 only, an already-documented SSR-proxy gap. Neither difference changes
the substantive conclusion: **both sides independently return an empty passing-cell set** — PEAD on the R0-decile
population does not clear the frozen VAL bar in this rebuild, matching the builder's FAIL verdict on 1633/1634/1635.
The event-set Jaccard bar (≥0.98) is not met and should not be waived silently; before this result is shown to the
owner, PREREG_1633's flag #2 (prior-session convention for after-close filings) should be resolved to one rule and
re-run, since it is the one open, unresolved methodology choice driving the gap.
