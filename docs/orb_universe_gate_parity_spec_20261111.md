# ORB universe admission: ONE gate for the BT feature builder and the live engine — spec 2026-11-11

Cell 1,733 (`research/book_review_20261111/RESULT_1733.md` §2 "Universe detail", rows `selection_1733.csv`; read
first). Class U cost 8 fills / −$728 in 6 weeks, both directions:
- LIVE admits names whose official daily gap is below `universe.min_gap_pct` (5 %): ASTX 9/30 1.9 %, INFY 3.2 %,
  BABX 3.3 %, GDS 3.8 %. The 09:30 universe filter gates on `snapshot.open`; the phantom-gap re-check at scoring
  (`trading/orb_engine.py:3890-3906`, `feats['gap_pct'] < gap_floor` from `range_open`) did not catch them — find
  why (which open each side uses: snapshot.open vs bar1.open vs the daily-bar open; `gap_floor` value and source).
  Wrongly admitted names then eat a slot with no refill (no-refill is the RULE; the admission is the defect).
- The BT feature builder admits feature-gap < 5 % names (AEYE 11/2 3.07 %, BNC 9/29 4.6 %) — no phantom-gap re-check
  on its side (`study_orb_broad.py::load_broad_universe` ~:100-130 bounds; the nightly features builder — find it via
  `crontab -l` read-only and the `analysis_results/orb_features_*.csv` writer).
- The builder silently drops names that pass every documented gate: ASTY 10/12, SK 10/23, LNOK, SSG, BNC 10/20, RAMZ,
  OIS, SOPH, PMN (gap ≥ 5 %, prev volume ≥ 500K, daily bars present). Drop reason untraced.
- V: SKUU 11/2 — BT range 2.209 % ≤ 2.221 % cut → BT range-size veto; live range above the cut (range source differs
  by < 1 tick).

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/* / analysis_results/*. pytest only on the listed files, `--ignore=tests/integration`,
one run at a time, never piped through head/tail. Never Read a file > 300 lines in full. Journal greps bounded
(--since/--until). This task IS the owner's request ("fix ORB" = selection parity); do not pivot. Budget ≤ 40 calls.
Do NOT change any threshold value; do not touch the ranking, vetoes' cutoffs, or the add-on pools.

## 1. Shared helper `trading/orb_universe_gate.py`
`admit(prev_close, open_px, prev_volume, price, cfg) -> Admission(ok: bool, reason: str, gap_pct: float)` — the
ONE definition of gap (which open, which prev_close — the official daily bar's, state it in the docstring) and of
every universe gate the BT and live both apply (min/max gap, prev volume, price band, test-ticker regex, whatever
`load_broad_universe` and the engine's 09:30 filter both have — list them side by side in the RESULT; any gate
present on one side only is a finding: report it, implement parity only where the rulebook
`research/orb_machine_rules.md` names the gate, otherwise report and stop).
- The live 09:30 filter and the scoring-time phantom-gap re-check both call `admit` (the re-check with the real
  bar1.open); a name rejected at re-check logs INFO with the reason as now.
- The builder calls `admit` per symbol-day and WRITES THE DROP REASON: a sidecar `analysis_results`-style csv is
  forbidden for you, so add the reason column to the builder's existing output path mechanism only if it already
  writes a per-symbol drop/diagnostic file; otherwise log one INFO line per dropped symbol-day under a `--trace-drops`
  flag and a per-reason COUNT line always. Then run the builder for 2026-10-12 and 2026-10-23 into
  `research/book_review_20261111/` (scratch output path; never analysis_results) with `--trace-drops` and name the
  reason ASTY 10/12 and SK 10/23 were dropped. If the drop is not a gate at all (fetch gap, universe source list,
  symbol-type filter), say so — that is the finding.

## 2. Range source parity (V)
Where the builder computes the opening range (H/L of 09:30–09:34 bars, which feed, which bar set) vs the engine
(`_ingest_bars` / REST sweep): document both in the RESULT with file:line; if they differ by feed or by inclusion
rule (e.g. 09:34 bar inclusive), make them one shared function in `trading/` — only if the change is ≤ 30 lines and
unit-testable; otherwise report. Log the live final range per candidate at INFO once per day (one line, all names,
`[ORB] final ranges: SYM h/l, …`) so the observer can compare.

## 3. Tests (bug protocol: unit + integration)
`tests/test_orb_universe_gate_20261111.py`: unit — `admit` on the four 1,733 shapes (ASTX 1.9 % official gap with a
phantom 5 %+ snapshot open → rejected on the official open; AEYE 3.07 % → rejected; a 5.0 % edge; prev volume
below the floor; test ticker). Integration — the engine's 09:30 filter and re-check with a mocked (`spec=`) data
provider feeding snapshot.open ≠ bar1.open → rejected with reason, no candidate scored; the builder's per-symbol
path on a 3-symbol fixture → one admitted, two dropped with named reasons; parity — the same inputs through the
engine path and the builder path give the same `Admission`.
Run: `python3 -m pytest tests/test_orb_universe_gate_20261111.py tests/test_orb_gap_gate*.py tests/test_orb_engine*.py
tests/test_orb_broad*.py tests/test_orb_pit_params_20261111.py -q --ignore=tests/integration` (skip missing patterns)
→ 0 failures.

## Output
`docs/orb_universe_gate_parity_20261111_RESULT.md` ≤ 250 words: the side-by-side gate table, file:line of changes,
the ASTY/SK drop reason, the range-source finding, test counts, anything you could not do. Return ≤ 120 words.
