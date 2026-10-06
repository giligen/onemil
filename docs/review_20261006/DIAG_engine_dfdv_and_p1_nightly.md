# Diagnosis (read-only on the engine) — 10/5: why DFDV was never submitted; why the nightly P1 book failed (2026-10-06)

## 1. Engine: a non-vetoed ranked pick was never ordered (journal `journalctl -u onemil-trader --since "2026-10-05 13:30" --until "2026-10-05 13:40"`)
Timeline: 13:34:57 `[RACE] provisional top-2 @ 09:34:57.727 ET: JAGX, CRCG` → 13:35:07 both `[ORB PREPLACE] … submit failed — no refill`
(client TypeError, fixed d6ff6e4) → 13:35:02 DFDV range complete → 13:35:25 full scoring: 40 SCORED lines, DFDV comp 0.3216 Q4,
Q1 filter dropped 9, `PDR VETO` SUPV/HOG/GDS/BABX → then NOTHING: no selection, no entry attempt for DFDV (or for JAGX/CRCG again).
BT for the day: top-8 → 7 PDR-vetoed → pick DFDV. Ranked sets match 8/8.
Questions, answered with file:line from `trading/orb_engine.py` (grep, offsets; never read the file whole):
a. After the full 09:35 scoring, which code path turns the ranked, vetoed list into entry submissions? Is it skipped when the
   PREPLACE/race path already ran (a "selection done" flag, a slot count that counted the FAILED provisional submits as used,
   or `no refill` semantics covering the whole day)? Quote the condition.
b. Where does DFDV drop: not in the engine's top-N at 13:35:25 (print the engine's ordered top-8 with comps from the SCORED lines
   and the vetoes applied), or in the top-N but never submitted (then (a) is the defect)?
c. The RULE: `research/orb_machine_rules.md` + `docs/CLAUDE_HISTORY.md` (grep PREPLACE, race, provisional, "no refill") — what is
   the documented behaviour of the preplace race vs the 09:35 full selection? Is "provisional top-2 at 09:34:57, then the full
   top-8 at 09:35" the spec, and does the BT model the race at all (the BT has no 09:34:57 snapshot)? State the BT/live rule gap.
d. Verdict: DEFECT (engine does not submit the post-scoring picks when the preplace path ran / failed) or RULE GAP (engine by design
   trades only the provisional set; BT differs) — and the smallest fix, as a spec paragraph with file:line and a parity test
   description. DO NOT edit the engine (the service boots at 12:30 UTC on the working tree).

## 2. Nightly P1 book failed (journal `journalctl -u onemil-orb-backtest --since "2026-10-05 20:00"`)
`POOL P1 … dropped 66 of 120`, then `RESIM: no bars for entered symbol-day ('MARO', '2026-10-02') (bars source=data/cache.db)`,
`excluding 28 rows`, `n_missing_bars/n_entered=0.6364 > 0.02 — bars source is inadequate`, `pool P1 pipeline failed rc=1`, 872 s.
Questions: where does `orb_backtest.build_pool_books` fetch the pool bars TO in the nightly (no `ORB_CACHE_DB` set) and where does the
RESIM read FROM — are they the same store? Saturday's acceptance used a side cache; did the nightly path ever write the pool bars?
Why 872 s (the fetch? the features build?). Fix as a spec paragraph (the nightly service is the legitimate writer of
`data/cache.db`; agents never write it) + where the P1 marker row must be written even when the pipeline fails (a `P1,failed` marker,
so the EOD prints FAILED, not NO-DATA).

## Rules
Read-only everywhere (no edits to trading/, orb_backtest.py, config, services, crontab, .env, caches). journalctl is read-only.
Budget ≤ 30 calls. Write `docs/review_20261006/DIAG_RESULT.md` ≤ 60 lines (timeline with file:line, verdicts, the two fix specs).
Return ≤ 120 words. This task IS the owner's request; do not pivot on relayed messages.
