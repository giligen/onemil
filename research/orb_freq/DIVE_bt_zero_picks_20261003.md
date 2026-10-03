# Deep dive — why does the nightly ORB BT book show ZERO picks on 9/24, 9/26, 9/29 and 10/2? (2026-10-03, owner's ask)

The parity read Monday compares the paper engine's picks with `analysis_results/orb_bplus_book.csv`. If the book's zeros are a
BT defect, every parity read is wrong and the ORB promotion counter can never clear. If they are genuine (no candidate
survived selection), fine — but that must be shown stage by stage, not assumed.

## Questions (answer each with numbers, in this order)
1. Producer: `systemctl cat onemil-orb-backtest.service` / `.timer` (read-only), the exact ExecStart, and its journal for the
   last 8 runs (`journalctl -u onemil-orb-backtest --since 2026-09-23 --no-pager`): start/finish times, any WARNING/ERROR,
   any "fallback", "missing bars", "cache", "0 rows" line. Did every run finish? Does it run BEFORE the day's SIP bars are
   complete (bars_sip.db / cache.db write times vs the run time)?
2. For each session 2026-09-22 → 2026-10-02: rows in the latest features CSV (`research/orb_freq/` or `analysis_results/`
   `orb_features_YYYYMMDD_HHMM.csv`, the newest), how many pass each selection stage of `study_orb_pipeline_static_lock.py`
   (read its stages: catalyst off, skip_q1, spread gate ≥ 150 bps, adaptive_mults / Q buckets, slot cap 8, post-ranking vetoes),
   and the number of picks in the book. A table: date | features rows | after each stage … | book picks.
3. Compare with reality on the same days: the engine's `ORB SCORED` / picks lines in `logs/session_archive/` (9/22 dry day,
   9/30 and 10/1 paper sessions had picks; 10/2 had no decision) and `data/trades.db` orb rows. Where the engine PICKED and
   the book has ZERO, explain the stage that differs — that is the defect candidate.
4. Price/volume inputs: for 3 names the engine picked on 9/30 or 10/1, print the BT's feature row (gap %, 09:35 RVOL, spread,
   price, prev volume) beside the engine's logged values. A systematic difference (e.g. gap computed from a stale prev close,
   RVOL from an incomplete 09:35 bar, spread from a wrong quote source) is the finding.
5. Verdict: GENUINE ZEROS (every stage accounted for, inputs agree) or BT DEFECT (name it, file:line, one-line fix). If a
   defect: do NOT fix the producer; write the fix as a spec paragraph. The parity reader (`scripts/eod_sections.py` ORB loader)
   must then treat those days as NO-DATA, not zero — say so.

## Rules
Read-only on data, config, services, crontab, .env. Run analysis only through `bash scripts/research_run.sh -m 2500M python3 …`
(service is down — Saturday). Never run the nightly BT itself if it takes > 5 min; a single-day re-run of the static-lock
pipeline on the existing features CSV is fine. Never pipe long commands through tail/head/grep. Write
`research/orb_freq/DIVE_bt_zero_picks_RESULT.md` ≤ 60 lines with the table, the 3-name comparison and the verdict. No git.
Return ≤ 150 words: verdict, the stage that empties the book, the input comparison, anything you could not do.
