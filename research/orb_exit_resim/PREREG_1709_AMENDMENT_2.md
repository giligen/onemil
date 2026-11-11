# PREREG 1,709 — Amendment 2 (2026-11-11): the ORB paper gate restarts on the parity-fixed engine

Owner 2026-11-11: "should we restart the count of 10 days of REAL parity?" — YES.

## Why
Cells 1,732 / 1,733 (`research/book_review_20261111/`): the paper book 9/29–11/10 shared only 21 of 32 fills with
the production BT. The BT re-scored history with today's weekly-refit parameters (look-ahead; fixed eafb557), the
live engine and the BT builder applied different universe gates (fixed 2a55c82 / ff2a6fa), and the BT's 15K
"9:35 volume" cache proxy dropped names live traded (fix queued). The 32 symbol-days counted so far were therefore
traded by a book that was not the rule's book; a gate read on them would judge the wrong thing. The segment stays
in the record as "pre-parity" (reported, never gated). The book is $0; the cost of restarting is time only.

## Rule (frozen; no number below changes without a new amendment)
1. **Pick-parity precondition.** From the first boot that loads ff2a6fa (2026-11-12 12:30 UTC): 10 CONSECUTIVE
   sessions on which the live final top-8 equals the point-in-time BT top-8, every difference carries a named
   reason (`scripts/orb_pick_parity.py`, spec `docs/orb_selection_parity_test_spec_20261111.md`), and the day has
   zero `unexplained` and no live fill outside the final set. One unexplained day resets the 10. Sessions with no
   ORB range (vendor failure, `[ORB] post-open range sweep … vendor bars late` ERROR) neither count nor reset.
2. **P&L gate.** The Amendment-1 conditions, counted on POST-11/12 paper symbol-days only (accidents of the CRDU/BEX
   class excluded, each named in the ramp read): at 40 symbol-days, realized > 0 AND above water ex-monster AND
   ≥ 15 sessions → live at the $375 stage under the auto-pause tripwire; else ORB is shelved for good.
   `scripts/orb_ramp_check.py --stage-start 2026-11-12` is the reader until the checker gets an Amendment mode.
3. Precondition 1 must hold at the time of the gate read; a gate read before it holds is void.
4. Nothing else changes: no knobs, no refits of `adaptive_mults`, the stop rule stays (cell 1,732: at parity).

Expected resolution: ~8 weeks at ~5 symbol-days/week → mid-January 2027; earlier only if frequency rises.
