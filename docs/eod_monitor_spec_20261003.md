# EOD monitor extension — sleeve, ORB paper parity, PROMOTION verdicts (spec, 2026-10-03, owner's ask)

Owner: "daily AI agent monitors to check the EOD results, share daily P&L, alignment with BT … and recommend when to move
forward with paper ⇒ live and the live $ ramp". Principle unchanged (owner 9/21): the NUMBERS and the VERDICT are deterministic
code; the LLM step in `scripts/eod_report.py` only phrases the assembled text. Nothing here places orders or changes config.

## 1. New module `scripts/eod_sections.py` (pure functions, each returns a list of text lines; unit-tested on fixtures)

### 1a. `sleeve_section(date)` — momentum sleeve (paper PA3NDODOGPC2)
Inputs: `logs/momentum_sleeve_ledger.csv` (fills, `slip_bps_vs_open`, `size_pct`, `client_order_id`), the state file the
script keeps (positions, equity marks, `last_rebalance`), `logs/momentum_sleeve_shadow_gate.csv`, `logs/momentum_sleeve.log`
(the COMPLETENESS and `MOM REFUSED` lines), and the broker (positions on the MOM account — read-only, via the existing client
factory in `scripts/momentum_sleeve.py`). Lines:
- `MOM P&L: day $x | week $x | since 2026-09-29 $x (equity $x, start $20,000) | DD from peak x %`
- `MOM rotation <Monday>: picks n/20 = BT top-20 (diff: +A −B …) | fills n/n | slip mean x bp (max y) | size 100%|50% | gate pNN`
  "BT top-20" = an INDEPENDENT recomputation for that Friday by `trading.momentum_sleeve.eligible_universe` + ranking on the
  cached panel (`cache_file(asof)`), guard on — the same functions the recon fixtures test; the comparison is name by name.
- `MOM reconcile: broker n names $x vs state n names $x | OK|MISMATCH …` and `MOM completeness: LOST x %, liquid x % | OK|REFUSED`.
On a non-rotation day only the P&L and reconcile lines. Every missing input → an explicit `NO-DATA (<why>)`, never a blank.

### 1b. `orb_paper_parity_section(date)` — ORB paper (PA3YNVRTKFMG) vs the nightly BT book
Inputs: `data/trades.db` rows with strategy orb for the date (paper), the engine's decision log lines (`journalctl` is NOT
read here — use the day's archive under `logs/session_archive/` and `logs/` the way `scripts/hod_dry_ledger.py` reads them),
and the nightly BT book for the date (find the producer: grep `research/orb_freq/` and the crontab for the nightly ORB BT CSV;
read it through `trading/orb_csv.read_orb_csv`). Lines:
- `ORB picks: engine n vs BT n | match n | engine-only: … | BT-only: …` (symbol set at the 09:35 decision)
- `ORB fills: n/n picks filled | entry diff vs BT entry: mean x bp (max y) | tilt mults engine vs BT: n/n equal | add-on events n (BT n)`
- `ORB P&L: day $x on n exits | BT book $x` and `ORB defects: Engine tick TIMEOUT n | GAP_GATE WARN n | ERROR n`
If the engine made no 09:35 decision: `ORB picks: NO DECISION (reason from the log)`.

### 1c. `promotion_section(date)` — the pre-committed gates (verdicts are code)
State file `logs/promotion_state.json` (clean-session counters, dates, last verdicts), updated once per trading day.
- **Sleeve paper → live**: counts CONSECUTIVE clean rotations: picks = BT top-20, fills n/n, slip mean ≤ 20 bp, reconcile OK,
  completeness OK, no `REFUSED`. Verdict `GO LIVE $20K on <next Monday>` when the count ≥ 2 AND the review fixes are
  committed (hard-coded flag `REVIEW_20261003_CLOSED` true once docs/review_20261003/FIX_*_result.md exist for A and C),
  else `HOLD n/2 clean (<first failing condition>)`.
- **ORB paper → live**: CONSECUTIVE sessions with picks = BT, fills within 30 bp of the BT entry on average, zero Engine tick
  TIMEOUT, zero ERROR; `GO $10K stage ($375 R) on <date>` when ≥ 5, else `HOLD n/5 (<reason>)`. A session with NO DECISION
  resets the count.
- **Live ramp (both books)**: reuse `scripts/orb_ramp_check.py` / the sleeve ledger: `+$10K` is recommended only when
  realized P&L since the last step ≥ 0 and ≥ 20 trading days have passed; otherwise `HOLD at $x`.
- **HOD**: fixed line `HOD: dry-run only — no live gate (closed as a money book 9/26)`.
Each verdict carries the rule in brackets so the owner sees WHY, e.g. `[2 clean rotations, slip ≤ 20 bp, reconcile OK]`.

## 2. Wiring
`scripts/eod_report.py` gains three sections `MOM`, `ORB PAPER PARITY`, `PROMOTION` after `RAMP`; the Telegram text includes
the PROMOTION lines verbatim (the Haiku rewrite must not touch them — pass them through). Any section that raises is caught and
printed as `<SECTION>: FAILED (<exc>)` — never a silent blank. The report keeps writing `logs/eod/<date>.md`.

## 3. Tests and proof
Unit tests on fixtures for every function (`tests/test_eod_sections.py`): a clean rotation → GO after two; a slip of 25 bp →
HOLD with that reason; NO DECISION resets ORB's count; missing inputs → NO-DATA lines. Then a REAL run for 2026-10-02:
`cd /home/ec2-user/onemil && bash scripts/research_run.sh -m 1500M python3 scripts/eod_report.py --date 2026-10-02 --no-telegram`
(add `--no-telegram` if absent; read the script's args first) → paste the three new sections into `docs/eod_monitor_result_20261003.md`.
Expected on 10/2: MOM P&L lines with the 9/29–10/2 ledger, rotation = none; ORB `NO DECISION`; PROMOTION all HOLD with reasons.
Never send a Telegram from the test run, never place orders, never git, never touch config/.env/crontab.
