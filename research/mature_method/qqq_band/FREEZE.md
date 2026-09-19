# FREEZE — what was read, and when (QQQ noise-band sleeve, candidate #2)

PREREG §2 split: TRAIN = IS 2016-01-04→2023-12-31 · VAL = 2024-01-01→2025-12-31 ·
**TEST = 2026-01-01→2026-09-15 (end of `etf_1min.db`)**.

## 1. TEST was opened ONCE, for ONE cell, under the pre-committed rule

PREREG §5: *"TEST is opened once, only for cells passing G1 ∧ G2."*

- G1 (TRAIN net mean > 0, t ≥ 2): **15 of 18 cells pass**.
- G2 (VAL same sign AND ≥ 55 % green weeks): **exactly one cell passes — `H2` (no stop, hold to
  the flat), VAL 55.24 % green weeks**. Every other cell is 39–53 %.
- TEST was therefore read for **`H2` only**: `supp.log` §"TEST". Result reported in `REPORT.md` §9
  whatever it says.

## 2. Disclosures — places where sealed-window numbers exist or were printed

Recorded here rather than repaired after the fact.

1. **TEST is not virgin for the BASE cell and never was.** The July program and `Q/REPORT.md`
   treated OOS as one window 2024-01→2026-09 and published a 2026-YTD row for the base cell
   (`Q/step4_years.csv`: +3.45 bps/traded day, t 0.64, at 0.5 bp/leg). This was stated in PREREG §2
   before scoring.
2. **`cells.csv` on disk contains TEST rows for all 18 cells.** `score.py` computed every split in
   one pass; the printed table in `score.log` is filtered to TRAIN/VAL, and only `H2`'s TEST row was
   read and is quoted. This is a design defect of the script, disclosed, not a second TEST read.
3. **The additivity stack necessarily spans the sealed window.** PREREG §6.4 declared the additivity
   measurement against the live ORB and BF weekly paths, which exist only from 2025-01. The overlap
   used is 2025-01-02→2026-09-15, i.e. VAL + TEST. It is DESCRIPTIVE (no cell was selected with it)
   and the per-cell weekly $ it prints for B0/H2/T2 inside 2026 are a consequence. Declared before
   scoring; named again here.
4. `score.log` prints a B0 weekly path for 2026-04→09 under "WEEKLY PATH (last two quarters)" —
   runbook step 8's requirement, and inside the §6.4 descriptive window. Not used to select anything.

## 3. What is still unread

Nothing inside `etf_1min.db` (2016-01-04 → 2026-09-15) for the base cell. The only genuinely unread
data for this sleeve is **forward**: sessions after 2026-09-15. That is the honest reason a dry run
is the only remaining instrument, and the reason `REPORT.md` §11 prices how long one would take.
