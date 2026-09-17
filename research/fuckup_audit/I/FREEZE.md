# Stage I — frozen stack (written before TEST was read)

stack = S0 (F6)
exit = hold
rule = highest VAL weekly R over the 8 declared cells (PREREG.md Part A)
VAL: n 519, 23.6 tr/wk, mean net R +0.251, t 3.28, weekly R +5.91, weeks green 0.82

TEST will be read once, for this cell and for S0 hold.

## Amendment, written 2026-09-17 BEFORE the TEST run (i_test.py had not been executed)

The declared freeze rule (highest VAL weekly R) selected **S0 = F6 alone**, i.e. the REFERENCE, not a stack.
S0 hold's TEST numbers are already public in `H/F6/f6_pdr_book.md` (n 376, +0.104 R, t 0.74, 43% weeks green),
so the stage's TEST allowance would otherwise be spent on a cell that costs nothing.

Amendment: the one TEST read of this stage covers **S0, S1, S2 and S3 on the `hold` exit — four cells,
DESCRIPTIVE ONLY**. They cannot and did not influence the freeze (the freeze is written above and the TEST
script is run after this file is closed). They exist so the owner's question — "does stacking help?" — has an
out-of-sample answer rather than a TRAIN/VAL-only one. The 2R exit is NOT read on TEST.

Cells added by this amendment: 3 (S1/S2/S3 hold on TEST). Counted in the report's denominator.
