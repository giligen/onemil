# FREEZE — TEST is sealed

TEST = 2026-06-01 .. 2026-09-11. No script in this directory reads a TEST-dated bar unless `--test`
is passed AND this file exists. This file exists as the marker; `--test` was NOT passed in the run
that produced REPORT.md unless REPORT.md says so explicitly and names the single cell that earned it.

Status at write time: **TEST NEVER OPENED** in this pass (F16 / F18 / F17, pass 5).
