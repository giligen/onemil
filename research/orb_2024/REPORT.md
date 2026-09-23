# REPORT — cells 1,415-1,416, 2024H2 ORB holdout (PREREG.md)

## Cell 1415

- Cell 1415 (2024H2): n=59 fills/wk=2.57 net_R/fill=-0.007 t=-0.09 total=$-147 ex-top5%_R=-0.062 no-fill=39.2% best_mo=$353 worst_mo=$-630 -> neither

## Cell 1416

- Cell 1416 (2024H2): n=4 fills/wk=1.33 net_R/fill=+0.455 t=+1.63 total=$683 ex-top5%_R=+0.285 no-fill=63.6% best_mo=$683 worst_mo=$683 -> SURVIVES

## 2025 comparison (runB_true, production seed)

- runB_true 2025 (beside cell 1,415): n=127 fills/wk=2.44 net_R/fill=+0.272 t=+3.19 total=$12,959 ex-top5%_R=+0.118 no-fill=21.1% best_mo=$4,040 worst_mo=$-187 -> SURVIVES


## Main-session review (2026-09-23)
* Robustness: the agent found that `cache.db::daily_bars` holds only SPY for 2024-06, so July's 20-day lookbacks were
  degraded. August–December alone (complete features): n 55, −0.012 R/fill (t −0.15), −$247, ex-top-5 % −0.072.
  Same answer.
* Monthly $ (1,415): Jul +100, Aug −206, Sep −630, Oct +353, Nov +322, Dec −87. No month resembles 2025.
* Adequacy: SE ≈ 0.08 R/fill. The 2025 level (+0.272) is ~3.5 SE above the 2024H2 estimate — rejected. A modest
  edge (+0.10 R) is ~1.4 SE away — NOT excluded. Pre-registered verdict: neither SURVIVES nor RED FLAG.
* Cell 1,416 (addon_p30): n 4 (the PDR veto removes most $30–50 names) — no information.
* **Operating consequence:** the ORB edge measured in 2025–26 is at least partly that regime. Do not scale ORB risk on
  backtest numbers; the live ramp's above-water rule (advance only on realized stage profit) is now the binding
  evidence. A decisive test needs more out-of-regime history (EQUS daily 2023–2024H1, ~$10).
Programme count 1,416.
