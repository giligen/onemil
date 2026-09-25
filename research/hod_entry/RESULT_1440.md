# Cell 1,440 — stop distance

Base = 1,438 causal_arming_causal.csv fills (n=9911).

### Variant A (floor 0.8% only)

| holdout | n | base R | cell R | ΔR | t | ex-top-5% | changed% | stop-slip ΔR |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 4398 | -0.2084 | -0.2082 | +0.0001 | 1.01 | -0.3222 | 0.0% | +0.0003 |
| VAL | 5513 | -0.2245 | -0.2250 | -0.0005 | -0.99 | -0.3397 | 0.0% | -0.0006 |

### Variant B (floor 0.8% + cap 3%)

| holdout | n | base R | cell R | ΔR | t | ex-top-5% | changed% | stop-slip ΔR |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 4398 | -0.2084 | -0.2140 | -0.0056 | -2.40 | -0.3283 | 9.4% | -0.0066 |
| VAL | 5513 | -0.2245 | -0.2315 | -0.0070 | -2.50 | -0.3465 | 10.5% | -0.0084 |

Variant A: FAIL (TRAIN ΔR +0.0001, VAL ΔR -0.0005, VAL t -0.99).
Variant B: FAIL (TRAIN ΔR -0.0056, VAL ΔR -0.0070, VAL t -2.50).

Caveats: variant A touches only 0.05%/0.02% of fills (floor almost never binds) — near a no-op by construction. Variant B touches ~9.4%/10.5% (the 3% cap). 0.0% of VAL variant-B recomputes are stopped inside the fill bar (conservative, minute-low check). Per the 20:15 amendment, a PASS here is a lift on the 1,438 base (raw R ~ 0), reported as a lift, never a book — ships to dry run only.
