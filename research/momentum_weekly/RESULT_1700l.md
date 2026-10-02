# RESULT 1,700l -- mid-week actions on the reconciled momentum sleeve (PREREG_1700l.md, frozen)

Daily sim 2017-02-06..2026-09-28 (9.64 y), $50K, 1700j engine/costs (band-based, not NBBO). REF reproduces 1700j (27.2%/-44.5%/$507,823). Paired = weekly return (Monday opens) minus REF; ex-top-5% drops the best 5% of weekly differences; halves 2017-21 / 2022-26 both must be > 0.

| cell | CAGR | max DD | CAGR/DD | end $ | Sharpe | worst yr | yrs>SPY | roll5y | turn/yr | cost/yr | ep1..5 depth | paired mean/wk (t) | ex-top5 | halves | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | 27.2% | -44.5% | 0.61 | 507,823 | 0.85 | -7.0% (2018) | 6/10 | 100% | 11.7x | 3.27% | -45% -37% -34% -29% -29% | +0.000% (nan) | +0.000% | +0.000/+0.000 | no |
| M1 | 24.2% | -38.3% | 0.63 | 403,704 | 0.80 | -7.9% (2018) | 5/10 | 100% | 17.3x | 4.93% | -36% -38% -32% -29% -31% | -0.051% (-1.7) | -0.132% | +0.017/-0.122 | no |
| M1b | 24.8% | -38.4% | 0.65 | 422,798 | 0.81 | -10.2% (2018) | 5/10 | 100% | 14.9x | 4.23% | -38% -37% -33% -29% -32% | -0.043% (-1.6) | -0.114% | -0.007/-0.080 | no |
| M2 | 26.2% | -47.2% | 0.55 | 469,805 | 0.83 | -10.8% (2018) | 6/10 | 98% | 16.4x | 4.58% | -47% -37% -29% -27% -29% | -0.015% (-0.6) | -0.101% | +0.038/-0.071 | no |
| M3a | 28.3% | -45.0% | 0.63 | 553,411 | 0.90 | -7.0% (2018) | 6/10 | 100% | 13.7x | 3.87% | -45% -37% -33% -28% -27% | +0.009% (0.4) | -0.061% | +0.028/-0.010 | no |
| M3b | 28.0% | -43.2% | 0.65 | 541,082 | 0.87 | -7.0% (2018) | 6/10 | 100% | 11.6x | 3.24% | -43% -37% -33% -30% -30% | +0.012% (1.1) | -0.025% | +0.019/+0.005 | no |
| M4a | 27.2% | -43.6% | 0.63 | 510,091 | 0.85 | -7.7% (2018) | 5/10 | 100% | 11.7x | 3.27% | -44% -36% -35% -29% -28% | +0.002% (0.1) | -0.049% | -0.028/+0.032 | no |
| M4b | 29.3% | -41.1% | 0.71 | 596,242 | 0.90 | -8.9% (2022) | 6/10 | 100% | 11.6x | 3.25% | -41% -37% -29% -27% -32% | +0.031% (0.8) | -0.086% | +0.085/-0.024 | no |
| M4c | 27.6% | -46.5% | 0.59 | 522,707 | 0.86 | -6.3% (2018) | 5/10 | 100% | 11.5x | 3.21% | -46% -38% -31% -29% -32% | +0.009% (0.3) | -0.079% | +0.057/-0.041 | no |
| M5 | 28.5% | -40.1% | 0.71 | 562,131 | 0.89 | -8.3% (2018) | 6/10 | 100% | 13.0x | 3.67% | -40% -36% -33% -30% -26% | +0.016% (0.8) | -0.047% | +0.027/+0.006 | no |
| M6 | 25.3% | -46.3% | 0.55 | 440,627 | 0.81 | -11.5% (2018) | 5/10 | 100% | 23.9x | 6.68% | -46% -37% -33% -30% -29% | -0.027% (-1.3) | -0.097% | -0.041/-0.013 | no |

Pass list: NONE (pass = ratio >= REF 0.61+0.10, CAGR >= 25%, ex-top5 paired >= 0, both halves > 0).
Timing cells equivalent to Monday open (within 1 pt CAGR and DD): M4a. M3 coverage (held name-weeks with a 2.02 event in prior 100 days): 64.5% over all 10080 name-weeks (81.3% from 2019-07; events_raw starts 2019) -> M3 VOID (<80%).
Conventions (PREREG silent, fixed before numbers): stops/events/gaps at the open; Monday pending stops executed + blocked from re-buy; M1/M5/M3b fill with next-ranked of Monday top-40; M3a slot in cash to Monday; M6 once per name per week, funded pro rata; M4 uses prior-day signal.
