# PREREG — Sleeve #4: weekly short-term reversal (Jegadeesh 1990 / Lehmann 1990), frozen 2026-10-29 (Fable). Cell 1,713.

Sleeves #1–#3 rejected (1,710–1,712): every one NEGATIVE in point estimate on VAL 2020–22 vs its own universe.
**Power note, frozen:** with ~35 VAL months and monthly sd ≈ 4 %, the MDE for "excess > 0 with CI > 0" is ≈ 1.4 %/mo —
no published factor sleeve delivers that, so Stage-1 rejections rest on the SIGN of the point estimate and the DD/green
criteria, not on power; a sleeve with a positive point estimate below the MDE is "undetermined", reported as such.

## Rule (`research/sleeves/strev.py`, Stage 1 rejection-only on the Alpaca panel, same loader/screens as lowvol.py)
Universe: top 500 by 60-day median dollar volume, price ≥ $10 (reversal in small caps is a bid-ask bounce artefact),
name-pattern + fund-name exclusions. Signal each Friday close: prior 5-session return. Long the 20 LOWEST (losers),
equal weight, enter next Monday's official open, exit the following Monday's open (1-week hold, full weekly turnover).
Cost: 20 bps round trip placeholder (larger names), sensitivity 40 bps — the sleeve's turnover is ~100 %/week so cost
≈ 0.8–1.6 %/month; it must clear that. Long-only (no short leg). Sensitivities: top 50 losers; price ≥ $20; skip the
last session (Thu close signal, Mon entry).
Trigger day: Monday open — COLLIDES with MOM (Monday 09:46). Report the position overlap and return correlation; if the
sleeve passes it would run on the ORB paper account first to keep MOM's Monday clean.

## Evaluation / pass bar (VAL 2020-01..2022-12, TEST 2023+ SEALED)
Weekly returns aggregated to months for comparability. Mean monthly return net of cost, excess vs the equal-weight
universe, Sharpe, max DD, green-week share vs a count-matched random-20 null (1,000 draws), year-block bootstrap CI,
turnover, gross-vs-net decomposition (the cost model is the null — report gross separately).
Pass on VAL: net excess vs universe > 0 with CI > 0 AND max DD ≤ 20 % AND ≥ 52 % green weeks above the null.
Undetermined: net excess > 0 but CI spans 0 — reported, no ladder step. Reject: net excess ≤ 0.

## Ladder
As PREREG_LOWVOL (PIT question only after a Stage-1 pass; rebuild; paper $10K; live $5K after 2 months realized > 0).
