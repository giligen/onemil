# BF walk-forward selection refit — REPORT (2026-09-08, owner "run it") — REJECT

Refit target: the two-tier composite z-params (conviction_mult, qf_vwap_dist_pct, qf_fill_vwap_dist_pct, entry_minute) — the only fitted selection layer left under P1. Everything else (P1 rules, conviction threshold 1.8, composite threshold, MACD tier mults, sizing) frozen. Walk-forward: fit strictly before each Monday on the trailing window of raw cache rows; the week traded through the REAL Stage-2 (batch_backtest.py) on the P1 exit cache at $2K / $50K, daily loss limit −5u. Frozen reproduces the static P1 run from 2025-03-03 ($138,978) — the loop is valid.

| variant | trades | total | MDD | red | worst mo | green | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| frozen (live P1) | 51 | **138,978** | −5,324 | 4 | −5,105 | 14 | 100,983 | 37,995 |
| refit 13w | 56 | 123,117 | −5,324 | 4 | −5,105 | 14 | 88,622 | 34,495 |
| refit 20w | 57 | 121,172 | −5,324 | 4 | −5,105 | 14 | 88,622 | 32,550 |
| refit 26w | 56 | 123,117 | −5,324 | 4 | −5,105 | 14 | 88,622 | 34,495 |
| refit 39w | 56 | 123,117 | −5,324 | 4 | −5,105 | 14 | 88,622 | 34,495 |
| refit expanding | 57 | 121,172 | −5,324 | 4 | −5,105 | 14 | 88,622 | 32,550 |

Read: every window is worse in BOTH eras (−$12K to −$18K total); the refit changes only the Extras-tier composite, which admits 5–6 extra trades that lose. Drawdown / worst month / month colours are unchanged because the big trades are A-tier and untouched. No window helps → the hypothesis is closed for BF. The reason it works for ORB and not here: ORB's composite ranks the WHOLE book (every pick passes through it) so a drifted scale mis-ranks everything; BF's composite gates a side tier under three hard rules that already carry the 2026 edge — there is nothing left for it to fix.

Nothing ships. P1 stays as launched.
