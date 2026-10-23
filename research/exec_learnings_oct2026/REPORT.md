# REPORT — execution learnings programme, Fri 2026-10-23 (cells 1,700–1,708; PREREG.md frozen before each number)

Every number below was built once and rebuilt independently from the PREREG prose (REBUILD_*.md); all headline
numbers agree (max disagreement 0.3 % in Σ$ on 1,708). Costs are the stored decision-time quotes; CIs are day-clustered.

## Weekly P&L 10/19–10/22 (context)
MOM-LIVE $10K −$288 (−2.9 %, first rotation) | MOM paper $17.4K −$44 (−12.9 % since 10/5) | HOD paper +$169 on 35 fills |
ORB paper −$195 on 3 | BF/TOM 0 | live account −$325 (= sleeve + fees).

## What the live record says (ORB live 123 fills 5/19–9/23, Σ −$5,281, mean R −0.11; + 14 paper)
1. **Latency is not where the money went.** Broker created→filled is wait-for-breakout (median 83 s), not latency; the
   engine clock before 10/19 is unrecoverable. The "fast fills −0.33 R / slow +0.27 R" live split (1,700) does NOT
   replicate on the 617-fill tick replay (1,707: 5–34 s is the BEST bucket, +0.23 R; ≤ 5 s −0.08 [−0.18, +0.02]).
   Recorded as selection noise. The engine's chase guard already removes marketable-and-too-far orders (140/617).
2. **Spread: the loss is in the middle, not the tail** (1,701). ≥ 150 bps = 2 fills, 3 % of losing dollars. The 50–150 bps
   bucket (40 fills, 33 %) holds −$3,270 = 62 % of the net loss at −0.20 R and 100 bps entry drift vs −0.05 R under
   50 bps. Difference 0.15 R < MDE 0.36 R → suggestive, unresolved. Gate untouched (never < 150 bps).
3. **Entry quality, not speed, is the live-vs-BT gap.** Only 35 % of live fills (21 % on the tape) sit inside the BT's
   30 bps entry model; median drift 50 bps.
4. **Stops are slow and too frequent** (1,702). 57 stop exits = −$15,026; only 4 of 57 within 5 min (median 32 min), so
   spread/latency on the entry bar does NOT explain them. Live stop share 41.6 % vs BT 28.4 % on the same window
   (13 pp, ≈ 2.2σ iid). Mechanism to test next: a fill 50–100 bps above the model on a ~3 % range puts the stop
   ⅙–⅓ R closer. 14 % of stopped trades had shown ≥ 1R open profit first. Winners: force_close +$6,304 on 29 (tail).
5. **The 9/25 ORB closure survives a real engine bug** (1,708). `replay.py` l.136 used a sizing ratio (0.0667) as the
   share count, so cell 1,426 modelled NO entry-price effect. Re-scored with true shares (mean 453): out-of-sample
   0 s +0.037 R [−0.005, +0.080] (was +0.028), 30 s +0.021, 60 s +0.019; ex-top-5 % −0.045; replay fills average
   23–32 bps BELOW the BT entry. Tape edge = hundredths of an R, tail-carried, not speed-dependent. Closure stands.
6. **BT↔live universe parity** (1,703, 13 sessions): top-8 differs in 4, orderable picks in 2 (10/12, 10/22). The nightly
   BT applies two gates that exist in neither orb.yaml nor the rulebook (< 5 range bars dropped — SSG, BNC; < 15K
   cached volume to 09:35 — ASTY, RAMZ). The ENGINE is per spec. 10/22 "SMST never ordered" = SSG missing from the BT.
   Also: the archive cron's grep misses `ORB:` lines, so decision_parity never sees below-threshold names.

## HOD (1,704; book = broker fills 9/25–10/22)
Book n 182, **−$339**, −0.04 R [−0.20, +0.12] on pnl/$50; ex-top-5 % −0.13 R (−$1,338) — tail-dependent. Weeks −140,
−556, −76, +265, +169; the green fortnight is z 1.05 (noise at this n). The book beats the all-armed population it
samples from (−0.21 R [−0.30, −0.11], n 960) by **+0.19–0.21 R, CI above 0** — the caps/selection are real; the book is
break-even. No first-break mechanism (slot 1–2 vs 3+: −0.02 R). **Book risk is not $50: median $37, 41 fills < $20** —
the $2K notional cap binds on wide stops; small-risk fills carry +0.82 R on tiny dollars, which inflates any R headline.
Dry prelude 9/14–9/24 (+$1,037, 70 modelled trades, two melt-up days at +2.0R) is a model, not a record — reported, not added.
Ramp rule (frozen): $50 → $100 R when realized book P&L since 9/25 > 0 AND fills ≥ 200; never under water. Not met.

## MOM sleeve (1,705)
The BT buys at Monday's official open, so the gap is in the BT. Uncharged cost = slippage from the open: paper −9 bps
(3 rotations), live 10/19 +55 bps (n 1; +29 bps vs paper on the same names, ~15 bps of it the 09:45→09:50 drift).
Action: live cron moved to 13:46/14:46 UTC (1 min after paper; the 5-min stagger had no function). Hygiene: the 10/2
build ledger has duplicated WDC/STX rows (−1,260 bps) — likely the $262 ledger-vs-broker gap of 10/12.

## Decisions (pre-committed rules, no menu)
* No entry filter from the chase split (FAIL on the frozen bar); ORB stays paper; closure statement updated to 1,708's numbers.
* Next ORB cell: stop-share mechanism — re-simulate exits on the tape at the live fill price (the replay never did).
* HOD stays live at $50 R under the ramp rule; sizing spec: size by risk, notional cap as a hard ceiling only.
* Spec for Sonnet: BT universe gates documented-or-removed (one helper, parity test) + archive grep `|ORB:`.
* Spec errors caught by builds this programme: 3 (L_brk ≠ latency; MOM gap already in BT; 1,702's "more than the BT
  implies" untestable). Engine bug caught by rebuild: 1 (replay shares). Cell count after this programme: 1,708.
