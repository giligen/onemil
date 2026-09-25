# ORB out-of-sample edge claim — adversarial verification (2026-09-25, workflow wf_946f368f-ef9: 5 Sonnet lenses + Opus judge)

**Claim tested:** at the live config (veto OFF) the out-of-sample ORB edge is ~+0.08 R/fill (n 554, t 2.71), tail-carried.

## Judge verdict: REFUTED as stated

### What survives
ORB is not a proven edge, and it has not been refuted either. It has a small positive point estimate, weak evidence, and about 20 big winners carry its P&L. Both earlier statements to the owner overclaimed: "proven edge" and "kill it".

What survives:
- **The only book run under the live exit rule** is 2025-07-01..2026-09-23: n 389 fills over 201 days, about 6.1 fills per week. It averages +0.086 $-per-$375 per fill (day-clustered t 2.44, $12,574 at the $10K stage).
- **That book is in-sample for selection.** The vetoes, the catalyst switch, the entry buffer and the sizing were all chosen on it.
- **Its significance comes from one half-year.** 2026H1 has t 2.37; 2025H2 has 0.89 and 2026-07+ has 0.54.
- **It is tail-carried.** Removing the top 16 fills (4%) takes the total to zero or below. Ex-top-5% is -0.012 and capping winners at +1R gives +0.035.
- **The untuned window is positive but under the bar.** 2023-24 (n 165) was never used for tuning. It gives +0.055 with t 1.19 (CR1) to 1.55, but it uses a legacy 2R-target exit, not the live rule.
- **Cost is not the problem.** The flat 30/10 bps band is conservative against 98 live fills (round trip about 45 bps cheaper than modelled).

Plain English: there is probably a small positive expectancy, roughly +0.03 to +0.09 per fill after a selection haircut, but it is not established. Confidence the live edge is above zero is modest, somewhat better than a coin flip. Confidence in the size is none. The book is lottery-shaped and fails the cadence bar: 2 weeks at or above +5R out of 64, and 51.6% green weeks. That fits the live exploration tier at minimum size, not "proven" and not a scale-up. Forty live fills can only detect a failure of 0.3R or worse; confirming +0.08 needs about 300 fills.

### Fatal / major findings
FATAL (Lens B, upheld): the window called "out-of-sample" is not out-of-sample. Between 7/04 and 9/18, live parameters were chosen or validated on 2025H2 and 2026 data:
- the PDR 11.0 veto and the G1 veto, including its short-history switch;
- the range-size veto;
- the catalyst veto OFF (TRAIN = 2025, VAL = 2026-01..05);
- touch-go (walk-forward Jan'25 to May'26);
- the 300 bps spread gate;
- the 50 bps entry buffer (validated through 9/17);
- 8 slots at $375, which is the R denominator itself.
No window that was never tuned on reaches t >= 2.

MAJOR (Lens C, widened by the judge): all 165 fills from 2023-24 (30% of n), not just the 106 in book_1418, were priced by the legacy exit simulator (2R target, range-low stop), not the live static-lock / scale-out / 15:45 rule.
- Exit-reason evidence: book_1418 has stop 55, target 40, eod 11; book_1415 has stop 32, target 17, eod 10; thermo uses tag_bb, lock and scale_*.
- Both run logs print "0 resimmed rows; 0 rows scaled".
- Cause: study_orb_pipeline_static_lock.py:553 hard-codes data/cache.db for bars and silently keeps the legacy P&L when bars are missing.
- Lens C said book_1415 "resimmed correctly". That is wrong.
- The CLAUDE.md out-of-regime figures (+0.089 and -0.007, labelled "live config") are therefore not live-config exits.

MAJOR (tail): the top 23 of 554 fills (4.2%) zero out the pooled total; the top 16 of 389 zero out the live-exit slice. 11 of the 28 tail fills are 2x wrappers, and two single-underlying days (NBIS 7/30, CRCL 10/02) make up about 21% of the tail dollars, so there are only about 23 independent tail events.

MINOR: Lens A reproduces every number exactly; dropping 2026H1 takes t to 1.54. Lens D finds the flat cost band conservative, but no measured-NBBO model exists and 2023-24 cost is unverified.

LENS E, part wrong: its sizing finding is incorrect. The BT does apply the per-slot cap: study_orb_pipeline_static_lock.py:694-697 computes per_pos_cap = account/n and clips _rp_position to it. R is "$ per $375 nominal", which is a labelling issue, not a sign bias. Its other findings stay open:
- live took 3 of the 5 BT fills on 9/21-9/23 (ETRA and BIAF missed);
- the cause is unexplained: the session archive is a filtered 164-line extract and the journal has no BIAF order lines;
- the VNCE exit diverged on the same entry;
- n = 4 days.

### What would change the verdict
Toward "OOS edge": re-simulate the 2023-24 legs through the live exit rule, pointing the bars source at research/orb_2023/bars.db plus a 2024 bars source with ATR14 available. The claim is restored if that untuned window clears the orb_2023/PREREG EDGE bar: mean >= +0.08, day-clustered t >= 2, ex-top-5% > 0, and both calendar halves positive.

Toward "no edge", any one of:
- the re-simulated 2023-24 book comes out <= 0;
- live captures fewer than 70% of BT picks over 4+ weeks (ETRA/BIAF-type misses confirmed as systematic);
- the first 40 live fills average <= -0.3R.

About 300 live fills (roughly 50-80 weeks) are needed to confirm +0.08 R at t >= 2. The 40-fill ramp can only catch a broken book.

Also needed before any ORB number is cited again: fix the silent resim fallback in study_orb_pipeline_static_lock.py (bars-source parameter plus a WARNING or FATAL on 0 resimmed rows), and correct the "live config" wording in CLAUDE.md and research/orb_2023/REPORT.md.

## Lens A — Independent recomputation — threat: MINOR (/home/ec2-user/onemil/research/orb_verify/A_recompute.md)

Independent Python/statsmodels rebuild (no shared code) of the 554-fill OOS ORB book from the three raw CSVs via read_orb_csv. Headline numbers reproduce almost exactly: n=554, mean R=+0.0769 (claim +0.077), day-clustered t=2.714 (claim 2.71), total $15,969.10, ex-top-5% mean -0.0130, 6/7 half-years positive with 2024H2 the exception (-0.0066). Zero duplicates, zero missing _sized_pnl, zero cross-file overlap, dates within stated bounds; fill-count reconciles exactly (106+59+389=554). Two adversarial findings not visible in the headline: (1) pooled t=2.71 depends heavily on one half-year, 2026H1 (55.7% of total $, t=2.37 alone); drop it and t falls to 1.54 on the remaining 380 fills. (2) top 5% of fills (28) carry 116% of net P&L — the bottom 95% are net negative, a lottery-ticket pattern per CLAUDE.md rule 5. Minor: 3 undocumented zero-row trading days at the 2023/2024 file seam.

Key numbers: n=554 (claim 554, match); mean R=+0.0769 (claim +0.077); day-clustered t=2.714 (claim 2.71); total $15,969.10 (claim $15,969); ex-top5% mean R=-0.0130 (claim -0.013); top-5% share of total P&L=116% (28 trades); half-years positive 6/7, 2024H2=-0.0066 (claim -0.007); t without 2026H1=1.54 on n=380, $7,067 (not in claim, new); 2025H1 in-sample mean R=+0.193, t=2.79 (hotter than any OOS half); all provenance: own script /tmp/claude-1000/.../scratchpad/recompute.py over research/orb_2023/book_1418.csv, research/orb_2024/book_1415.csv, research/thermo/book_2025_26.csv.

Threat reason: Every headline number in the claim reproduces to 2-3 significant figures under fully independent code, and all data-integrity checks (duplicates, missing values, overlap, date bounds, fill-count reconciliation) are clean — no coding or data-plumbing error found. The claim is not overturned. But two adversarial facts materially qualify the "+0.08R, t=2.71" framing: the clustered significance collapses to t=1.54 if the single best half-year (2026H1) is excluded, and the entire net edge is carried by the top 5% of fills (bottom 95% net negative) — a textbook lottery-ticket pattern the claim's own phrasing ("top 5% carry the whole P&L") already half-admits but the headline doesn't foreground. These are caveats for confidence/robustness, not evidence the point estimate is wrong.

## Lens B — in-sample audit — threat: FATAL (/home/ec2-user/onemil/research/orb_verify/B_insample.md)

Traced every live orb.yaml parameter's fit/validation window against orb.yaml comments, frozen-params yaml, and 6 linked reports. Only the composite z-params/quintile cutoffs were fit on the excluded 2025H1. Everything else — PDR veto (11.0), G1 veto, catalyst-veto OFF, touchgo, spread gate (300bps), entry buffer (50bps), and the 8-slot/$375-risk sizing that defines R itself — was chosen or validated using 2025H2 and/or 2026 data, which is most of the claim's declared "out-of-sample" window. Reproduced the claim's pooled numbers from raw CSVs (matches: $15,969, +0.077R, -0.013 ex-top-5%) to confirm correct data, then recomputed on the one genuinely untouched slice: 2023-01..2024-12 only.

Key numbers: Untouched window (2023-2024, n=165/133 days): mean R=+0.055, day-clustered t=1.55, total $3,395 — below the t≥2 EDGE bar, matches orb_2023/PREREG.md's own pre-committed FLAT verdict. Contaminated window (thermo≥2025-07-01, n=389): mean R=+0.086, t=2.63. Pooled (claim as stated, n=554): mean R=+0.0769, t=3.05 (claim states t=2.71), total $15,969.10, ex-top-5% -0.013 (28 dropped) — reproduces the claim almost exactly. Sealed sub-slice inside the contaminated window (thermo≥2026-06-01, n=124/54 days, post-dating Stage-Q's 9/17 tuning cutoff): mean R=+0.067, t=1.05 — also sub-bar.

Threat reason: The claim's significance (t=2.71) is manufactured by pooling in data that repeatedly served as TRAIN/VAL/pre-ship-unveil for PDR, G1, catalyst-veto, touchgo, spread gate, entry buffer, and sizing between 2026-07-04 and 2026-09-18 (6 days before the claim's own cutoff). No window in ORB's history that was never used to tune a live parameter clears t≥2 — the only clean slice (2023-2024) gives t=1.55, and even the most literally sealed slice inside the touched window gives t=1.05. The "+0.08R out-of-sample, t 2.71" framing is not supportable as stated; a defensible restatement would be a positive but underpowered point estimate on data that was never independent of the config's own selection process.

## Lens C — Obtainability of the tail — threat: MAJOR (/home/ec2-user/onemil/research/orb_verify/C_tail.md)

Reproduced the 554-fill/+0.077R/ex-top5%-0.013R numbers exactly from the CSVs. Rebuilt all 28 top-5% trades against real minute bars (orb_2023/bars.db pre-2024-07, data/cache.db 2025-26, all reads done ~11:53 UTC). All 28 entries are obtainable (buy-stop limit price was inside a printed bar within 1 min of the range close, participation ≤2.7% of opening volume, EOD closes on real-volume bars). But found a material parity defect: study_orb_pipeline_static_lock.py sources bars only from data/cache.db, which its own run log shows had 0/4567 ATR14 hits and 0 scaled rows for cell 1418 — confirmed independently: cache.db has zero coverage before 2025-01-02. So all 106 entered trades in the 2023–2024-06-28 leg (book_1418), including 5 of the 28 tail trades (SIDU, NRGV, LIFW, SLNO, BFRG), silently kept pnl from the older non-production simulator (2R fixed target/range_low stop/60min timestop), never resimmed through the live static-lock/touchgo/3R-scale/15:45-close rule. Dropping just those 5 tail trades from the 554-pool: mean R 0.077→0.065, and ex-top-5% (recomputed) goes to -0.021, worse than claimed. Also 11/28 tail trades are 2x leveraged single-stock wrapper ETFs (NBIS×4, CRCL×3, OKLO×2, RGTI, GLXY, CRWV, LUNR) — same-day same-underlying duplicates cut independent tail events to ~23, with ~21% of tail P&L from just two single-underlying days.

Key numbers: 554 fills reproduced exactly; mean R +0.0769 (claim +0.077), total $15,969.10; ex-top-5% (28 trades) mean -0.0130 (claim -0.013) — all match. New: data/cache.db intraday coverage = 2025-01-02..2026-09-24 only (0 rows before 2024-07-01, verified by direct query). research/orb_2023/run_all.log: "ATR14 available for 0/4567 symbol-days" + "0 rows scaled 40%@+3.0R" for cell 1418, same for 1419 (0/1378). thermo book_2025_26 by contrast: ATR14 12953/13280, 489 rows scaled (correctly resimmed). Removing the 5 non-parity tail trades (SIDU $703.68, NRGV $478.55, LIFW $452.31, SLNO $449.55, BFRG $442.57 — combined $2,526.60): n 554→549, mean R 0.0769→0.0653, total pnl $15,969→$13,442; ex-top-5% of the cleaned 549 (new top-27) mean R = -0.0214 (worse than the -0.013 claimed). 11/28 tail symbols are 2x wrapper ETFs per data/research/orb_asset_class_map_20260711.csv; NBIS-wrapper cluster (2026-07-30, 4 trades) = $1,812 (11% of tail $); CRCL-wrapper cluster (2025-10-02, 3 trades) = $1,676 (10% of tail $). Entry obtainability: 28/28 obtainable (fill within 0-1 min of breakout bar), participation 0.0%-2.7% of opening 5-min volume (median ~0.3%).

Threat reason: The headline numbers reproduce exactly and no fill is literally unobtainable off the tape — so this isn't a "fatal, the whole number is fake" result. But a material, previously-undetected parity defect was found: 19% of the pooled sample (the entire 2023-2024H1 leg) never ran through the live exit simulator due to a silent bars-source fallback, and this affects 5 of the 28 tail-carrying trades. Removing just those 5 makes the tail (which the claim already flags as carrying the whole book) go from -0.013R to -0.021R ex-top-5% — the claim's own qualifier gets worse, not better, once the parity bug is corrected. Separately, tail independence is overstated by wrapper-cluster duplication. Together these mean the claim as stated ("out-of-sample net edge ~+0.08R, positive in 6/7 half-years") rests partly on non-live-parity trades and an overcounted tail; the sign likely survives but the confidence in the -0.013R/positive-tail-minus-lottery framing does not.

## Lens D — cost model (ORB OOS +0.077R claim) — threat: MINOR (/home/ec2-user/onemil/research/orb_verify/D_cost.md)

Checked what the ORB book actually charges vs CLAUDE.md's measured-NBBO rule. Both study_orb_pipeline_static_lock.py/study_orb.py (BT) and trading/orb_planner.py + orb.yaml (live) use the SAME flat band — entry_slip_bps=30 on range_high, exit_slip_bps=10 — not measured per-trade NBBO. No BT/live parity defect, but the flat-band mechanism itself is the class of error CLAUDE.md forbids. Three independent checks (research/exec_cost/REPORT_RECAL.md's ORB table; ran scripts/analyze_orb_slippage.py fresh on data/trades.db 169 orb rows; direct join of book_2025_26.csv modeled entry_price vs trades.db fill_price, n=45) all agree the flat band is mildly CONSERVATIVE (overstates cost) vs realized live fills, not permissive — so cost is not the mechanism inflating +0.077R. Coverage gap: all three checks use only the 2026-05..09 live book, covering the claim's 2025-07..2026-09-23 tail; 2023-2024 (289/554 fills, 52% of n) has no live fills to validate against.

Key numbers: Model: entry 30bps flat (orb.yaml:61, orb_planner.py:89-151, study_orb.py:46), exit 10bps flat (orb.yaml:330, study_orb_pipeline_static_lock.py:75). Recal report (research/exec_cost/REPORT_RECAL.md ll.30-40): current P=13.5bps vs realized O=3.1bps entry, moving to O changes ORB $ by only +$150-290 TRAIN / +$70-140 VAL out of $6-7K (favorable direction). Live trades.db (169 orb rows, 123 filled/98 closed, 2026-05-18..09-23): entry BT 30bps vs live mean 15.9bps/median 16.3bps; exit BT 10bps vs live mean -21.0bps/median 0.4bps; round-trip BT 40bps vs live -5.2bps (45.2bps cheaper than modeled). Direct join book_2025_26.csv entry_price vs trades.db fill_price (n=45 matched of 473 entered rows): mean diff -10.9bps, median -5.5bps (live cheaper than modeled).

Threat reason: Every available check points the same direction — the flat band overstates realized cost rather than hiding a loss, so recomputing under realized cost would raise the point estimate, not refute it. The genuine gap is that no true measured-NBBO model exists for ORB (only flat-vs-live-fills comparisons), and all live validation comes from the 2026-05..09 book, leaving 52% of the claim's n (2023-2024) with an unverified — not disproven — cost assumption. Doesn't address the claim's separately-disclosed tail dependence (top-5% carries all the P&L), which cost realism can't rescue.

## Lens E — Live vs backtest parity and frequency — threat: MAJOR (/home/ec2-user/onemil/research/orb_verify/E_parity.md)

Checked live ORB fills (data/trades.db, strategy='orb', since 2026-09-21) against research/thermo/book_2025_26.csv for the same dates, using session_archive logs (journalctl was too slow, killed after 120s). Only 4 live trading days exist under the current config (9/21-9/24); BT book only covers through 9/23. Live took 3 fills, BT claims 5 in the overlapping 3 days — live missed ETRA (9/21, zero entries all day, no logged veto explains it) and BIAF (9/22, the day's only winner, +$38 sized, un-vetoed but never submitted). Matched entries (CRCA/VNCE/GDXD) agree to the cent, but VNCE's exit mechanism diverged sharply (BT tag_bb -$3, live stop_loss -$44). Live's actual dollar risk ($87-168) never hit the $375 nominal unit the claim's R is built on — the $3,333/slot position-value cap binds, and BT's `_sized_pnl` uses a separate quintile-multiplier formula that doesn't replicate that cap.

Key numbers: Live fills since 9/21: 3 total (CRCA -$147, VNCE -$44, GDXD +$72); 0 on 9/21 and 9/24. BT book same window: 5 fills (ETRA -$40, CRCA -$126, BIAF +$38, VNCE -$3, GDXD +$68), max date 2026-09-23. Match rate: 3/5 BT fills captured live, 0 false positives. BT frequency 2025H2-2026: 389/63.9wk=6.09/wk (claim's cited 6.3/wk roughly confirmed); live observed: 3/4 trading days ≈3.75/wk. Live dollar risk per trade: $87-168 vs nominal $375 (position-value cap $3,333.33/8 slots binds, not the risk cap).

Threat reason: In the only window where live and BT overlap, live captured just 60% of BT's claimed fills and missed the single best trade (BIAF) with no logged veto explaining why; one matched entry (VNCE) had a materially different exit outcome; and live's actual per-trade dollar risk ($87-168) never matches the $375 unit the claim's R is normalized on, because a separate $3,333/slot position-value cap binds and BT's sizing formula (quintile multiplier) doesn't replicate that cap. Not "fatal" only because n=4 live days is too small to be statistically dispositive on its own — but every discrepancy found points the same direction (live underperforming/diverging from the BT book), and it compounds Lens B's finding that the "OOS" window wasn't clean and Lens A's tail-concentration finding. The zero-entry 9/21 day also warrants a direct check against the 9/21 corpse-gate fix given the project's documented history of silent universe-completeness bugs.

