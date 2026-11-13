# RESULT 1,763 — levels & clocks

PREREG_1763. 7,487 TRAIN/VAL symbol-days; minute coverage 99.7 %; pre-market coverage (Alpaca SIP, 230/230 days)
98.8 % TRAIN / 98.7 % VAL (B1 not VOID). TRAIN written before VAL; VAL read once; TEST sealed. Cost: symbol-median
half-spread (≈ 50 % of trades) else price-decile band. Exits: imported 1752 `simulate`
(stop +13 bp, +2 R, 15:55); entry = trigger-bar close + half-spread + 2 bp; R floor 1 %.

| Read | VAL n/wk | VAL net R | day-t | ex-top-5 % | H1 / H2 | Verdict |
|---|---|---|---|---|---|---|
| B1 PM high | 84.8 | −0.151 | −1.86 | −0.265 | −0.21 / −0.19 | KILL |
| B2 5-min close | 49.5 | −0.113 | −1.61 | −0.225 | −0.32 / −0.18 | KILL (paired Δ −0.010, t −0.15) |
| B4 round number | 27.8 | −0.176 | −2.59 | −0.292 | −0.26 / −0.25 | KILL |
| B7 VWAP pullback | 68.3 | −0.197 | −2.93 | −0.314 | −0.32 / −0.33 | KILL |
| B8 halt proxy | 1.5 (≈ 80–100/yr) | +0.314 | 1.42 | +0.209 | −0.74 / −0.32 | KILL (t < 2; 2025 −, 2026 +; MDE80 0.62 R) |
| HOD comparator | 54.8 | −0.145 | −2.29 | −0.260 | −0.25 / −0.25 | reference |

B6 (HOD age at 10:30; breaks ≥ 10:30; VAL net, Δ vs rest, t): ≤ 15 −0.34 (−0.27, t −2.3); 15–45 −0.05 (+0.16,
t 1.3); > 45 −0.09 (+0.10, t 0.8). 15–45 Δ is positive in H1/H2/VAL (t 1.2/1.2/1.3), and so is its 10:30→close
return Δ (t 1.2/2.3/1.9). No bucket reaches t ≥ 2 in all three samples. **KILL.**

Gross R is about 0 on every read (−0.09…+0.09). Cost of ≈ 0.2 R/trade comes from 1–2 % stops on wide-spread names.
Green weeks fall below the null; weekly P10 is −17 to −34 R.

Caveats: B8 is a missing-minutes proxy, not real LULD halts. B7 VWAP uses the typical price.
B6 is causal only for breaks ≥ 10:30.

CHILDREN:
1. B8 on the real halt list (Nasdaq Trader LULD history), 2024-07..2026-05, gross-first PREREG. VAL is +0.31 R but TRAIN is opposite and needs about 4× n.
2. HOD-age 15–45 as a conditioning variable for a wide-stop entry (R ≥ 3 %, cost/R ≤ 0.07), pooled Δ test, PREREG first.
