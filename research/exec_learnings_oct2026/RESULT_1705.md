# RESULT 1,705 -- MOM sleeve entry timing (paper 10/5, 10/12, 10/19; live 10/19). Script mom_1705.py, rows mom_1705.csv
Data: logs/momentum_sleeve_ledger.csv (buys 12/13/14; 10/2 = 51-name initial build, shown only for reference), live ledger (20 buys, $10K).
Prior close = last daily_20261016.parquet bar before the date (all 4 dates: prior session, 0 NaN). 09:45/09:50 bar opens from Alpaca SIP 1-min (read-only, 59 symbol-days).
BT fill assumption (docs/momentum_sleeve_external_spec.md:27): "Backtest trades at Monday's official open. Live trades 09:45 ET market orders (measured half-spread 9.5 bp mean)". PREREG_1700: "market-on-open".

| rotation | n buys | buy $ | turnover $ (buy+sell) | gap paid, notional-wtd (bps) | median | slip vs 09:45/09:50 bar open (bps) | slip vs official daily open (ledger, bps) |
|---|---|---|---|---|---|---|---|
| paper 10/2 (build) | 51 | 36,895 | 52,690 | +19.3 | +31 | n/a | +46.6 |
| paper 10/5 | 12 | 4,163 | 8,337 | +9.4 | -79 | -0.9 | -15.2 |
| paper 10/12 | 13 | 4,658 | 9,304 | -37.7 | -206 | +12.4 | -26.1 |
| paper 10/19 | 14 | 4,916 | 9,828 | +187.1 | +251 | +8.3 | +13.3 |
| live 10/19 (09:50) | 20 | 10,000 | 10,000 (no sells) | +320.9 | +254 | +26.2 | +54.9 |

Frozen read: mean paper gap paid over the 3 rotations = +52.9 bps (> 50) -> nominally FLAGGED, but see caveat 1: 10/19 alone supplies it (10/5 +9, 10/12 -38; mean |gap| 103-277 bps per name).
Same-name 10/19, paper 09:45 vs live 09:50 (14 names): live fill higher by +29.1 bps mean (median +32.6), 11/14 higher. Of this ~15 bps is the market's own 09:45->09:50 bar-open drift (mean +14.8, median +9.9); remaining ~14 bps is extra slip (live 27.5 vs paper 13.2 vs own bar open).

Adversarial caveats
1. Gap vs prior close is NOT a cost relative to the BT: the BT buys at the official open, which already contains the overnight gap. The cost the BT did not charge is open->09:45/09:50 drift + spread. That is slip vs official open: paper -9.3 bps mean over 10/5-10/19 (n=3 rotations, sign mixed), live +54.9 bps on one rotation. The "> 50 bps" read as worded measures the wrong quantity; on the right one only live 10/19 exceeds 50 bps.
2. n = 3 paper + 1 live rotation; one day (10/19, a strong gap-up Monday: AXTI +930/+1078 bps, SNDK +814) drives both sign and size. No CI is meaningful; the sleeve holds momentum winners so gap direction is positively biased on up-markets, negative on down days (10/12).
3. Tail: single names move the notional-weighted figure (AXTI, SNDK, LITE 78 bps over bar open on 10/12). Not tested ex-top names.
4. Ledger slip_vs_open uses the SIP daily open (can be the opening auction print, not obtainable by a 09:45 market order); bar open (09:45/09:50) is the obtainable reference. Ledger rotation order timestamps are not in the CSV; I assumed the 09:45 / 09:50 minute per spec, not measured fill times (fills may land later in the minute or beyond).
5. 10/2 paper bars not fetched (initial build, 51 names, panel prior-close fine); its ledger WDC/STX rows (~-1260 bps vs prior close) look like a split/price-scale issue or duplicate rows (WDC listed twice) -- not investigated, excluded from the read.
6. Paper fills are simulated by Alpaca paper (own quote/size model); paper-vs-live difference mixes 5 minutes of drift, venue model and size ($500 vs ~$350 per name).
