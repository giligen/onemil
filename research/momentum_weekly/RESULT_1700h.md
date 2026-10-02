# RESULT 1700h -- per-name attribution of the A1 momentum book, 2017-2026

Book = cell 1,700g's V1_N20 (U2 large caps: price>=$10, ADV20>=$200M point-in-time incl. delisted; top 20
by 12-1 momentum; equal weight; weekly Monday rebalance; cost 5bps/side + half the high-low/close spread
proxy, capped 20bps). Script: 1700h_attribution.py, copied verbatim from 1700g_vol.py's panel loader,
universe filter, 12-1 signal, calendar and cost model; only addition is per-name weekly contribution
tracking. Full detail (all names, every year) in 1700h_by_year_names.csv (1,004 rows).

Convention: contribution_pts = 100 * sum of weekly `fwd_ret.fillna(0)/port_n` for that name that year --
sums EXACTLY to the year's arithmetic gross (pre-cost) return (checked in 1700h.log, max float32 gap
2.4e-8). NOT pts of `book_net_return` (compounded, cost-adjusted) -- cost + compounding explain the gap.
top5_share = top-5 contribution / that year's sum of POSITIVE contributions only (gross upside), matching
1700d_grid.py's existing top5_share_of. AI/semi list is the owner's (NVDA AVGO AMD SMCI ARM TSM MU VST
PLTR APP MSTR COIN) -- membership below is what the data shows, not assumed.

Year: book% / SPY% (excess) / distinct names / top5-of-gross-upside share / top10 (pts, weeks) / bottom5 / AI-in-top10 (share of top10 $) / verdict

2017: +12.4 / +21.3 (-9.0) / 93 names / top5sh 35% / MU+2.90(47) XYZ+1.74(26) NFLX+1.67(31) LRCX+1.47(40) BABA+1.36(25) AVGO+1.32(11) AMAT+1.28(43) BA+1.14(23) BAC+1.00(35) PYPL+0.98(17) / bottom: AAOI-1.70 FCX-1.11 CLF-0.71 LUV-0.66 ALNY-0.64 / AVGO,MU=28% / partial
2018: -5.5 / -4.3 (-1.2) / 106 names / top5sh 33% / XYZ+2.97(52) TWLO+1.77(15) SHOP+1.50(23) ADBE+1.33(28) ISRG+1.28(16) BA+1.16(30) NTAP+1.10(7) PANW+0.87(13) W+0.80(14) TAL+0.72(12) / bottom: OLED-2.06 AMD-1.95 RIOT-1.76 AMRN-1.69 M-1.37 / none=0% / did NOT drive
2019: +32.3 / +29.2 (+3.0) / 110 names / top5sh 43% / ROKU+5.20(41) SHOP+4.26(42) AMD+3.94(41) CMG+2.77(50) LULU+2.41(30) TEAM+2.36(22) TTD+2.17(44) ZS+1.48(12) LRCX+1.20(13) EW+1.17(14) / bottom: AMRN-1.84 ULTA-1.51 MDB-1.19 DXCM-0.56 QCOM-0.49 / AMD=15% / partial
**2020: +68.0 / +19.3 (+48.7) / 117 names / top5sh 32% / TSLA+9.04(46) NIO+8.05(22) SE+7.27(41) ENPH+6.63(33) NVAX+6.47(33) PLUG+6.31(21) SHOP+5.64(40) PDD+5.22(22) DOCU+4.33(30) RIOT+3.90(2) / bottom: HYLN-4.42 IBIO-3.86 SRNE-3.59 SPCE-3.55 CVNA-3.15 / none=0% / did NOT drive -- EV/solar/vaccine, not AI/semi**
2021: -11.9 / +28.6 (-40.5) / 101 names / top5sh 48% / MARA+8.34(51) GME+8.27(45) NVAX+6.22(23) RIOT+5.88(52) CAR+4.26(14) ARCT+3.29(4) MRNA+3.20(27) NXH+2.75(10) CLF+2.43(24) FVRR+2.36(12) / **top3 detractors: OCGN-3.88(4) TIGR-3.81(7) NNDM-3.00(5)** / none=0% / did NOT drive -- crypto-mining/meme/biotech
2022: -15.7 / -18.0 (+2.3) / 99 names / top5sh 36% / AR+2.85(43) DVN+2.72(52) COP+2.07(44) FANG+2.06(32) APA+1.87(44) BILL+1.71(4) RRC+1.61(8) ON+1.50(8) XOM+1.32(23) NUE+1.22(15) / bottom: AMC-2.95 MRNA-1.95 SIGA-1.83 BOIL-1.82 SQM-1.55 / none=0% / did NOT drive -- energy
2023: +20.6 / +24.7 (-4.1) / 131 names / top5sh 36% / SMCI+4.88(34) RIOT+4.22(24) MARA+3.86(4) DKNG+3.70(27) NVDA+3.40(35) MSTR+2.65(12) COIN+2.41(5) TQQQ+1.74(11) FSLR+1.60(35) SHOP+1.45(19) / **top3 detractors: VFS-1.92(2) CELH-1.80(12) AI-1.61(20)** / SMCI,RIOT(not on list),NVDA,MSTR,COIN=45% / partial -- AI+crypto mix present but book still LOST to SPY
**2024: +68.9 / +27.9 (+41.0) / 93 names / top5sh 39% / APP+9.93(44) CVNA+7.16(47) MSTR+6.92(50) NVDA+6.30(53) VRT+5.46(50) VST+5.11(39) PLTR+4.67(29) SMCI+3.78(39) HOOD+3.47(18) SPOT+2.58(21) / bottom: CLSK-3.26 VKTX-3.04 MU-1.53 AFRM-1.49 MARA-1.40 / APP,MSTR,NVDA,PLTR,SMCI,VST=66% / DOMINATED**
**2025: +59.8 / +16.5 (+43.3) / 86 names / top5sh 34% / HIMS+7.91(29) HOOD+7.33(52) ASTS+7.24(25) PLTR+6.21(47) SBET+6.01(3) QBTS+5.12(34) OKLO+5.09(42) BE+4.83(18) APP+4.09(45) LEU+3.92(26) / bottom: QUBT-3.12 LUNR-2.99 SOUN-2.59 UAMY-2.52 UPST-1.89 / APP,PLTR=18% / partial -- top names are GLP-1/fintech/space/quantum/nuclear, NOT core AI/semi**
**2026 (thru Sep): +49.8 / +12.8 (+37.0) / 68 names / top5sh 40% / AXTI+6.57(30) BE+6.42(37) SNDK+6.10(27) LITE+5.95(38) STX+5.91(34) WDC+5.32(38) MU+4.45(32) APLD+2.74(20) NBIS+2.68(13) COHR+2.14(6) / bottom: ONDS-4.21 EOSE-2.58 WOLF-2.53 CDE-2.49 QBTS-2.15 / MU=9% / partial -- top names are memory/storage/optical hardware (AI-datacenter-ADJACENT), not the named megacaps**

Bottom line: only 2024 is actually DOMINATED by the owner's named AI/semi list (66% of top-10). 2020's
blowout was EVs/solar/vaccines (0% list). 2025 was GLP-1/fintech/space/quantum (18%). 2026 is memory/
storage/optical hardware feeding AI datacenters but mostly off-list (9%, only MU). 2023 had real AI+crypto
representation (45%) and still lost to SPY by 4pts -- presence of the theme didn't guarantee beating the
index. Top-5-of-20 share of gross upside sits in a tight 32-48% band in every year regardless of whether
the book won or lost -- concentration in the top names is a constant feature of this book, not what
separates good years from bad ones; distinct names held per year (68-131) shows no single-name dependence.
