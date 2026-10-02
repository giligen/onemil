# RESULT 1700h -- per-name attribution of the A1 momentum book, 2017-2026

Generated 2026-10-02T12:14:50.732989+00:00. Book = cell 1,700g's V1_N20 (U2 large caps: price>=$10, ADV20>=$200M point-in-time incl. delisted; top 20 by 12-1 momentum; equal weight; weekly Monday rebalance; cost 5bps/side + half the high-low/close spread proxy, capped 20bps).

Convention: contribution_pts = 100 * sum of weekly `fwd_ret.fillna(0)/port_n` for that name in that year. By construction this sums EXACTLY to the year's arithmetic sum of weekly gross (pre-cost) returns -- checked explicitly in 1700h.log, not assumed. It is NOT percentage points of the `book_net_return` column (the compounded, cost-adjusted headline figure) -- cost and arithmetic-vs-geometric compounding both drive the gap, left visible rather than asserted away. top5_share is the top-5 names' contribution as a share of that year's sum of POSITIVE contributions only (gross upside), matching 1700d_grid.py's existing top5_share_of.

## 2017: book net +12.4%  SPY +21.3%  (excess -9.0 pts)  --  93 distinct names held
Top-5 share of gross upside: 35%

Top 10 contributors (symbol, weeks held, contribution pts):
  MU     weeks= 47   +2.90 pts
  XYZ    weeks= 26   +1.74 pts
  NFLX   weeks= 31   +1.67 pts
  LRCX   weeks= 40   +1.47 pts
  BABA   weeks= 25   +1.36 pts
  AVGO   weeks= 11   +1.32 pts
  AMAT   weeks= 43   +1.28 pts
  BA     weeks= 23   +1.14 pts
  BAC    weeks= 35   +1.00 pts
  PYPL   weeks= 17   +0.98 pts

Bottom 5 detractors:
  AAOI   weeks= 14   -1.70 pts
  FCX    weeks=  6   -1.11 pts
  CLF    weeks=  3   -0.71 pts
  LUV    weeks=  5   -0.66 pts
  ALNY   weeks=  6   -0.64 pts

AI/semiconductor complex in top 10: ['AVGO', 'MU'] -- 28% of top-10 contribution
Verdict: the AI/semiconductor complex partially drove this year's top contributors.

## 2018: book net -5.5%  SPY -4.3%  (excess -1.2 pts)  --  106 distinct names held
Top-5 share of gross upside: 33%

Top 10 contributors (symbol, weeks held, contribution pts):
  XYZ    weeks= 52   +2.97 pts
  TWLO   weeks= 15   +1.77 pts
  SHOP   weeks= 23   +1.50 pts
  ADBE   weeks= 28   +1.33 pts
  ISRG   weeks= 16   +1.28 pts
  BA     weeks= 30   +1.16 pts
  NTAP   weeks=  7   +1.10 pts
  PANW   weeks= 13   +0.87 pts
  W      weeks= 14   +0.80 pts
  TAL    weeks= 12   +0.72 pts

Bottom 5 detractors:
  OLED   weeks=  8   -2.06 pts
  AMD    weeks= 16   -1.95 pts
  RIOT   weeks=  3   -1.76 pts
  AMRN   weeks=  7   -1.69 pts
  M      weeks= 11   -1.37 pts

AI/semiconductor complex in top 10: none -- 0% of top-10 contribution
Verdict: the AI/semiconductor complex did NOT drive this year's top contributors.

## 2019: book net +32.3%  SPY +29.2%  (excess +3.0 pts)  --  110 distinct names held
Top-5 share of gross upside: 43%

Top 10 contributors (symbol, weeks held, contribution pts):
  ROKU   weeks= 41   +5.20 pts
  SHOP   weeks= 42   +4.26 pts
  AMD    weeks= 41   +3.94 pts
  CMG    weeks= 50   +2.77 pts
  LULU   weeks= 30   +2.41 pts
  TEAM   weeks= 22   +2.36 pts
  TTD    weeks= 44   +2.17 pts
  ZS     weeks= 12   +1.48 pts
  LRCX   weeks= 13   +1.20 pts
  EW     weeks= 14   +1.17 pts

Bottom 5 detractors:
  AMRN   weeks=  4   -1.84 pts
  ULTA   weeks= 14   -1.51 pts
  MDB    weeks= 23   -1.19 pts
  DXCM   weeks= 10   -0.56 pts
  QCOM   weeks=  6   -0.49 pts

AI/semiconductor complex in top 10: ['AMD'] -- 15% of top-10 contribution
Verdict: the AI/semiconductor complex partially drove this year's top contributors.

## 2020: book net +68.0%  SPY +19.3%  (excess +48.7 pts)  --  117 distinct names held
Top-5 share of gross upside: 32%

Top 10 contributors (symbol, weeks held, contribution pts):
  TSLA   weeks= 46   +9.04 pts
  NIO    weeks= 22   +8.05 pts
  SE     weeks= 41   +7.27 pts
  ENPH   weeks= 33   +6.63 pts
  NVAX   weeks= 33   +6.47 pts
  PLUG   weeks= 21   +6.31 pts
  SHOP   weeks= 40   +5.64 pts
  PDD    weeks= 22   +5.22 pts
  DOCU   weeks= 30   +4.33 pts
  RIOT   weeks=  2   +3.90 pts

Bottom 5 detractors:
  HYLN   weeks=  5   -4.42 pts
  IBIO   weeks=  4   -3.86 pts
  SRNE   weeks=  4   -3.59 pts
  SPCE   weeks= 13   -3.55 pts
  CVNA   weeks=  3   -3.15 pts

AI/semiconductor complex in top 10: none -- 0% of top-10 contribution
Verdict: the AI/semiconductor complex did NOT drive this year's top contributors.

## 2021: book net -11.9%  SPY +28.6%  (excess -40.5 pts)  --  101 distinct names held
Top-5 share of gross upside: 48%

Top 10 contributors (symbol, weeks held, contribution pts):
  MARA   weeks= 51   +8.34 pts
  GME    weeks= 45   +8.27 pts
  NVAX   weeks= 23   +6.22 pts
  RIOT   weeks= 52   +5.88 pts
  CAR    weeks= 14   +4.26 pts
  ARCT   weeks=  4   +3.29 pts
  MRNA   weeks= 27   +3.20 pts
  NXH    weeks= 10   +2.75 pts
  CLF    weeks= 24   +2.43 pts
  FVRR   weeks= 12   +2.36 pts

Bottom 5 detractors:
  OCGN   weeks=  4   -3.88 pts
  TIGR   weeks=  7   -3.81 pts
  NNDM   weeks=  5   -3.00 pts
  AMC    weeks= 27   -2.43 pts
  SOS    weeks=  5   -2.41 pts

AI/semiconductor complex in top 10: none -- 0% of top-10 contribution
Verdict: the AI/semiconductor complex did NOT drive this year's top contributors.

## 2022: book net -15.7%  SPY -18.0%  (excess +2.3 pts)  --  99 distinct names held
Top-5 share of gross upside: 36%

Top 10 contributors (symbol, weeks held, contribution pts):
  AR     weeks= 43   +2.85 pts
  DVN    weeks= 52   +2.72 pts
  COP    weeks= 44   +2.07 pts
  FANG   weeks= 32   +2.06 pts
  APA    weeks= 44   +1.87 pts
  BILL   weeks=  4   +1.71 pts
  RRC    weeks=  8   +1.61 pts
  ON     weeks=  8   +1.50 pts
  XOM    weeks= 23   +1.32 pts
  NUE    weeks= 15   +1.22 pts

Bottom 5 detractors:
  AMC    weeks= 16   -2.95 pts
  MRNA   weeks=  4   -1.95 pts
  SIGA   weeks=  3   -1.83 pts
  BOIL   weeks=  8   -1.82 pts
  SQM    weeks=  5   -1.55 pts

AI/semiconductor complex in top 10: none -- 0% of top-10 contribution
Verdict: the AI/semiconductor complex did NOT drive this year's top contributors.

## 2023: book net +20.6%  SPY +24.7%  (excess -4.1 pts)  --  131 distinct names held
Top-5 share of gross upside: 36%

Top 10 contributors (symbol, weeks held, contribution pts):
  SMCI   weeks= 34   +4.88 pts
  RIOT   weeks= 24   +4.22 pts
  MARA   weeks=  4   +3.86 pts
  DKNG   weeks= 27   +3.70 pts
  NVDA   weeks= 35   +3.40 pts
  MSTR   weeks= 12   +2.65 pts
  COIN   weeks=  5   +2.41 pts
  TQQQ   weeks= 11   +1.74 pts
  FSLR   weeks= 35   +1.60 pts
  SHOP   weeks= 19   +1.45 pts

Bottom 5 detractors:
  VFS    weeks=  2   -1.92 pts
  CELH   weeks= 12   -1.80 pts
  AI     weeks= 20   -1.61 pts
  PDD    weeks= 25   -1.56 pts
  SRPT   weeks= 11   -1.55 pts

AI/semiconductor complex in top 10: ['COIN', 'MSTR', 'NVDA', 'SMCI'] -- 45% of top-10 contribution
Verdict: the AI/semiconductor complex partially drove this year's top contributors.

## 2024: book net +68.9%  SPY +27.9%  (excess +41.0 pts)  --  93 distinct names held
Top-5 share of gross upside: 39%

Top 10 contributors (symbol, weeks held, contribution pts):
  APP    weeks= 44   +9.93 pts
  CVNA   weeks= 47   +7.16 pts
  MSTR   weeks= 50   +6.92 pts
  NVDA   weeks= 53   +6.30 pts
  VRT    weeks= 50   +5.46 pts
  VST    weeks= 39   +5.11 pts
  PLTR   weeks= 29   +4.67 pts
  SMCI   weeks= 39   +3.78 pts
  HOOD   weeks= 18   +3.47 pts
  SPOT   weeks= 21   +2.58 pts

Bottom 5 detractors:
  CLSK   weeks= 36   -3.26 pts
  VKTX   weeks= 32   -3.04 pts
  MU     weeks=  5   -1.53 pts
  AFRM   weeks= 23   -1.49 pts
  MARA   weeks= 20   -1.40 pts

AI/semiconductor complex in top 10: ['APP', 'MSTR', 'NVDA', 'PLTR', 'SMCI', 'VST'] -- 66% of top-10 contribution
Verdict: the AI/semiconductor complex DOMINATED this year's top contributors.

## 2025: book net +59.8%  SPY +16.5%  (excess +43.3 pts)  --  86 distinct names held
Top-5 share of gross upside: 34%

Top 10 contributors (symbol, weeks held, contribution pts):
  HIMS   weeks= 29   +7.91 pts
  HOOD   weeks= 52   +7.33 pts
  ASTS   weeks= 25   +7.24 pts
  PLTR   weeks= 47   +6.21 pts
  SBET   weeks=  3   +6.01 pts
  QBTS   weeks= 34   +5.12 pts
  OKLO   weeks= 42   +5.09 pts
  BE     weeks= 18   +4.83 pts
  APP    weeks= 45   +4.09 pts
  LEU    weeks= 26   +3.92 pts

Bottom 5 detractors:
  QUBT   weeks= 30   -3.12 pts
  LUNR   weeks=  8   -2.99 pts
  SOUN   weeks= 10   -2.59 pts
  UAMY   weeks=  2   -2.52 pts
  UPST   weeks= 14   -1.89 pts

AI/semiconductor complex in top 10: ['APP', 'PLTR'] -- 18% of top-10 contribution
Verdict: the AI/semiconductor complex partially drove this year's top contributors.

## 2026: book net +49.8%  SPY +12.8%  (excess +37.0 pts)  --  68 distinct names held
Top-5 share of gross upside: 40%

Top 10 contributors (symbol, weeks held, contribution pts):
  AXTI   weeks= 30   +6.57 pts
  BE     weeks= 37   +6.42 pts
  SNDK   weeks= 27   +6.10 pts
  LITE   weeks= 38   +5.95 pts
  STX    weeks= 34   +5.91 pts
  WDC    weeks= 38   +5.32 pts
  MU     weeks= 32   +4.45 pts
  APLD   weeks= 20   +2.74 pts
  NBIS   weeks= 13   +2.68 pts
  COHR   weeks=  6   +2.14 pts

Bottom 5 detractors:
  ONDS   weeks= 13   -4.21 pts
  EOSE   weeks=  2   -2.58 pts
  WOLF   weeks= 11   -2.53 pts
  CDE    weeks=  6   -2.49 pts
  QBTS   weeks=  8   -2.15 pts

AI/semiconductor complex in top 10: ['MU'] -- 9% of top-10 contribution
Verdict: the AI/semiconductor complex partially drove this year's top contributors.
