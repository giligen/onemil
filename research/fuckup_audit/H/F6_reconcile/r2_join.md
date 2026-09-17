# R2 join - booked HOLD trades, A vs B, on (day, symbol)

counts by split (rows: split x in_A; cols in_B)
       A_only  B_only   both     A_n     B_n  A_netR  B_netR  A_only_netR  B_only_netR  both_A_netR  both_B_netR
split                                                                                                           
TEST    159.0   149.0  217.0   376.0   366.0   39.00  -15.36        49.79        -4.78       -10.79       -10.59
TRAIN   389.0   381.0  733.0  1122.0  1114.0   87.40   69.79        16.18         5.34        71.22        64.45
VAL     208.0   220.0  311.0   519.0   531.0  130.04  109.73        38.36        24.29        91.68        85.43

shared trades: field differences (all splits, n=1261)
      field  n_diff  pct  mean_abs  p50      min   max
    d_sig_m     394 31.2    2.3783  0.0 -140.000 0.000
  d_entry_m     394 31.2    2.3695  0.0 -140.000 0.000
   d_exit_m       4  0.3    0.6598  0.0 -379.000 0.000
    d_entry     381 30.2    0.0479  0.0   -4.000 0.935
     d_stop       6  0.5    0.0002  0.0    0.000 0.130
      d_net     520 41.2    0.0447  0.0   -6.454 1.477
    d_gross     387 30.7    0.0442  0.0   -6.247 1.491
why_differs       2  0.2       NaN  NaN      NaN   NaN

histogram of d_entry_m (A minus B, shared trades)
d_entry_m
-140.0      1
-115.0      1
-108.0      1
-80.0       1
-74.0       1
-72.0       1
-69.0       1
-66.0       1
-64.0       2
-61.0       1
-55.0       1
-53.0       2
-52.0       1
-43.0       1
-40.0       1
-38.0       1
-37.0       2
-35.0       1
-33.0       1
-32.0       1
-31.0       1
-30.0       1
-29.0       1
-27.0       2
-25.0       3
-24.0       1
-23.0       2
-21.0       2
-19.0       2
-18.0       3
-17.0       2
-16.0       1
-15.0       1
-14.0       6
-13.0       6
-12.0       3
-11.0       3
-10.0       4
-9.0        5
-8.0        3
-7.0        8
-6.0       10
-5.0       24
-4.0       29
-3.0       40
-2.0       59
-1.0      149
 0.0      867

histogram of d_exit_m bucketed
d_exit_m
(-1000000000.0, -60.0]       4
(-60.0, -10.0]               0
(-10.0, -1.0]                0
(-1.0, -1e-09]               0
(-1e-09, 1e-09]           1257
(1e-09, 1.0]                 0
(1.0, 10.0]                  0
(10.0, 60.0]                 0
(60.0, 1000000000.0]         0

histogram of d_net (A minus B)
d_net
(-1000000000.0, -2.0]      2
(-2.0, -1.0]               2
(-1.0, -0.25]             11
(-0.25, -0.01]            81
(-0.01, 0.01]            944
(0.01, 0.25]             188
(0.25, 1.0]               30
(1.0, 2.0]                 3
(2.0, 1000000000.0]        0

exit-reason cross-tab (shared)
B_why  eod  stop
A_why           
eod    854     0
stop     2   405

TRAIN: shared n=733  A_net=71.2  B_net=64.4  identical(entry,stop,exit_m,net)=414

VAL: shared n=311  A_net=91.7  B_net=85.4  identical(entry,stop,exit_m,net)=187

TEST: shared n=217  A_net=-10.8  B_net=-10.6  identical(entry,stop,exit_m,net)=140
