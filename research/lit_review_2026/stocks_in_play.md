# H-B1 stocks-in-play ORB, OOS 2025-26 | picks 7734 | filled 6205 (80%) | median R 0.40% of price

TRAIN n  3470 (15.7/day) gross -0.124R hit 9.9% stop-rate 89% | net@1bp -0.185R net@20bp -0.740R | weekly net@1bp -13.7R green 12/47 worst -76.1
VAL   n  1652 (16.2/day) gross -0.250R hit 9.9% stop-rate 89% | net@1bp -0.305R net@20bp -0.794R | weekly net@1bp -22.9R green 5/22 worst -70.5
TEST  n  1083 (15.9/day) gross -0.165R hit 11.2% stop-rate 87% | net@1bp -0.216R net@20bp -0.681R | weekly net@1bp -16.7R green 4/14 worst -54.7
ALL   n  6205 (15.9/day) gross -0.164R hit 10.1% stop-rate 89% | net@1bp -0.222R net@20bp -0.744R | weekly net@1bp -16.8R green 20/82 worst -76.1

by RV bucket (paper: monotone, +0.08R at >=1, +0.38R at >30x):
           n                gross_R               net_1bp                  hit              
split   TEST   TRAIN    VAL    TEST  TRAIN    VAL    TEST  TRAIN    VAL   TEST  TRAIN    VAL
rvb                                                                                         
1-2      NaN     3.0    NaN     NaN -1.000    NaN     NaN -1.043    NaN    NaN  0.000    NaN
2-5    287.0  1022.0  264.0  -0.205 -0.105 -0.128  -0.259 -0.166 -0.180  0.143  0.135  0.148
5-10   492.0  1550.0  887.0  -0.168 -0.195 -0.275  -0.219 -0.258 -0.331  0.108  0.091  0.089
10-30  269.0   788.0  443.0  -0.477 -0.009 -0.210  -0.527 -0.070 -0.263  0.078  0.062  0.090
>30     35.0   107.0   58.0   2.614 -0.082 -0.739   2.566 -0.137 -0.784  0.171  0.131  0.086

long vs short:
               n  gross_R  net_1bp
split side                        
TEST  -1     507    0.014   -0.035
       1     576   -0.321   -0.375
TRAIN -1    1701   -0.076   -0.136
       1    1769   -0.170   -0.233
VAL   -1     803   -0.273   -0.328
       1     849   -0.229   -0.283