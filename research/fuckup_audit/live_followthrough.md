# Follow-through volume on the honest BF cache (raw detector, 896 detections)

rows with bars 885 of 896

TRAIN: n 459 meanR -0.014 WR 42% | vol>=1.5x n 169 +0.075 WR 46% | vol<1.5x n 290 -0.066 WR 41% | gap>=2 n 216 +0.060 | gap<2 n 240 -0.081 | both n 80 +0.209 | EXIT-RULE book (all trades, low-vol sold at next open): meanR -0.102 vs -0.014, sum -46.8 vs -6.4
VAL: n 227 meanR -0.058 WR 41% | vol>=1.5x n 85 -0.072 WR 44% | vol<1.5x n 142 -0.049 WR 39% | gap>=2 n 122 -0.114 | gap<2 n 104 +0.002 | both n 44 -0.286 | EXIT-RULE book (all trades, low-vol sold at next open): meanR -0.188 vs -0.058, sum -42.6 vs -13.2
TEST: n 199 meanR -0.088 WR 41% | vol>=1.5x n 61 +0.140 WR 46% | vol<1.5x n 138 -0.189 WR 39% | gap>=2 n 98 -0.089 | gap<2 n 101 -0.087 | both n 30 +0.437 | EXIT-RULE book (all trades, low-vol sold at next open): meanR -0.087 vs -0.088, sum -17.3 vs -17.5
ALL: n 885 meanR -0.042 WR 42% | vol>=1.5x n 315 +0.048 WR 45% | vol<1.5x n 570 -0.091 WR 40% | gap>=2 n 436 -0.023 | gap<2 n 445 -0.063 | both n 154 +0.112 | EXIT-RULE book (all trades, low-vol sold at next open): meanR -0.121 vs -0.042, sum -106.7 vs -37.1

# ORB 10-minute time stop on the real trades (exit at fill+10 min open if below +0.25R)
n 113 real -13.4R (-0.118) -> time-stop -2.4R (-0.021); trades cut early 50

cells looked at in this file: BF 6 buckets x 4 splits + 1 exit rule; ORB 1 rule (0.25R at 10 min).