# Stage H (F14 / F8 N=30 / F11) — search-adjusted permutation over all 83 TRAIN cells

observed max |t| = **2.44** | null mean 3.04 | null 95th pct **3.85** | null max 5.07 | **p = 0.932** (500 day-level sign-flip draws)

 book                                                               cell     mode  abs_t
  F14                   stack:V4 dist_open>0.8+V2 vwap>=0+V1 adv20 known   refill   2.44
  F14                   stack:V4 dist_open>0.8+V2 vwap>=0+V1 adv20 known norefill   2.41
  F14                                  stack:V4 dist_open>0.8+V2 vwap>=0   refill   2.19
  F14                                  stack:V4 dist_open>0.8+V2 vwap>=0 norefill   2.18
  F14                                                   V4 dist_open>0.8 norefill   2.08
  F14                                             stack:V4 dist_open>0.8 norefill   2.08
  F14                                             stack:V4 dist_open>0.8   refill   1.97
  F14                                                   V4 dist_open>0.8   refill   1.97
  F14                                                  V4b dist_open>0.5 norefill   1.97
  F14                                                        V2b vwap>=1 norefill   1.87
  F14                                                       V3b range<10 norefill   1.74
  F14                                                  V4b dist_open>0.5   refill   1.69
F11F6                                        stack:P1 pdr>=5+B1 body<1.0   refill   1.69
F8N30               alt:S1 spy_gap not [0,0.3)+X1 spy_vs_sma20 not [0,2)   refill   1.68
F8N30               alt:S1 spy_gap not [0,0.3)+X1 spy_vs_sma20 not [0,2) norefill   1.68
  F14                                                     V1 adv20 known norefill   1.63
F8N30                     stack:W1 no wrapper+M1 close_pos>=0.5+G1 gap<3 norefill   1.62
F11F6                                        stack:P1 pdr>=5+B1 body<1.0 norefill   1.62
  F14                                                       V3b range<10   refill   1.61
  F14                                                        V3 range<12 norefill   1.60
  F14                                                        V2b vwap>=1   refill   1.57
  F14                                                     V1 adv20 known   refill   1.56
F8N30                                          X1 spy_vs_sma20 not [0,2)   refill   1.53
F8N30                                          X1 spy_vs_sma20 not [0,2) norefill   1.53
  F14                                                       V3c range<15 norefill   1.52
F8N30                                         alt:S1 spy_gap not [0,0.3) norefill   1.49
F8N30                                             S1 spy_gap not [0,0.3) norefill   1.49
F8N30                                         alt:S1 spy_gap not [0,0.3)   refill   1.49
F8N30                                             S1 spy_gap not [0,0.3)   refill   1.49
  F14                                                       V3c range<15   refill   1.48
  F14                                                         V2 vwap>=0   refill   1.46
  F14                                                        V3 range<12   refill   1.41
  F14                                                         V2 vwap>=0 norefill   1.40
F8N30                                                  M1 close_pos>=0.5   refill   1.33
  F14                                                           baseline norefill   1.32
F8N30 alt:S1 spy_gap not [0,0.3)+X1 spy_vs_sma20 not [0,2)+D1 not Friday norefill   1.26
F8N30 alt:S1 spy_gap not [0,0.3)+X1 spy_vs_sma20 not [0,2)+D1 not Friday   refill   1.26
  F14                                                  V4c dist_open>1.5 norefill   1.25
F11F6                                                    stack:P1 pdr>=5   refill   1.20
F11F6                                                          P1 pdr>=5   refill   1.20
  F14                                                  V4c dist_open>1.5   refill   1.14
F8N30                                                      W1 no wrapper norefill   1.10
F8N30                                                stack:W1 no wrapper norefill   1.10
F8N30                                                      D1 not Friday   refill   1.07
F8N30                                                      D1 not Friday norefill   1.07
F8N30                              stack:W1 no wrapper+M1 close_pos>=0.5 norefill   1.02
F8N30                              stack:W1 no wrapper+M1 close_pos>=0.5   refill   1.01
F8N30                                                M1b close_pos>=0.25 norefill   1.00
F11F6                                                    stack:P1 pdr>=5 norefill   0.99
F11F6                                                          P1 pdr>=5 norefill   0.99
F8N30                                                   C1 dist_open>2.5   refill   0.97
F8N30                                                   C1 dist_open>2.5 norefill   0.95
F8N30                                                           G1 gap<3 norefill   0.91
F8N30                                                  M1 close_pos>=0.5 norefill   0.86
F8N30                                          I1 iwm_entry not [-0.5,0) norefill   0.77
F11F6                                                      D0 not Monday norefill   0.65
F11F6                                                      D0 not Monday   refill   0.65
F11F6                                                        B1 body<1.0   refill   0.60
F11F6                                                      W1 no wrapper   refill   0.57
F11F6                                                      W1 no wrapper norefill   0.55
F8N30                                                         G1b gap<10 norefill   0.55
F11F6                                                      PR1b price<20   refill   0.55
F11F6                                                        B1 body<1.0 norefill   0.54
F11F6                                                       PR1 price<50   refill   0.50
F8N30                                          I1 iwm_entry not [-0.5,0)   refill   0.48
F11F6                                                       B1b body<0.5   refill   0.45
F8N30                                                M1b close_pos>=0.25   refill   0.38
F11F6                                                           baseline norefill   0.35
F11F6                                                         P1b pdr>=8   refill   0.34
F8N30                                                           baseline norefill   0.34
F11F6                                                       B1b body<0.5 norefill   0.32
F8N30                                                     V1 adv20 known norefill   0.32
F8N30                                                           G1 gap<3   refill   0.30
F11F6                                                         P1b pdr>=8 norefill   0.29
F8N30                                                     V1 adv20 known   refill   0.28
F8N30                                                         G1b gap<10   refill   0.26
F8N30                     stack:W1 no wrapper+M1 close_pos>=0.5+G1 gap<3   refill   0.23
F11F6                                                     V1 adv20 known norefill   0.16
F11F6                                                       PR1 price<50 norefill   0.14
F8N30                                                stack:W1 no wrapper   refill   0.09
F8N30                                                      W1 no wrapper   refill   0.09
F11F6                                                     V1 adv20 known   refill   0.06
F11F6                                                      PR1b price<20 norefill   0.05
