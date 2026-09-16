# A3 — intraday market state at the entry minute, entries >= 10:00, contract (c)

cells: 208 bucket x split rows | 104 buckets | TRAIN->VAL sign agreement 0.51

           key exit feat  bucket  n_TRAIN  meanR_TRAIN  t_TRAIN  n_VAL  meanR_VAL  t_VAL  sign_agree
F1 {"P": 0.12}   2r  brd  T1 low    155.0       -0.207    -1.73   42.0     -0.358  -1.47        True
F1 {"P": 0.12}   2r  brd  T2 mid    150.0        0.006     0.05   50.0     -0.131  -0.63       False
F1 {"P": 0.12}   2r  brd T3 high    152.0       -0.433    -3.28   92.0     -0.113  -0.71        True
F1 {"P": 0.12}   2r  iwm  T1 low    152.0       -0.109    -0.87   62.0     -0.313  -1.66        True
F1 {"P": 0.12}   2r  iwm  T2 mid    152.0       -0.329    -2.93   52.0     -0.055  -0.25        True
F1 {"P": 0.12}   2r  iwm T3 high    153.0       -0.199    -1.49   70.0     -0.140  -0.76        True
F1 {"P": 0.12}   2r  iwm  sign +    241.0       -0.252    -2.53   96.0     -0.192  -1.26        True
F1 {"P": 0.12}   2r  iwm  sign -    216.0       -0.167    -1.63   88.0     -0.155  -0.94        True
F1 {"P": 0.12}   2r  spy  T1 low    152.0       -0.047    -0.39   58.0     -0.329  -1.69        True
F1 {"P": 0.12}   2r  spy  T2 mid    152.0       -0.382    -3.26   56.0     -0.380  -2.00        True
F1 {"P": 0.12}   2r  spy T3 high    153.0       -0.207    -1.57   70.0      0.119   0.63       False
F1 {"P": 0.12}   2r  spy  sign +    244.0       -0.286    -2.91  110.0     -0.062  -0.42        True
F1 {"P": 0.12}   2r  spy  sign -    213.0       -0.128    -1.23   74.0     -0.341  -2.01        True
F1 {"P": 0.12} hold  brd  T1 low    154.0        0.483     1.02   42.0     -0.293  -0.54       False
F1 {"P": 0.12} hold  brd  T2 mid    149.0        1.447     2.26   50.0     -0.283  -0.78       False
F1 {"P": 0.12} hold  brd T3 high    151.0       -0.308    -1.27   90.0     -0.334  -1.28        True
F1 {"P": 0.12} hold  iwm  T1 low    151.0        0.792     1.76   61.0     -0.184  -0.43       False
F1 {"P": 0.12} hold  iwm  T2 mid    151.0        0.548     1.04   52.0     -0.731  -2.86       False
F1 {"P": 0.12} hold  iwm T3 high    152.0        0.271     0.59   69.0     -0.105  -0.32       False
F1 {"P": 0.12} hold  iwm  sign +    239.0        0.617     1.41   94.0     -0.230  -0.86       False
F1 {"P": 0.12} hold  iwm  sign -    215.0        0.446     1.36   88.0     -0.396  -1.28       False
F1 {"P": 0.12} hold  spy  T1 low    151.0        0.324     1.01   58.0     -0.289  -0.65       False
F1 {"P": 0.12} hold  spy  T2 mid    151.0        0.111     0.31   54.0     -0.796  -4.08       False
F1 {"P": 0.12} hold  spy T3 high    152.0        1.169     1.74   70.0      0.047   0.13        True
F1 {"P": 0.12} hold  spy  sign +    242.0        0.628     1.42  108.0     -0.268  -1.13       False
F1 {"P": 0.12} hold  spy  sign -    212.0        0.432     1.36   74.0     -0.372  -1.02       False
         F6 {}   2r  brd  T1 low    348.0        0.036     0.83   84.0      0.071   0.87        True
         F6 {}   2r  brd  T2 mid    348.0        0.044     1.10  124.0      0.093   1.42        True
         F6 {}   2r  brd T3 high    348.0        0.000     0.00  240.0      0.054   1.12       False
         F6 {}   2r  iwm  T1 low    348.0       -0.001    -0.04  166.0      0.147   2.40       False
         F6 {}   2r  iwm  T2 mid    348.0       -0.003    -0.07  124.0     -0.069  -1.10        True
         F6 {}   2r  iwm T3 high    348.0        0.084     2.00  158.0      0.091   1.64        True
         F6 {}   2r  iwm  sign +    543.0        0.071     2.08  231.0      0.041   0.90        True
         F6 {}   2r  iwm  sign -    501.0       -0.022    -0.69  217.0      0.097   1.80       False
         F6 {}   2r  spy  T1 low    348.0        0.029     0.72  135.0      0.014   0.24        True
         F6 {}   2r  spy  T2 mid    348.0       -0.016    -0.39  151.0      0.115   1.81       False
         F6 {}   2r  spy T3 high    348.0        0.066     1.63  162.0      0.068   1.19        True
         F6 {}   2r  spy  sign +    565.0        0.060     1.82  253.0      0.110   2.32        True
         F6 {}   2r  spy  sign -    479.0       -0.013    -0.38  195.0      0.014   0.26       False
         F6 {} hold  brd  T1 low    340.0        0.054     1.01   82.0      0.067   0.83        True
         F6 {} hold  brd  T2 mid    339.0        0.031     0.73  121.0      0.048   0.76        True
         F6 {} hold  brd T3 high    339.0        0.024     0.51  233.0      0.141   1.50        True
         F6 {} hold  iwm  T1 low    339.0       -0.023    -0.56  162.0      0.125   1.93       False
         F6 {} hold  iwm  T2 mid    339.0        0.024     0.45  120.0     -0.031  -0.41       False
         F6 {} hold  iwm T3 high    340.0        0.107     2.26  154.0      0.179   1.41        True
         F6 {} hold  iwm  sign +    525.0        0.094     2.40  223.0      0.125   1.34        True
         F6 {} hold  iwm  sign -    493.0       -0.026    -0.68  213.0      0.076   1.36       False
         F6 {} hold  spy  T1 low    339.0        0.015     0.34  131.0     -0.017  -0.29       False
         F6 {} hold  spy  T2 mid    340.0       -0.012    -0.24  146.0      0.252   1.78       False
         F6 {} hold  spy T3 high    339.0        0.106     2.20  159.0      0.060   1.06        True
         F6 {} hold  spy  sign +    547.0        0.082     2.15  246.0      0.200   2.25        True
         F6 {} hold  spy  sign -    471.0       -0.017    -0.43  190.0     -0.027  -0.55        True
  F8 {"N": 30}   2r  brd  T1 low    383.0       -0.007    -0.20   98.0     -0.042  -0.70        True
  F8 {"N": 30}   2r  brd  T2 mid    382.0       -0.003    -0.09  142.0      0.051   0.80       False
  F8 {"N": 30}   2r  brd T3 high    383.0        0.025     0.67  223.0     -0.026  -0.56       False
  F8 {"N": 30}   2r  iwm  T1 low    382.0        0.002     0.06  188.0     -0.006  -0.11       False
  F8 {"N": 30}   2r  iwm  T2 mid    383.0       -0.002    -0.06  116.0     -0.087  -1.33        True
  F8 {"N": 30}   2r  iwm T3 high    382.0        0.017     0.47  159.0      0.054   0.93        True
  F8 {"N": 30}   2r  iwm  sign +    542.0        0.008     0.26  226.0      0.022   0.46        True
  F8 {"N": 30}   2r  iwm  sign -    605.0        0.004     0.12  237.0     -0.032  -0.75       False
  F8 {"N": 30}   2r  spy  T1 low    383.0        0.076     1.98  161.0     -0.152  -2.82       False
  F8 {"N": 30}   2r  spy  T2 mid    385.0       -0.105    -3.13  125.0      0.050   0.81       False
  F8 {"N": 30}   2r  spy T3 high    380.0        0.044     1.12  177.0      0.088   1.68        True
  F8 {"N": 30}   2r  spy  sign +    589.0        0.014     0.46  265.0      0.075   1.77        True
  F8 {"N": 30}   2r  spy  sign -    559.0       -0.005    -0.16  198.0     -0.113  -2.28        True
  F8 {"N": 30} hold  brd  T1 low    375.0       -0.006    -0.16   98.0     -0.042  -0.70        True
  F8 {"N": 30} hold  brd  T2 mid    374.0        0.006     0.14  137.0      0.062   0.88        True
  F8 {"N": 30} hold  brd T3 high    375.0        0.035     0.84  221.0     -0.033  -0.70       False
  F8 {"N": 30} hold  iwm  T1 low    375.0        0.021     0.52  181.0     -0.015  -0.29       False
  F8 {"N": 30} hold  iwm  T2 mid    374.0       -0.005    -0.13  120.0     -0.071  -1.10        True
  F8 {"N": 30} hold  iwm T3 high    374.0        0.021     0.51  155.0      0.053   0.86        True
  F8 {"N": 30} hold  iwm  sign +    524.0        0.006     0.18  221.0      0.026   0.50        True
  F8 {"N": 30} hold  iwm  sign -    599.0        0.018     0.56  235.0     -0.036  -0.84       False
  F8 {"N": 30} hold  spy  T1 low    376.0        0.102     2.33  156.0     -0.160  -2.92       False
  F8 {"N": 30} hold  spy  T2 mid    375.0       -0.125    -3.89  129.0      0.065   1.04       False
  F8 {"N": 30} hold  spy T3 high    373.0        0.057     1.29  171.0      0.080   1.44        True
  F8 {"N": 30} hold  spy  sign +    574.0        0.009     0.26  259.0      0.072   1.61        True
  F8 {"N": 30} hold  spy  sign -    550.0        0.015     0.43  197.0     -0.109  -2.17       False
   F8 {"N": 5}   2r  brd  T1 low    499.0       -0.105    -2.19  131.0      0.068   0.73       False
   F8 {"N": 5}   2r  brd  T2 mid    498.0       -0.097    -2.04  171.0      0.030   0.37       False
   F8 {"N": 5}   2r  brd T3 high    498.0       -0.042    -0.91  281.0      0.092   1.51       False
   F8 {"N": 5}   2r  iwm  T1 low    498.0       -0.129    -2.60  227.0      0.031   0.44       False
   F8 {"N": 5}   2r  iwm  T2 mid    498.0       -0.121    -2.71  173.0      0.075   0.91       False
   F8 {"N": 5}   2r  iwm T3 high    499.0        0.006     0.14  183.0      0.109   1.48        True
   F8 {"N": 5}   2r  iwm  sign +    737.0       -0.020    -0.52  263.0      0.139   2.16       False
   F8 {"N": 5}   2r  iwm  sign -    758.0       -0.141    -3.65  320.0      0.010   0.18       False
   F8 {"N": 5}   2r  spy  T1 low    498.0       -0.129    -2.60  205.0     -0.077  -1.04        True
   F8 {"N": 5}   2r  spy  T2 mid    498.0       -0.059    -1.29  167.0      0.089   1.15       False
   F8 {"N": 5}   2r  spy T3 high    499.0       -0.056    -1.22  211.0      0.194   2.71       False
   F8 {"N": 5}   2r  spy  sign +    777.0       -0.025    -0.69  312.0      0.170   2.95       False
   F8 {"N": 5}   2r  spy  sign -    718.0       -0.142    -3.52  271.0     -0.048  -0.75        True
   F8 {"N": 5} hold  brd  T1 low    456.0       -0.023    -0.26  123.0      0.234   1.48       False
   F8 {"N": 5} hold  brd  T2 mid    455.0       -0.063    -1.07  158.0     -0.073  -0.89        True
   F8 {"N": 5} hold  brd T3 high    455.0        0.002     0.02  252.0      0.125   1.54        True
   F8 {"N": 5} hold  iwm  T1 low    455.0       -0.134    -2.12  210.0     -0.047  -0.58        True
   F8 {"N": 5} hold  iwm  T2 mid    455.0        0.006     0.07  159.0      0.220   1.71        True
   F8 {"N": 5} hold  iwm T3 high    456.0        0.044     0.67  164.0      0.144   1.48        True
   F8 {"N": 5} hold  iwm  sign +    670.0        0.039     0.67  238.0      0.205   2.46        True
   F8 {"N": 5} hold  iwm  sign -    696.0       -0.093    -1.44  295.0      0.000   0.00       False
   F8 {"N": 5} hold  spy  T1 low    455.0        0.035     0.37  193.0     -0.074  -0.68       False
   F8 {"N": 5} hold  spy  T2 mid    455.0       -0.104    -1.91  153.0      0.082   0.86       False
   F8 {"N": 5} hold  spy T3 high    456.0       -0.015    -0.20  187.0      0.270   2.87       False
   F8 {"N": 5} hold  spy  sign +    709.0       -0.004    -0.07  280.0      0.222   2.98       False
   F8 {"N": 5} hold  spy  sign -    657.0       -0.054    -0.78  253.0     -0.053  -0.59        True

by feature, share of buckets whose TRAIN and VAL means share a sign:

      mean  size
feat            
brd   0.54    24
iwm   0.52    40
spy   0.48    40
