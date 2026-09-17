# R5 cause attribution -- TEST, pre-book candidate divergences

reproduction of the divergence by the parameterised pipeline:
repro_A    0    1
repro_B    1    0
side             
A_only     0  339
B_only   395    0

first-gate reasons (A convention x B convention):
reason_B  cap  day_open5  late  no_next  no_signal  r_small  taken
reason_A                                                          
cap         0          0     0        0          0        0    174
few_bars    0          0     0        0          0        0      1
floor       0          0     0        0          0        0    217
r_small     0          0     0        0          0        0      3
taken     165         30    43        1         99        1      0

cause (set of single switch flips that change the verdict) -> trades -> booked net R:
  side                         flips   n  booked  net_R
B_only                    level_mult 309      76  -0.06
A_only                    level_mult 308      75  56.05
B_only                   none_single  34       9  -1.20
B_only                     floor_den  32      11  -8.08
B_only          level_mult|floor_den  20       1   3.15
A_only                     day_open5  19       5  -0.49
A_only            price_on|day_open5   5       2  -2.13
A_only          level_mult|day_open5   5       2  -0.65
A_only level_mult|price_on|day_open5   1       1  -0.12
A_only           level_mult|price_on   1       0   0.00
