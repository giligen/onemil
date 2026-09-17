# J/verify_rows.py — source `research/fuckup_audit/J/candidates_u3_TRAIN.csv`

rows 4,859,792 | days 240 (2025-01-17..2025-12-31) | filled 4,719,541

## A. hand recomputation of 8 filled rows (reservoir, seed 20260917)

  2025-09-25 ACMR S5 {} side=S bars=387
      OK  sig_m          hand=575                    file=575
      OK  level          hand=37.56999969482422      file=37.57
      OK  stop           hand=37.77000045776367      file=37.77
      OK  entry          hand=37.56999969482422      file=37.57
      OK  entry_m        hand=575                    file=575.0
      OK  r_pct          hand=0.5323416677243304     file=0.53234167
      OK  dv5            hand=1611663.3981208801     file=1611663.4
      OK  dv4prior       hand=1305144.8818397522     file=1305144.9
      OK  n_bars_in_5    hand=5                      file=5.0
      OK  rr_hold        hand=-1.1888492818859704    file=-1.1888493
      OK  rr_2r          hand=-1.1888492818859704    file=-1.1888493
      OK  rr_pp          hand=-1.1888492818859704    file=-1.1888493
      OK  mae_pct        hand=0.6388120319837542     file=0.63881203
      OK  mfe_r          hand=0.0                    file=0.0
      OK  why_hold       hand=stop                   file=stop
      OK  why_2r         hand=stop                   file=stop
      OK  why_pp         hand=stop                   file=stop
      OK  exit_m_hold    hand=576                    file=576.0
      OK  exit_m_2r      hand=576                    file=576.0
      OK  exit_m_pp      hand=576                    file=576.0
      -> all fields reproduced

  2025-04-08 BANC S5 {} side=S bars=390
      OK  sig_m          hand=575                    file=575
      OK  level          hand=12.704999923706055     file=12.705
      OK  stop           hand=12.710000038146973     file=12.71
      OK  entry          hand=12.704999923706055     file=12.705
      OK  entry_m        hand=575                    file=575.0
      OK  r_pct          hand=0.039355485800423626   file=0.039355486
      OK  dv5            hand=662394.6419811249      file=662394.64
      OK  dv4prior       hand=467195.1166791916      file=467195.12
      OK  n_bars_in_5    hand=5                      file=5.0
      OK  rr_hold        hand=-3.541941827197916     file=-3.5419418
      OK  rr_2r          hand=2.0                    file=2.0
      OK  rr_pp          hand=-0.27047091359888986   file=-0.27047091
      OK  mae_pct        hand=0.0                    file=0.0
      OK  mfe_r          hand=16.999618539004388     file=16.999619
      OK  why_hold       hand=stop                   file=stop
      OK  why_2r         hand=target                 file=target
      OK  why_pp         hand=pp+stop                file=pp+stop
      OK  exit_m_hold    hand=579                    file=579.0
      OK  exit_m_2r      hand=576                    file=576.0
      OK  exit_m_pp      hand=579                    file=579.0
      -> all fields reproduced

  2025-04-22 ETH F9 {"G": 0.03} side=L bars=373
      OK  sig_m          hand=575                    file=575
      OK  level          hand=15.42                  file=15.42
      OK  stop           hand=15.335                 file=15.335
      OK  entry          hand=15.415                 file=15.415
      OK  entry_m        hand=576                    file=576.0
      OK  r_pct          hand=0.5189750243269432     file=0.51897502
      OK  dv5            hand=559531.8569            file=559531.86
      OK  dv4prior       hand=488368.4569            file=488368.46
      OK  n_bars_in_5    hand=5                      file=5.0
      OK  rr_hold        hand=7.062500000000167      file=7.0625
      OK  rr_2r          hand=2.0                    file=2.0
      OK  rr_pp          hand=4.5312500000000835     file=4.53125
      OK  mae_pct        hand=0.16217969510216398    file=0.1621797
      OK  mfe_r          hand=4.500000000000111      file=4.5
      OK  why_hold       hand=eod                    file=eod
      OK  why_2r         hand=target                 file=target
      OK  why_pp         hand=pp+eod                 file=pp+eod
      OK  exit_m_hold    hand=955                    file=955.0
      OK  exit_m_2r      hand=644                    file=644.0
      OK  exit_m_pp      hand=955                    file=955.0
      -> all fields reproduced

  2025-07-29 SCS S2 {"N": 15} side=S bars=282
      OK  sig_m          hand=599                    file=599
      OK  level          hand=10.845000267028809     file=10.845
      OK  stop           hand=10.9350004196167       file=10.935
      OK  entry          hand=10.845000267028809     file=10.845
      OK  entry_m        hand=600                    file=600.0
      OK  r_pct          hand=0.8298769052271112     file=0.82987691
      OK  dv5            hand=119175.56410217285     file=119175.56
      OK  dv4prior       hand=100251.03863620758     file=100251.04
      OK  n_bars_in_5    hand=5                      file=5.0
      OK  rr_hold        hand=3.8888865341414824     file=3.8888865
      OK  rr_2r          hand=2.0                    file=2.0
      OK  rr_pp          hand=2.944443267070741      file=2.9444433
      OK  mae_pct        hand=0.5993506572877996     file=0.59935066
      OK  mfe_r          hand=2.0                    file=2.0
      OK  why_hold       hand=eod                    file=eod
      OK  why_2r         hand=target                 file=target
      OK  why_pp         hand=pp+eod                 file=pp+eod
      OK  exit_m_hold    hand=955                    file=955.0
      OK  exit_m_2r      hand=679                    file=679.0
      OK  exit_m_pp      hand=955                    file=955.0
      -> all fields reproduced

  2025-07-08 IRT S5 {} side=S bars=370
      OK  sig_m          hand=575                    file=575
      OK  level          hand=17.530000686645508     file=17.530001
      OK  stop           hand=17.639999389648438     file=17.639999
      OK  entry          hand=17.530000686645508     file=17.530001
      OK  entry_m        hand=575                    file=575.0
      OK  r_pct          hand=0.6274882983132315     file=0.6274883
      OK  dv5            hand=95822.62762260437      file=95822.628
      OK  dv4prior       hand=45833.01087760925      file=45833.011
      OK  n_bars_in_5    hand=4                      file=4.0
      OK  rr_hold        hand=-1.160365521665993     file=-1.1603655
      OK  rr_2r          hand=-1.160365521665993     file=-1.1603655
      OK  rr_pp          hand=-1.160365521665993     file=-1.1603655
      OK  mae_pct        hand=0.7130632920923157     file=0.71306329
      OK  mfe_r          hand=0.6363857051204245     file=0.63638571
      OK  why_hold       hand=stop                   file=stop
      OK  why_2r         hand=stop                   file=stop
      OK  why_pp         hand=stop                   file=stop
      OK  exit_m_hold    hand=637                    file=637.0
      OK  exit_m_2r      hand=637                    file=637.0
      OK  exit_m_pp      hand=637                    file=637.0
      -> all fields reproduced

  2025-07-16 MLAB S2 {"N": 30} side=S bars=106
      OK  sig_m          hand=648                    file=648
      OK  level          hand=77.41                  file=77.41
      OK  stop           hand=80.84                  file=80.84
      OK  entry          hand=78.79                  file=78.79
      OK  entry_m        hand=649                    file=649.0
      OK  r_pct          hand=2.6018530270338838     file=2.601853
      OK  dv5            hand=129984.3               file=129984.3
      OK  dv4prior       hand=116274.84              file=116274.84
      OK  n_bars_in_5    hand=2                      file=2.0
      OK  rr_hold        hand=-0.0926829268292673    file=-0.092682927
      OK  rr_2r          hand=-0.0926829268292673    file=-0.092682927
      OK  rr_pp          hand=-0.0926829268292673    file=-0.092682927
      OK  mae_pct        hand=1.294580530524173      file=1.2945805
      OK  mfe_r          hand=1.1853658536585416     file=1.1853659
      OK  why_hold       hand=eod                    file=eod
      OK  why_2r         hand=eod                    file=eod
      OK  why_pp         hand=eod                    file=eod
      OK  exit_m_hold    hand=955                    file=955.0
      OK  exit_m_2r      hand=955                    file=955.0
      OK  exit_m_pp      hand=955                    file=955.0
      -> all fields reproduced

  2025-01-31 SUM S5 {} side=S bars=358
      OK  sig_m          hand=575                    file=575
      OK  level          hand=52.349998474121094     file=52.349998
      OK  stop           hand=52.38999938964844      file=52.389999
      OK  entry          hand=52.349998474121094     file=52.349998
      OK  entry_m        hand=575                    file=575.0
      OK  r_pct          hand=0.07641053809603826    file=0.076410538
      OK  dv5            hand=3864770.0345344543     file=3864770.0
      OK  dv4prior       hand=2398919.0451774597     file=2398919.0
      OK  n_bars_in_5    hand=5                      file=5.0
      OK  rr_hold        hand=1.4999046347510967     file=1.4999046
      OK  rr_2r          hand=1.4999046347510967     file=1.4999046
      OK  rr_pp          hand=1.4999046347510967     file=1.4999046
      OK  mae_pct        hand=0.03820526904801913    file=0.038205269
      OK  mfe_r          hand=4.12483311081442       file=4.1248331
      OK  why_hold       hand=eod                    file=eod
      OK  why_2r         hand=eod                    file=eod
      OK  why_pp         hand=eod                    file=eod
      OK  exit_m_hold    hand=955                    file=955.0
      OK  exit_m_2r      hand=955                    file=955.0
      OK  exit_m_pp      hand=955                    file=955.0
      -> all fields reproduced

  2025-03-25 PTLO S2 {"N": 30} side=S bars=298
      OK  sig_m          hand=616                    file=616
      OK  level          hand=12.725000381469727     file=12.725
      OK  stop           hand=12.932700157165527     file=12.9327
      OK  entry          hand=12.699999809265137     file=12.7
      OK  entry_m        hand=617                    file=617.0
      OK  r_pct          hand=1.832286231458262      file=1.8322862
      OK  dv5            hand=120231.51774787903     file=120231.52
      OK  dv4prior       hand=110394.69786643982     file=110394.7
      OK  n_bars_in_5    hand=4                      file=4.0
      OK  rr_hold        hand=0.8164989098539368     file=0.81649891
      OK  rr_2r          hand=0.8164989098539368     file=0.81649891
      OK  rr_pp          hand=0.8164989098539368     file=0.81649891
      OK  mae_pct        hand=1.6535436322902113     file=1.6535436
      OK  mfe_r          hand=1.020626711037524      file=1.0206267
      OK  why_hold       hand=eod                    file=eod
      OK  why_2r         hand=eod                    file=eod
      OK  why_pp         hand=eod                    file=eod
      OK  exit_m_hold    hand=955                    file=955.0
      OK  exit_m_2r      hand=955                    file=955.0
      OK  exit_m_pp      hand=955                    file=955.0
      -> all fields reproduced

## B. obtainability
  rows 4,859,792 | filled 4,719,541 (0.9711) | fside_entry == 1 on filled rows: 1.000000
  fill rate by family (scan=first):
                                   n  filled  fill_rate
side fam cfg                                           
L    F10 {}                   461291  454248     0.9847
     F11 {"base": "F6"}       180925  164680     0.9102
     F14 {"N": 15}             45419   44249     0.9742
     F5  {"K": 5, "X": 0.04}  593251  569432     0.9599
     F6  {}                   188555  173448     0.9199
     F8  {"N": 15}            518502  508068     0.9799
         {"N": 30}            444511  437106     0.9833
         {"N": 5}             607959  584711     0.9618
     F9  {"G": 0.03}           15497   13631     0.8796
S    S1  {}                    18075   16239     0.8984
     S2  {"N": 15}            507761  498242     0.9813
         {"N": 30}            431227  424785     0.9851
     S3  {}                   213422  197305     0.9245
     S5  {}                   572520  572520     1.0000

  ADDENDUM 3 — scan rule rows: {'first': 4798915, 'keep': 60877}; a `keep` row exists only where `first` did not fill, and every one of them must be filled:
    keep rows 60,877 | filled 60,877 | by family {'F8 {"N": 5}': 19417, 'F11 {"base": "F6"}': 13482, 'F6 {}': 12609, 'F8 {"N": 15}': 8536, 'F8 {"N": 30}': 5954, 'F14 {"N": 15}': 879}

  ADDENDUM 1 — univ_flag rows: {'ok': 4712166, 'no_daily_bars': 147626} (no_daily_bars rows are kept in the file and excluded by the scorer; test tickers are never built)
    test tickers present in the file (must be none): []

## C. tape + detector parity vs E/candidates_causal.csv (the same detectors, a different store order)
  parity window: the first 40 built days (2025-01-17..2025-03-17), 272,464 J rows
  E rows on J days: 44,396
  F8: n=16,458 | sig_m differ 0 | level differ 0 | stop differ 0 | entry differ 0 | entry_m differ 0
  F6: n=1,874 | sig_m differ 963 | level differ 1874 | stop differ 195 | entry differ 895 | entry_m differ 1150
      F6 level carries the ADDENDUM-2 x1.003 buffer: rows where J level != E level x 1.003: 0
      F6 stop is the engine convention (running low THROUGH the signal bar): J stop <= E stop on 1874/1874 rows; strictly lower on 195
      (F6 sig_m may differ from Stage E BECAUSE of the 1.003 buffer — a higher level is reached later or not at all; it is reported, not failed)

## verdict
  all checks passed
