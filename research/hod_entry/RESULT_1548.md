# RESULT 1,548-1,549 -- extension anatomy and the sweep entry

## Parameters (TRAIN-H2 extenders only, frozen BEFORE any VAL statistic)
d (limit below level)  = 0.9441%
s (stop below level)   = 1.9713%
W (resting window)     = 120.00 min

## Part A -- anatomy (extenders vs non-extenders, both holdouts, all rows and real-SIP)
- split=TRAIN real_sip_only=False group=extender n=865 dd%_q={0.1: 0.15445753325325096, 0.25: 0.43280182232345726, 0.5: 0.9441362916006293, 0.75: 1.9712793733681344, 0.9: 3.457638990698664} min_to_touch_q={0.1: 11.934904502400013, 0.25: 29.49645497885001, 0.5: 71.9129167443167, 0.75: 151.7042670285167, 0.9: 244.42598490163337} breach_before=0.237 stopped_before=0.230 exit_mix={'target': 0.711, 'stop': 0.2428, 'eod': 0.0358, 'stop_bar': 0.0081, 'eod_fallback': 0.0023}
- split=TRAIN real_sip_only=False group=non_extender n=518 dd%_q={0.1: 1.9441345231771445, 0.25: 3.5918617495246883, 0.5: 6.325103576644088, 0.75: 10.34906224237428, 0.9: 14.571776865205189} min_to_touch_q=None breach_before=0.876 stopped_before=0.844 exit_mix={'stop': 0.8398, 'eod': 0.0869, 'target': 0.0618, 'eod_fallback': 0.0077, 'stop_bar': 0.0039}
- split=TRAIN real_sip_only=True group=extender n=584 dd%_q={0.1: 0.16481259042511487, 0.25: 0.45160403422655016, 0.5: 1.0137935023756008, 0.75: 2.040181524253213, 0.9: 3.5471054054694027} min_to_touch_q={0.1: 12.98997146636832, 0.25: 31.74076370976252, 0.5: 80.30629212049996, 0.75: 156.91222394621252, 0.9: 246.97881736282} breach_before=0.243 stopped_before=0.238 exit_mix={'target': 0.7038, 'stop': 0.25, 'eod': 0.036, 'stop_bar': 0.0086, 'eod_fallback': 0.0017}
- split=TRAIN real_sip_only=True group=non_extender n=396 dd%_q={0.1: 2.0208911427829768, 0.25: 3.707032826397383, 0.5: 6.195656357464986, 0.75: 10.215755521764462, 0.9: 14.599210395671722} min_to_touch_q=None breach_before=0.871 stopped_before=0.843 exit_mix={'stop': 0.8409, 'eod': 0.0934, 'target': 0.0606, 'stop_bar': 0.0025, 'eod_fallback': 0.0025}
- split=VAL real_sip_only=False group=extender n=833 dd%_q={0.1: 0.269201382082097, 0.25: 0.5986394557823098, 0.5: 1.3032422123331218, 0.75: 2.514237397726991, 0.9: 4.234680304683724} min_to_touch_q={0.1: 13.279020342869966, 0.25: 26.248196045883333, 0.5: 62.85841751429996, 0.75: 144.95053377249997, 0.9: 251.91324920458354} breach_before=0.271 stopped_before=0.267 exit_mix={'target': 0.6471, 'stop': 0.2833, 'eod': 0.0624, 'eod_fallback': 0.0048, 'stop_bar': 0.0024}
- split=VAL real_sip_only=False group=non_extender n=1135 dd%_q={0.1: 1.4737578587155458, 0.25: 2.772851966008351, 0.5: 4.5425667090215995, 0.75: 6.941943924336264, 0.9: 10.23656826609321} min_to_touch_q=None breach_before=0.842 stopped_before=0.805 exit_mix={'stop': 0.7974, 'eod': 0.1057, 'target': 0.0846, 'stop_bar': 0.0079, 'eod_fallback': 0.0044}
- split=VAL real_sip_only=True group=extender n=589 dd%_q={0.1: 0.33633190487713155, 0.25: 0.6944444444444378, 0.5: 1.3377926421404738, 0.75: 2.514237397726991, 0.9: 4.165743105022385} min_to_touch_q={0.1: 14.840204602586619, 0.25: 29.640585987916666, 0.5: 64.9617531066167, 0.75: 145.9706695440167, 0.9: 250.49965805083346} breach_before=0.263 stopped_before=0.260 exit_mix={'target': 0.6486, 'stop': 0.2767, 'eod': 0.0679, 'eod_fallback': 0.0051, 'stop_bar': 0.0017}
- split=VAL real_sip_only=True group=non_extender n=879 dd%_q={0.1: 1.4868613564989848, 0.25: 2.7970655474661226, 0.5: 4.538653366583541, 0.75: 6.823127945645911, 0.9: 10.209488252662021} min_to_touch_q=None breach_before=0.845 stopped_before=0.811 exit_mix={'stop': 0.8055, 'eod': 0.1058, 'target': 0.0785, 'stop_bar': 0.0057, 'eod_fallback': 0.0046}

## Part B -- cell x population x holdout
{'cell': '1548_kept', 'n': 1026, 'fill_share': 0.6998635743519782, 'mean_R': 0.48660376384385085, 'mean_pct': 0.5045755706755429, 't': 4.089957305873892, 'ex_top5_R': 0.20934924968706792, 'winner_capped_R': -0.13362636161913174, 'fills_wk': 30.74074074074074, 'real_sip_n': 746, 'real_sip_mean_R': 0.46939962117063205, 'real_sip_t': 3.462283643944422, 'exit_mix': {'stop': 0.7329, 'target': 0.2115, 'eod': 0.0556}, 'median_R_pct_of_price': 1.036933143898695, 'split': 'TRAIN', 'cell_id': 1548, 'population': 'kept', 'null_pctile': 100.0, 'passes_bar': None}
{'cell': '1549_kept', 'n': 1466, 'fill_share': 1.0, 'mean_R': 0.4890611964177884, 'mean_pct': 0.9896224023802509, 't': 7.297022500685221, 'ex_top5_R': 0.3829095102696835, 'winner_capped_R': 0.48886138291286646, 'fills_wk': 34.333333333333336, 'real_sip_n': 1035, 'real_sip_mean_R': 0.4086251580177602, 'real_sip_t': 5.289152647997003, 'exit_mix': {'stop': 0.4966, 'target': 0.4379, 'eod': 0.0655}, 'median_R_pct_of_price': 2.032467655661402, 'split': 'TRAIN', 'cell_id': 1549, 'population': 'kept', 'null_pctile': 100.0, 'passes_bar': None}
{'cell': '1548_dropped', 'n': 2105, 'fill_share': 0.7179399727148704, 'mean_R': -0.3738156159354294, 'mean_pct': -0.38762180187035167, 't': -7.808718698860345, 'ex_top5_R': -0.63034666971099, 'winner_capped_R': -0.44994970656016287, 'fills_wk': 42.925925925925924, 'real_sip_n': 1835, 'real_sip_mean_R': -0.4312091314619348, 'real_sip_t': -8.639268334627197, 'exit_mix': {'stop': 0.7188, 'eod': 0.2646, 'target': 0.0152, 'eod_fallback': 0.0014}, 'median_R_pct_of_price': 1.036933143898695, 'split': 'TRAIN', 'cell_id': 1548, 'population': 'dropped'}
{'cell': '1549_dropped', 'n': 2932, 'fill_share': 1.0, 'mean_R': -0.3939971746884647, 'mean_pct': -0.7989418886025315, 't': -11.325074934992259, 'ex_top5_R': -0.5391661394513181, 'winner_capped_R': -0.3939971746884647, 'fills_wk': 35.81481481481482, 'real_sip_n': 2464, 'real_sip_mean_R': -0.47128590119475183, 'real_sip_t': -13.925879120098973, 'exit_mix': {'stop': 0.5464, 'eod': 0.4025, 'target': 0.0495, 'eod_fallback': 0.0017}, 'median_R_pct_of_price': 2.0197604419273016, 'split': 'TRAIN', 'cell_id': 1549, 'population': 'dropped'}
{'cell': '1548_kept', 'n': 1641, 'fill_share': 0.80126953125, 'mean_R': -0.02485103017529988, 'mean_pct': -0.025768856848795, 't': -0.2887732302031551, 'ex_top5_R': -0.3305447548532967, 'winner_capped_R': -0.4120407900768689, 'fills_wk': 48.40909090909091, 'real_sip_n': 1249, 'real_sip_mean_R': -0.02872948863029911, 'real_sip_t': -0.30551090125767466, 'exit_mix': {'stop': 0.7965, 'target': 0.1322, 'eod': 0.0713}, 'median_R_pct_of_price': 1.0369331438986953, 'split': 'VAL', 'cell_id': 1548, 'population': 'kept', 'null_pctile': 100.0, 'passes_bar': False}
{'cell': '1549_kept', 'n': 2048, 'fill_share': 1.0, 'mean_R': -0.11339567757367633, 'mean_pct': -0.23122885418135256, 't': -2.1406077666085737, 'ex_top5_R': -0.24992831097565904, 'winner_capped_R': -0.11353197942841459, 'fills_wk': 47.0, 'real_sip_n': 1522, 'real_sip_mean_R': -0.1573010909931216, 'real_sip_t': -2.9774601113390133, 'exit_mix': {'stop': 0.645, 'target': 0.2588, 'eod': 0.0962}, 'median_R_pct_of_price': 2.0359003266294047, 'split': 'VAL', 'cell_id': 1549, 'population': 'kept', 'null_pctile': 0.0, 'passes_bar': False}
{'cell': '1548_dropped', 'n': 2318, 'fill_share': 0.668975468975469, 'mean_R': -0.08019191256724965, 'mean_pct': -0.08315365201360729, 't': -1.3188229307445507, 'ex_top5_R': -0.3891299716231757, 'winner_capped_R': -0.25412413778352205, 'fills_wk': 45.81818181818182, 'real_sip_n': 2036, 'real_sip_mean_R': -0.19993839222895804, 'real_sip_t': -3.367929105861221, 'exit_mix': {'stop': 0.6833, 'eod': 0.267, 'target': 0.0496}, 'median_R_pct_of_price': 1.036933143898695, 'split': 'VAL', 'cell_id': 1548, 'population': 'dropped'}
{'cell': '1549_dropped', 'n': 3465, 'fill_share': 1.0, 'mean_R': -0.14488388493612556, 'mean_pct': -0.29264981662595646, 't': -3.4794359852669423, 'ex_top5_R': -0.2825621268094138, 'winner_capped_R': -0.14488388493612556, 'fills_wk': 37.54545454545455, 'real_sip_n': 2962, 'real_sip_mean_R': -0.23794297649741925, 'real_sip_t': -5.504998627518262, 'exit_mix': {'stop': 0.484, 'eod': 0.3986, 'target': 0.1175}, 'median_R_pct_of_price': 2.0174105360159693, 'split': 'VAL', 'cell_id': 1549, 'population': 'dropped'}

## Caveats
- d/s/W computed on TRAIN-H2 extenders only, before any VAL statistic (see log order).
- SWEEP mean/t/ex-top5/etc. are computed over FILLED trades only; unfilled sweeps are reported in cell_1548_fills.csv with their base outcome_R but do not enter the mean.
- Missing-bar / no-touch-found / invalid-R rows are excluded and counted in the run log, never imputed.
- Real-SIP subset = store_served_1438==0 (features_1478_A / predictions.csv).

## Pass-bar checklist (VAL, per cell)
### cell 1548 -- FAIL
  [ ] VAL mean_R >= +0.15
  [ ] VAL mean_pct >= +0.15%
  [ ] VAL t >= 2.5
  [ ] VAL ex-top-5% > 0
  [ ] VAL winner-capped > 0
  [x] VAL fills/wk >= 3
  [x] VAL null percentile >= 99
  [ ] VAL real-SIP mean >= +0.10R, t>=2
  [x] VAL median R >= 0.5% of price (rail)
  [ ] TRAIN-H2 same sign, t>=1
  [x] kept > dropped (VAL)
### cell 1549 -- FAIL
  [ ] VAL mean_R >= +0.15
  [ ] VAL mean_pct >= +0.15%
  [ ] VAL t >= 2.5
  [ ] VAL ex-top-5% > 0
  [ ] VAL winner-capped > 0
  [x] VAL fills/wk >= 3
  [ ] VAL null percentile >= 99
  [ ] VAL real-SIP mean >= +0.10R, t>=2
  [x] VAL median R >= 0.5% of price (rail)
  [ ] TRAIN-H2 same sign, t>=1
  [x] kept > dropped (VAL)

## Judge (main session, 2026-09-26 19:25 UTC) — FAIL both cells; the extension is predictable in-sample only

* Anatomy (VAL kept set): extenders dip a median 1.3 % under the level (p75 2.5 %) and touch +5 % after a median 63 min;
  27 % breach the consolidation low first. Non-extenders dip a median 4.5 %, 84 % breach the low, 81 % are stopped.
  The shapes differ because the dip IS the outcome; the model cannot tell them apart well enough (precision 44.6 %).
* 1,548 SWEEP (bid at level × (1 − 0.94 %), stop −1.97 %, target +5 %, W 120 min): VAL −0.025 R (rebuild −0.006), t −0.3;
  the look-ahead refuter found the sweep bid filling inside the fill bar on prints from BEFORE the break (the population
  exists because that break happened); removing those fills → −0.19 R (t −2.3). 1,549 WIDE: VAL −0.11 R (rebuild −0.09).
  TRAIN-H2 +0.49 R on both is doubly in-sample (the model and d, s, W are fitted there; in-sample AUC 0.92 vs 0.715).
* Wording corrected per the statistics refuter: NOT "regime, not edge" — "in-sample only, negative out of sample". The
  VAL kept-minus-dropped gap (+0.06 / +0.03 R) is not significant (P 0.22); positive only on real-SIP rows, reversed
  on cache-only rows. Every VAL cut (real-SIP, cache-only, capped book, ex-top-5 %, ex-top-1 %, best-2-days dropped,
  winner-capped) is ≤ 0.
* Builder vs rebuild: fill-set Jaccard 0.97; d, s differ 8 % (the builder excludes 4.6 % no-touch extenders); row-level
  net R agreement 31 % (binary stop/target flips from the s gap) — irrelevant to a verdict where no cell is within
  0.15 R of the bar under either implementation.
Consequence per PREREG: the extension predictor is closed as a money signal on this population — predictable at the
arm bar (AUC 0.715), not tradable. Programme count 1,549.
