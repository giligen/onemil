# Per-tier lever isolation (10%-frame, conv>=1.4 baseline)

Each row = single knob change applied to ONE tier only. Shows Δ vs baseline on TRAIN+VAL and HOLDOUT Q1 2026 independently.

## Baseline reference

- TRAIN (2025 Jan-Jul) — A: n=77, PnL=$+41,877
- TRAIN (2025 Jan-Jul) — E: n=170, PnL=$-10,364
- TRAIN (2025 Jan-Jul) — total: n=252, PnL=$+36,151
- VAL (2025 Aug-Dec) — A: n=46, PnL=$+10,475
- VAL (2025 Aug-Dec) — E: n=141, PnL=$+4,659
- VAL (2025 Aug-Dec) — total: n=192, PnL=$+15,429
- HOQ1 (2026 Q1) — A: n=48, PnL=$-5,130
- HOQ1 (2026 Q1) — E: n=131, PnL=$+14,637
- HOQ1 (2026 Q1) — total: n=181, PnL=$+9,357

## A. MACD strong multiplier sweep per tier

Current strong_mult=1.5 for macd>0.5%. What if we change it?


### Target tier: A

- **A: macd_strong 0.5**: T+V Δ $+214, HOQ1 Δ $-798, **grand $-585**
- **A: macd_strong 0.75**: T+V Δ $+160, HOQ1 Δ $-599, **grand $-439**
- **A: macd_strong 1.0**: T+V Δ $+107, HOQ1 Δ $-399, **grand $-292**
- **A: macd_strong 1.25**: T+V Δ $+53, HOQ1 Δ $-200, **grand $-146**
- **A: macd_strong 1.5**: T+V Δ $+0, HOQ1 Δ $+0, **grand $+0**
- **A: macd_strong 1.8**: T+V Δ $-64, HOQ1 Δ $+240, **grand $+175**
- **A: macd_strong 2.0**: T+V Δ $-107, HOQ1 Δ $+399, **grand $+292**
- **A: macd_strong 2.5**: T+V Δ $-214, HOQ1 Δ $+798, **grand $+585**

### Target tier: E

- **E: macd_strong 0.5**: T+V Δ $+871, HOQ1 Δ $-9,082, **grand $-8,211**
- **E: macd_strong 0.75**: T+V Δ $+1,228, HOQ1 Δ $-6,766, **grand $-5,538**
- **E: macd_strong 1.0**: T+V Δ $+1,586, HOQ1 Δ $-4,451, **grand $-2,866**
- **E: macd_strong 1.25**: T+V Δ $+1,943, HOQ1 Δ $-2,136, **grand $-193**
- **E: macd_strong 1.5**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: macd_strong 1.8**: T+V Δ $+2,729, HOQ1 Δ $+2,957, **grand $+5,686**
- **E: macd_strong 2.0**: T+V Δ $+3,015, HOQ1 Δ $+4,809, **grand $+7,824**
- **E: macd_strong 2.5**: T+V Δ $+3,730, HOQ1 Δ $+9,439, **grand $+13,169**

## B. MACD normal multiplier sweep per tier

Current normal_mult=1.0. Drop it to downsize the neutral-zone trades, especially E-tier MACD 1.0 bucket (the $-14,734 loser).


### Target tier: A

- **A: macd_normal 0.0**: T+V Δ $-8,545, HOQ1 Δ $+2,092, **grand $-6,453**
- **A: macd_normal 0.25**: T+V Δ $-6,408, HOQ1 Δ $+1,569, **grand $-4,839**
- **A: macd_normal 0.5**: T+V Δ $-4,272, HOQ1 Δ $+1,046, **grand $-3,226**
- **A: macd_normal 0.75**: T+V Δ $-2,136, HOQ1 Δ $+523, **grand $-1,613**
- **A: macd_normal 1.0**: T+V Δ $+0, HOQ1 Δ $+0, **grand $+0**
- **A: macd_normal 1.25**: T+V Δ $+2,136, HOQ1 Δ $-523, **grand $+1,613**

### Target tier: E

- **E: macd_normal 0.0**: T+V Δ $+13,994, HOQ1 Δ $+3,219, **grand $+17,213**
- **E: macd_normal 0.25**: T+V Δ $+11,071, HOQ1 Δ $+2,459, **grand $+13,529**
- **E: macd_normal 0.5**: T+V Δ $+8,147, HOQ1 Δ $+1,699, **grand $+9,846**
- **E: macd_normal 0.75**: T+V Δ $+5,224, HOQ1 Δ $+939, **grand $+6,163**
- **E: macd_normal 1.0**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: macd_normal 1.25**: T+V Δ $-623, HOQ1 Δ $-581, **grand $-1,204**

## C. Rule weight drops per tier

Zero out each rule per tier, see if PnL improves (= rule was noise/counter-signal in that tier).


### Target tier: A

- **A: drop r1 (0.0)**: T+V Δ $-2,870, HOQ1 Δ $+817, **grand $-2,052**
- **A: drop r3 (0.0)**: T+V Δ $-7,529, HOQ1 Δ $+3,115, **grand $-4,413**
- **A: drop r5 (0.0)**: T+V Δ $-2,755, HOQ1 Δ $-39, **grand $-2,794**
- **A: drop r7 (0.0)**: T+V Δ $-7,421, HOQ1 Δ $-318, **grand $-7,739**
- **A: drop r4p (0.0)**: T+V Δ $-8,912, HOQ1 Δ $+1,259, **grand $-7,653**
- **A: drop r4n (0.0)**: T+V Δ $-454, HOQ1 Δ $-785, **grand $-1,240**
- **A: drop r2p (0.0)**: T+V Δ $-3,389, HOQ1 Δ $+1,668, **grand $-1,722**
- **A: drop r2n (0.0)**: T+V Δ $-1,428, HOQ1 Δ $+735, **grand $-693**

### Target tier: E

- **E: drop r1 (0.0)**: T+V Δ $-2,721, HOQ1 Δ $-3,533, **grand $-6,254**
- **E: drop r3 (0.0)**: T+V Δ $+3,047, HOQ1 Δ $+1,806, **grand $+4,852**
- **E: drop r5 (0.0)**: T+V Δ $-6,896, HOQ1 Δ $-2,755, **grand $-9,651**
- **E: drop r7 (0.0)**: T+V Δ $-22, HOQ1 Δ $+298, **grand $+277**
- **E: drop r4p (0.0)**: T+V Δ $+4,370, HOQ1 Δ $-4,406, **grand $-37**
- **E: drop r4n (0.0)**: T+V Δ $+4,981, HOQ1 Δ $+213, **grand $+5,195**
- **E: drop r2p (0.0)**: T+V Δ $-4,299, HOQ1 Δ $-715, **grand $-5,014**
- **E: drop r2n (0.0)**: T+V Δ $+3,276, HOQ1 Δ $+1,749, **grand $+5,024**

## D. V-reversal bonus sweep (A-tier only — r9 doesn't fire in E)

- **A: v_rev_bonus 0.2**: T+V Δ $-3,501, HOQ1 Δ $-2,135, **grand $-5,636**
- **A: v_rev_bonus 0.4**: T+V Δ $+0, HOQ1 Δ $+0, **grand $+0**
- **A: v_rev_bonus 0.5**: T+V Δ $+540, HOQ1 Δ $-88, **grand $+452**
- **A: v_rev_bonus 0.6**: T+V Δ $+1,923, HOQ1 Δ $+134, **grand $+2,057**
- **A: v_rev_bonus 0.7**: T+V Δ $+2,301, HOQ1 Δ $+356, **grand $+2,656**
- **A: v_rev_bonus 0.8**: T+V Δ $+3,139, HOQ1 Δ $+578, **grand $+3,717**
- **A: v_rev_bonus 1.0**: T+V Δ $+4,823, HOQ1 Δ $+1,022, **grand $+5,845**
- **A: v_rev_bonus 1.2**: T+V Δ $+6,541, HOQ1 Δ $+1,459, **grand $+8,000**

## E. Conv threshold per tier

Current threshold 1.4. Lower = more trades pass. Higher = stricter.


### Target tier: A

- **A: conv_threshold 1.0**: T+V Δ $-1,293, HOQ1 Δ $-311, **grand $-1,603**
- **A: conv_threshold 1.2**: T+V Δ $-843, HOQ1 Δ $-311, **grand $-1,153**
- **A: conv_threshold 1.3**: T+V Δ $+0, HOQ1 Δ $+0, **grand $+0**
- **A: conv_threshold 1.4**: T+V Δ $+0, HOQ1 Δ $+0, **grand $+0**
- **A: conv_threshold 1.5**: T+V Δ $-1,583, HOQ1 Δ $-2,811, **grand $-4,394**
- **A: conv_threshold 1.6**: T+V Δ $-4,196, HOQ1 Δ $-1,170, **grand $-5,366**
- **A: conv_threshold 1.7**: T+V Δ $-945, HOQ1 Δ $+628, **grand $-317**

### Target tier: E

- **E: conv_threshold 1.0**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: conv_threshold 1.2**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: conv_threshold 1.3**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: conv_threshold 1.4**: T+V Δ $+2,301, HOQ1 Δ $+179, **grand $+2,479**
- **E: conv_threshold 1.5**: T+V Δ $+76, HOQ1 Δ $-18, **grand $+59**
- **E: conv_threshold 1.6**: T+V Δ $-1,702, HOQ1 Δ $+357, **grand $-1,345**
- **E: conv_threshold 1.7**: T+V Δ $-2,940, HOQ1 Δ $-964, **grand $-3,904**
