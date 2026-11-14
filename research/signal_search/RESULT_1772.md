# RESULT 1,772 — overnight leg of attention gappers (MOC buy D, MOO sell D+1): FAIL

Population: 7,487 symbol-nights (TRAIN 3,800 / VAL 3,687); gap = open_D/close_{D-1}-1, median 7.7 % (P10 5.4, P90 24 %; one scale outlier max 26,796 % — flagged, not adjusted). D+1 open coverage 99.97 % given a D row (TRAIN 92.4 %, VAL 99.5 % overall); winner/loser gap 0.1 pp: availability rail MET.
Executable (capacity) share 57.1 % TRAIN / 56.0 % VAL (< 60 % bar). Executable n: TRAIN 2,005 (128 nights), VAL 2,054 (102 nights), median 13-14 names/night.

PRIMARY (VAL, executable, excess vs SPY, gross): -37.9 bp, night-t -1.7; net at 10 bp/side: -57.9 bp (t -2.6). Ex-top-5 % -125.7; winner-capped -59.0; worst-1 % mean -2,645; weekly P10 -1,049 bp. TRAIN gross +24.3 (t 0.4); TRAIN halves +12.8 / +32.5 gross but -7.2 / +12.5 net at 10 bp: not same-signed positive. Non-executable VAL names +148 bp (t 1.6), TRAIN +17 — sign unstable.
PLACEBOS (gross VAL p95): random universe 87 bp (mean 6.5), same-name +5d 137 bp (mean +58). Primary beats neither.
(a) fade vs held, gross: TRAIN fade -19 / held +91; VAL fade -48 (t -1.8) / held -22 — the reversal across splits kills the held-cohort read. Tier $1-5 negative in both splits (-66/-101); high-gap tercile negative in both (-129/-115, t -3.4/-2.9). Short mirror (report-only) VAL +37 gross (t 1.7), TRAIN -25: no.
INDEX FRAME (SPY close->open vs open->close, mean bp/day 2016-26): overnight +3.1, intraday +2.8; QQQ +4.7 / +4.1; per-year signs flip (2022 on -6.1; 2024 on +9.0, Sharpe 2.7).
VERDICT: FAIL on net >= +30, t >= 2, halves, ex-top-5, both placebos, executable share. Gapper overnight is noise around zero with heavy left tails (-27 % worst-1 % average).
CHILDREN: (1) D+1 intraday continuation of the day-D fade cohort (open_N to close_N) for the owner list; (2) week-ahead overnight persistence (D+1..D+5 summed overnights) with the same-name placebo, tail-trimmed; (3) high-gap/low-price left-tail screen as a veto input (high-gap tercile -115/-129 bp both splits).
