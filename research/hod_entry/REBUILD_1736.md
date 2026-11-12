# REBUILD 1,736 (independent, from PREREG prose; script 1736_rebuild.py)

Match on (day, symbol, entry_m): 8,245 arms, 0 unmatched.
Mismatches vs 1736_events.csv, strict reading (missing bar F+1 = no trade): event 0, touch_m 0, fail_m 0, half_spread 0, rails 27, entry/stop/R 92, E1 net R 65, E2 65 (all = 92 arms whose bar F+1 has no print; first file enters at the next available bar). With that next-bar fallback (1736_rebuild_events_nextbar.csv): ALL fields 0 mismatches. Stop leg read as half_spread + 13 bp (the only reading that matches).

E1 mean R (n), first file / rebuild next-bar (identical) / rebuild strict:
- ALL TRAIN -0.627 (386) / -0.627 (386) / -0.573 (356)
- ALL VAL -0.682 (550) / -0.682 (550) / -0.656 (515)
- ETB-now TRAIN -0.717 (193) / -0.717 (193) / -0.661 (177)
- ETB-now VAL -0.778 (299) / -0.778 (299) / -0.751 (280)

PREREG ARM CHECK FAILS: entry_m is NOT the bar before the break. On 400 random arms, 100 % had a bar with high >= level at or before entry_m (first breach a median 16 min earlier, never later than entry_m - 2); 17/20 rows fail the PREREG's own check. The "touch" is a re-touch after the break, so the event is not the failed first touch of the HOD. Spec issue, not a coding mismatch.
