#!/usr/bin/env python3
"""Multi-window ORB book builder (research/orb_multiwindow/PREREG.md).

Replays the shipped ORB slot+veto mechanics across one or more opening-range
windows W, from the per-window ranked dumps the pipeline writes with
`ORB_BT_DUMP_RANKED` (post-threshold, post-Q1, pre-slot, per-day `_rank`).

The PRE-REGISTERED rule, per trading day:

  * windows are processed in ASCENDING W order (5 → 15 → 30);
  * inside a window, candidates are taken in the pipeline's own rank order
    with the pipeline's family / super-group dedup;
  * **overlap**: a symbol already ORDERED by an earlier window (filled OR
    resting — the book cannot tell the difference at submit time) is skipped
    by every later window. No refill: the later window moves on to its next
    candidate, it does not get a free slot back;
  * slots are SHARED — `--slots` total across all windows for the day, spent
    in the order picks are taken (the earlier window has priority);
  * the shipped no-refill vetoes (PDR → G1 → range-size → catalyst) are then
    applied to the taken list; a vetoed pick's slot stays spent.

With a single window this reproduces `study_orb_pipeline_static_lock.py`'s
book exactly — that identity is the script's parity test (`--parity BOOK.csv`).

Family / super-group dedup is applied WITHIN a window only: the pre-registered
overlap rule is symbol-level. The count of cross-window family collisions is
reported so the omission is visible.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from trading.orb_csv import read_orb_csv
from study_orb_correlation_filter import symbol_family, symbol_super_group
from trading.orb_pdr_veto import pdr_veto_applies
from trading.orb_g1_veto import g1_reject
from trading.orb_range_size_veto import range_size_veto_applies
from trading.orb_catalyst_veto import (
    DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies)
from trading.orb_asset_class import (
    DEFAULT_CLASS_MAP, load_class_map, underlying_anchor)


def load_news_map() -> Dict[Tuple[str, str], bool]:
    """RAW own-ticker premarket news, tri-state via absence (fail-open)."""
    out: Dict[Tuple[str, str], bool] = {}
    for p in sorted(glob.glob('data/research/orb_news_catalyst_*.csv')):
        for _, r in read_orb_csv(p).iterrows():
            out[(r['symbol'], r['day'])] = (r['n_articles'] or 0) > 0
    return out


def anchors_for(symbols) -> Dict[str, Optional[str]]:
    names: Dict[str, str] = {}
    try:
        with open(DEFAULT_CLASS_MAP, newline='') as fh:
            for row in csv.DictReader(fh):
                names[row['symbol']] = row.get('name', '')
    except OSError as e:
        print(f"WARNING: class map unavailable ({e}) — anchors fall back to "
              f"the family sets (fail-open, same as the pipeline)")
    cmap = load_class_map()
    return {s: underlying_anchor(s, names.get(s), cmap) for s in set(symbols)}


class Window:
    """One opening-range width and everything the book needs from it."""

    def __init__(self, w: int, ranked_csv: str, features_csv: str,
                 rs_min: float):
        self.w = w
        self.rs_min = rs_min
        self.ranked = read_orb_csv(ranked_csv)
        self.ranked['date'] = pd.to_datetime(self.ranked['date'])
        self.ranked = self.ranked.sort_values(['date', '_rank'])
        feats = read_orb_csv(features_csv)
        feats['date'] = pd.to_datetime(feats['date'])
        # Catalyst cohort = the FULL candidate universe this window saw that
        # morning (what the live engine would count at 09:30+W).
        anchors = anchors_for(feats['symbol'])
        day = feats['date'].dt.strftime('%Y-%m-%d')
        self.cohorts = {d: anchor_cohort_counts(g.map(anchors))
                        for d, g in feats['symbol'].groupby(day)}
        self.anchors = anchors
        self.by_day = {d: g for d, g in self.ranked.groupby('date', sort=True)}


def build(windows: List[Window], slots: int, news: Dict) -> pd.DataFrame:
    days = sorted({d for w in windows for d in w.by_day})
    taken_rows: List[Dict] = []
    n_overlap_skips = 0
    n_fam_cross = 0
    veto_counts = defaultdict(int)
    for day in days:
        taken: List[Tuple[Window, pd.Series]] = []
        seen_sym = set()
        cross_fam = set()
        for win in windows:
            g = win.by_day.get(day)
            if g is None:
                continue
            seen_fam, seen_sup = set(), set()
            for _, r in g.iterrows():
                if len(taken) >= slots:
                    break
                sym = r['symbol']
                if sym in seen_sym:
                    n_overlap_skips += 1
                    continue
                fam = symbol_family(sym)
                sup = symbol_super_group(sym)
                if fam and fam in seen_fam:
                    continue
                if sup and sup in seen_sup:
                    continue
                if fam:
                    seen_fam.add(fam)
                if sup:
                    seen_sup.add(sup)
                if fam and fam in cross_fam:
                    n_fam_cross += 1
                if fam:
                    cross_fam.add(fam)
                taken.append((win, r))
                seen_sym.add(sym)
            if len(taken) >= slots:
                break
        # --- shipped no-refill vetoes, in pipeline order -------------------
        dstr = pd.Timestamp(day).strftime('%Y-%m-%d')
        for win, r in taken:
            reason = None
            pdr = r.get('prev_day_range_pct')
            if pdr_veto_applies(None if pd.isna(pdr) else float(pdr), PDR_MIN):
                reason = 'pdr'
            elif g1_reject(r.get('return_volatility_20d'), pdr,
                           G1_RV_MIN, G1_PDR_MIN,
                           short_history_veto=G1_SHORT_HIST) is not None:
                reason = 'g1'
            elif range_size_veto_applies(r.get('range_size_pct'), win.rs_min):
                reason = 'range_size'
            else:
                anchor = win.anchors.get(r['symbol'])
                if catalyst_veto_applies(news.get((r['symbol'], dstr)), anchor,
                                         win.cohorts.get(dstr, {}),
                                         DEFAULT_MIN_COHORT):
                    reason = 'catalyst'
            if reason:
                veto_counts[reason] += 1
                continue
            d = r.to_dict()
            d['_window'] = win.w
            taken_rows.append(d)
    out = pd.DataFrame(taken_rows)
    print(f"overlap skips (symbol already ordered by an earlier window): "
          f"{n_overlap_skips}")
    print(f"cross-window family collisions kept (dedup is within-window "
          f"by the pre-registered rule): {n_fam_cross}")
    print(f"vetoed (slot spent, no refill): {dict(veto_counts)}")
    return out


PDR_MIN = 11.0
G1_RV_MIN = 7.106
G1_PDR_MIN = 9.226
G1_SHORT_HIST = False


def main() -> int:
    global PDR_MIN, G1_RV_MIN, G1_PDR_MIN, G1_SHORT_HIST
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', action='append', required=True,
                    metavar='W:RANKED.csv:FEATURES.csv:RS_MIN',
                    help='one per opening-range width, ascending W')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--parity', default=None,
                    help='a pipeline book CSV this run must reproduce')
    ap.add_argument('--pdr-min', type=float, default=PDR_MIN)
    ap.add_argument('--g1-rv-min', type=float, default=G1_RV_MIN)
    ap.add_argument('--g1-pdr-min', type=float, default=G1_PDR_MIN)
    a = ap.parse_args()
    PDR_MIN, G1_RV_MIN, G1_PDR_MIN = a.pdr_min, a.g1_rv_min, a.g1_pdr_min

    wins = []
    for spec in a.window:
        w, ranked, feats, rs = spec.split(':')
        wins.append(Window(int(w), ranked, feats, float(rs)))
    wins.sort(key=lambda x: x.w)
    print(f"windows: {[(w.w, w.rs_min) for w in wins]}  slots={a.slots}")
    news = load_news_map()
    book = build(wins, a.slots, news)
    book.to_csv(a.out, index=False)
    n_nf = int((book['entered'] == 0).sum()) if 'entered' in book else 0
    print(f"BOOK {a.out}: {len(book)} picks = {len(book) - n_nf} filled + "
          f"{n_nf} no-fill | P&L ${book['_sized_pnl'].sum():,.2f}")
    if a.parity:
        ref = read_orb_csv(a.parity)
        ref['date'] = pd.to_datetime(ref['date'])
        kr = set(zip(ref['symbol'], ref['date']))
        kb = set(zip(book['symbol'], book['date']))
        same = kr == kb
        d = abs(ref['_sized_pnl'].sum() - book['_sized_pnl'].sum())
        print(f"PARITY vs {a.parity}: picks {len(ref)} vs {len(book)}, "
              f"keys_equal={same}, |ΔP&L|={d:.6f}")
        if not same or d > 0.005:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
