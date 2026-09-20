"""BT-band gate — is the live book's R/trade consistent with its backtest?

Gate-2 item 4 of docs/scaling_plan_2026.md:

> Realized stage R/trade inside the BT's [p10, p90] band for that n. Below p10
> -> not advancing (and see demote). ABOVE p90 -> also not advancing until n
> grows — a live book beating its own backtest is a leak or a bug before it is
> luck. Demote on R/trade below the BT's p5 band for that n after >= 8 trades.

The band is a BOOTSTRAP of the BT's own per-trade R distribution: draw n trades
with replacement 2,000 times, take p5/p10/p90 of the mean. It answers "how
lucky/unlucky can n draws of the BACKTEST look?", which is the only fair bar for
a live sample of n.

References (the honest books — one per shipping config):
  ORB catalyst-veto ON : research/fuckup_audit/Q_fill/book_measured_n8.csv
  ORB catalyst-veto OFF: research/orb_gates2/book_G3_meas.csv   <- boots Monday
  BF ADV gate 200K     : research/bf_frequency/runs/P1.csv      <- shipped (reverted 9/19)
  BF ADV gate 0        : research/bf_frequency/runs/VOL_OFF.csv

R definitions (chosen so the live side is computable from trades.db):
  ORB  R = pnl_pct / range_size_pct  — entry range_high, stop range_low, so the
       5-min range IS 1R; size-independent, matching live pnl / total_risk.
  BF   R = pnl / 2000 — the BT's canonical $2K risk normalization; live is
       pnl / trading.risk_per_trade (the stage base).

Lives in `trading/` for the same reason as ramp_freeze: ONE spec shared by both
ramp checkers, which already import from the repo root.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

DRAWS = 2000
SEED = 20260919          # fixed: the printed band must be reproducible

BAND_RULE = ("below p10 → no advance; below p5 after ≥ 8 trades → demote; "
             "above p90 → no advance (leak or bug before luck)")

BELOW_P5, BELOW_P10, IN_BAND, ABOVE_P90 = (
    'BELOW-p5', 'BELOW-p10', 'IN-BAND', 'ABOVE-p90')
NO_DATA = 'NO-DATA'      # no live trades yet, or no usable BT reference

ORB_REF_VETO_ON = ROOT / 'research' / 'fuckup_audit' / 'Q_fill' / 'book_measured_n8.csv'
ORB_REF_VETO_OFF = ROOT / 'research' / 'orb_gates2' / 'book_G3_meas.csv'
BF_REF_P1 = ROOT / 'research' / 'bf_frequency' / 'runs' / 'P1.csv'
BF_REF_ADV_OFF = ROOT / 'research' / 'bf_frequency' / 'runs' / 'VOL_OFF.csv'
BF_REF = BF_REF_P1   # legacy name; the shipped config is P1 (ADV gate 200K)
BF_BT_RISK_USD = 2000.0

# The HOD-break dry stream's reference: the B2 book six research passes are measured against
# (research/mature_method/hod_frames6/REPORT.md). R = the book's own net R per trade. The dry
# stream earns nothing, so this reference only ever feeds the POOLED band's n and width
# (trading/ramp_pool.py) — never a P&L clause.
HOD_REF_B2 = (ROOT / 'research' / 'mature_method' / 'hod_frames6' / 'book6.csv')

#: Last-resort per-trade SDs, used ONLY when a reference file is unreadable (logged at WARNING).
#: The live numbers come from `sd_of(load_reference_r(...))` so that the SD and the band are always
#: computed from the SAME distribution — a band built on one distribution and an SD on another is
#: the defect class this house keeps shipping.
BOOK_SD_FALLBACK = {'orb': 1.431, 'bf': 3.151, 'hod_dry': 1.260}


@dataclass(frozen=True)
class Reference:
    """Which BT book the band came from — printed so the bar is auditable."""
    path: Path
    label: str

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass(frozen=True)
class Band:
    """Bootstrap band of the BT mean-R for a sample of n trades."""
    p5: float
    p10: float
    p90: float
    n: int
    draws: int
    n_ref: int

    def fmt(self) -> str:
        return (f"[p5 {self.p5:+.2f}, p10 {self.p10:+.2f}, "
                f"p90 {self.p90:+.2f}]")


def orb_reference(catalyst_veto_enabled: bool) -> Reference:
    """The ORB book matching the config that is actually running."""
    if catalyst_veto_enabled:
        return Reference(ORB_REF_VETO_ON, 'Q_fill/book_measured_n8.csv '
                                          '(catalyst veto ON)')
    return Reference(ORB_REF_VETO_OFF, 'orb_gates2/book_G3_meas.csv '
                                       '(catalyst veto OFF — the variant '
                                       'that boots Monday)')


def bf_reference(min_daily_volume: int = 200_000) -> Reference:
    """The BF book matching the config that is actually running.

    2026-09-19: the ADV gate was dropped and reverted the same day
    (research/mature_method/entry_cost_audit/REPORT.md — the gate's
    "wrong-side" separation was a spread-blind artefact; the names it
    removes carry a measured 250 bps spread). Follow the live knob like
    orb_reference() follows the catalyst veto, so the band can never be
    built on a book the engine is not running.
    """
    if min_daily_volume <= 0:
        return Reference(BF_REF_ADV_OFF, 'bf_frequency/runs/VOL_OFF.csv '
                                         '(min_daily_volume 0, R = pnl/$2,000)')
    return Reference(BF_REF_P1, 'bf_frequency/runs/P1.csv '
                                '(min_daily_volume 200000 — shipped P1, '
                                'R = pnl/$2,000)')


def _f(v) -> Optional[float]:
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_orb_bt_r(path: Path) -> List[float]:
    """Per-trade R of an ORB book CSV: pnl_pct / range_size_pct, fills only.

    No-fill rows (entered=0) carry no R — they are slot cost, not a trade, and
    the live side counts fills too.
    """
    out: List[float] = []
    skipped = 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            entered = (row.get('entered') or '1').strip()
            if entered in ('0', 'False', 'false'):
                continue
            pnl_pct, rng = _f(row.get('pnl_pct')), _f(row.get('range_size_pct'))
            if pnl_pct is None or not rng:
                skipped += 1
                continue
            out.append(pnl_pct / rng)
    if skipped:
        logger.warning(f"{path.name}: {skipped} filled row(s) without "
                       f"pnl_pct/range_size_pct — excluded from the BT band")
    return out


def load_bf_bt_r(path: Path, risk_usd: float = BF_BT_RISK_USD) -> List[float]:
    """Per-trade R of a BF book CSV: pnl / the BT's $2K risk normalization."""
    out: List[float] = []
    skipped = 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            pnl = _f(row.get('pnl'))
            if pnl is None:
                skipped += 1
                continue
            out.append(pnl / risk_usd)
    if skipped:
        logger.warning(f"{path.name}: {skipped} row(s) without pnl — "
                       f"excluded from the BT band")
    return out


def load_reference_r(ref: Reference, book: str) -> List[float]:
    """R distribution of a reference book; [] (loudly) when unreadable."""
    if not ref.exists:
        logger.error(f"BT reference missing: {ref.path} — the BT-band gate "
                     f"cannot be scored (treated as NO-DATA, blocks ADVANCE)")
        return []
    try:
        return (load_orb_bt_r(ref.path) if book == 'orb'
                else load_bf_bt_r(ref.path))
    except Exception as e:  # noqa: BLE001 - a decision aid must not crash
        logger.error(f"BT reference {ref.path} unreadable ({e}) — BT-band "
                     f"gate NO-DATA (blocks ADVANCE)")
        return []


def load_hod_bt_r(path: Path = HOD_REF_B2,
                  splits: Sequence[str] = ('TRAIN', 'VAL')) -> List[float]:
    """Per-trade R of the HOD-break B2 reference book: the book's own net R, TRAIN+VAL only.

    TEST is excluded because it is sealed for the research programme
    (research/mature_method/frames10/FREEZE.md); the band only needs the shape.
    """
    out: List[float] = []
    skipped = 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if splits and (row.get('split') or '') not in splits:
                continue
            v = _f(row.get('net'))
            if v is None:
                skipped += 1
                continue
            out.append(v)
    if skipped:
        logger.warning(f"{path.name}: {skipped} row(s) without net R — excluded from the BT band")
    return out


def sd_of(r_values: Sequence[float]) -> Optional[float]:
    """Per-trade SD of a reference R distribution (>= 2 values), else None."""
    if not r_values or len(r_values) < 2:
        return None
    import statistics
    sd = statistics.stdev(r_values)
    return sd if sd > 0 else None


def pool_reference_r(book: str, orb_catalyst_veto: bool = False,
                     bf_min_daily_volume: int = 200_000) -> List[float]:
    """The reference R distribution for a POOL member ('orb' | 'bf' | 'hod_dry')."""
    if book == 'orb':
        return load_reference_r(orb_reference(orb_catalyst_veto), 'orb')
    if book == 'bf':
        return load_reference_r(bf_reference(bf_min_daily_volume), 'bf')
    if book == 'hod_dry':
        ref = Reference(HOD_REF_B2, 'hod_frames6/book6.csv (B2 dry reference, net R)')
        if not ref.exists:
            logger.error(f"BT reference missing: {ref.path} — the HOD dry stream has no pooled "
                         f"band contribution")
            return []
        try:
            return load_hod_bt_r(ref.path)
        except Exception as e:  # noqa: BLE001 - a decision aid must not crash
            logger.error(f"BT reference {ref.path} unreadable ({e}) — HOD dry excluded")
            return []
    logger.error(f"pool_reference_r: unknown book {book!r} — no reference")
    return []


def bootstrap_band(r_values: Sequence[float], n: int, draws: int = DRAWS,
                   seed: int = SEED) -> Optional[Band]:
    """p5/p10/p90 of the mean of n BT trades drawn with replacement."""
    if not r_values or n <= 0:
        return None
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(r_values, dtype=float)
    means = rng.choice(arr, size=(draws, n), replace=True).mean(axis=1)
    p5, p10, p90 = (float(x) for x in np.percentile(means, [5, 10, 90]))
    return Band(p5=p5, p10=p10, p90=p90, n=n, draws=draws, n_ref=len(arr))


def classify(mean_r: Optional[float], band: Optional[Band]) -> str:
    """BELOW-p5 / BELOW-p10 / IN-BAND / ABOVE-p90 / NO-DATA."""
    if mean_r is None or band is None:
        return NO_DATA
    if mean_r < band.p5:
        return BELOW_P5
    if mean_r < band.p10:
        return BELOW_P10
    if mean_r > band.p90:
        return ABOVE_P90
    return IN_BAND


def band_line(status: str, mean_r: Optional[float], band: Optional[Band],
              ref: Reference) -> str:
    """The single line both ramp checkers print for Gate-2 item 4."""
    if band is None or mean_r is None:
        return (f"  BT band: NO-DATA (ref {ref.label}) — no live R sample yet; "
                f"rule: {BAND_RULE}")
    return (f"  BT band: live mean R {mean_r:+.2f} on n={band.n} vs "
            f"{band.fmt()} from {ref.label} [{band.n_ref} BT trades, "
            f"{band.draws} draws] → {status} | rule: {BAND_RULE}")
