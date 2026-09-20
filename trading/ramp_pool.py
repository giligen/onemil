"""Pooled ramp statistic — ONE precision gate across ORB, BF and the HOD-break dry stream.

Why this exists (research/mature_method/frames9/REPORT.md F30, built in frames10 F33):

    The ramp is the only place in this system where statistical power converts directly into
    money per unit time.  At ORB's 6.5 trades a week a 0.2 R portfolio question needs 1.66 YEARS
    to resolve at 80 % power; pooling ORB with BF buys almost nothing (1.55 years — BF is 9 % of
    the trades).  Adding HOD-break's ~27 PAPER trades a week takes the same question to
    **10 weeks**, because a dry book earns nothing and therefore risks nothing: it can enter a
    DIAGNOSTIC for free.

So the dry stream contributes to **n and the band ONLY**.  It is excluded from every P&L clause by
construction, and `apply_pooled_gate` can only ever HOLD a book that its own gate would have
advanced — never the reverse.  The above-water rule (`project_orb_ramp_above_water_rule`,
owner 2026-07-23) is inviolable and is asserted in `tests/test_ramp_pool.py`.

The statistic
-------------
For every closed trade *i* of book *b*: ``z_i = R_i / SD_b``, where ``R_i`` is that book's own
realized R and ``SD_b`` is the per-trade SD of the SAME R definition in that book's frozen BT
reference (`trading/ramp_bt_band.py`, one source for both the SD and the band — a band built on one
distribution and a SD on another is exactly the class of defect this house keeps shipping).

    ORB      R = pnl / total_risk        (BT: pnl_pct / range_size_pct)
    BF       R = pnl / risk_per_trade    (BT: pnl / $2,000)
    HOD dry  R = the simulated bracket's own R at risk_usd (BT: the B2 reference book's net R)

The pooled estimate is ``z_bar`` with a **day-clustered** SE — one cluster per SESSION across all
books, because the three books share the session, the account and the market factor, so a naive SE
overstates precision.

Lives in `trading/` for the same reason as `ramp_freeze` and `ramp_bt_band`: ONE spec imported by
both ramp checkers.
"""
from __future__ import annotations

import csv
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trading import ramp_bt_band as band_mod

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TRADES_DB = ROOT / 'data' / 'trades.db'

BOOK_ORB = 'orb'
BOOK_BF = 'bf'
BOOK_HOD_DRY = 'hod_dry'

#: Books that earn nothing. They may enter n and the band; never a P&L clause.
DRY_BOOKS = frozenset({BOOK_HOD_DRY})

#: Below this many pooled trades the reading is NO-DATA and blocks nothing.
MIN_POOLED_N = 10

#: The pooled DEMOTE trigger needs this much n before the portfolio, not the book, is the story.
DEMOTE_MIN_N = 30

#: Where `scripts/hod_break_eod_check.py` appends the dry stream (day,symbol,r per closed sim
#: trade). Absent = the dry stream is simply missing from the pool, logged at WARNING.
DRY_POOL_PATH = ROOT / 'data' / 'hod_dry_pool.csv'
DRY_POOL_HEADER = ('day', 'symbol', 'r')

POOL_RULE = ("advisory: pooled z BELOW-p10 holds a book its own gate would advance; "
             "pooled z below p5 with n >= 30 demotes every live book one stage; "
             "the dry stream counts toward n and the band ONLY")


@dataclass(frozen=True)
class Trade:
    """One closed trade of one book, already expressed in that book's own R."""
    book: str
    day: str
    r: float


@dataclass(frozen=True)
class PooledStat:
    """The pooled standardised effect and its day-clustered SE."""
    z_bar: Optional[float]
    se: Optional[float]
    n: int
    per_book_n: Mapping[str, int]
    dry_n: int
    n_days: int
    excluded: Tuple[str, ...] = ()

    @property
    def live_n(self) -> int:
        """Trades from books that actually risk money."""
        return sum(v for k, v in self.per_book_n.items() if k not in DRY_BOOKS)

    @property
    def t(self) -> Optional[float]:
        if self.z_bar is None or not self.se:
            return None
        return self.z_bar / self.se

    def fmt_books(self) -> str:
        return ', '.join(f"{b} {self.per_book_n[b]}" for b in sorted(self.per_book_n))


# --------------------------------------------------------------------------- the statistic
def pooled_z(trades: Sequence[Trade],
             sds: Mapping[str, float]) -> PooledStat:
    """Pool per-trade R across books, each standardised by its OWN BT SD.

    A book with no usable SD is EXCLUDED and named in `PooledStat.excluded` at WARNING — never
    silently dropped (CLAUDE.md fallback rule). A trade with a non-finite R is excluded at ERROR.
    """
    excluded: List[str] = []
    zs: List[float] = []
    days: List[str] = []
    books: List[str] = []
    bad_r = 0
    for t in trades:
        sd = sds.get(t.book)
        if sd is None or not math.isfinite(sd) or sd <= 0:
            msg = f"{t.book}: no usable BT SD — excluded from the pool"
            if msg not in excluded:
                excluded.append(msg)
                logger.warning(f"pooled_z: {msg}")
            continue
        if t.r is None or not math.isfinite(t.r):
            bad_r += 1
            continue
        zs.append(t.r / sd)
        days.append(t.day)
        books.append(t.book)
    if bad_r:
        logger.error(f"pooled_z: {bad_r} trade(s) with an uncomputable R excluded from the pool")
    per_book: Dict[str, int] = {}
    for b in books:
        per_book[b] = per_book.get(b, 0) + 1
    dry_n = sum(v for k, v in per_book.items() if k in DRY_BOOKS)
    n = len(zs)
    n_days = len(set(days))
    if n == 0:
        return PooledStat(None, None, 0, per_book, dry_n, 0, tuple(excluded))
    z_bar = sum(zs) / n
    se = _clustered_se(zs, days, z_bar)
    return PooledStat(z_bar, se, n, per_book, dry_n, n_days, tuple(excluded))


def _clustered_se(zs: Sequence[float], days: Sequence[str],
                  z_bar: float) -> Optional[float]:
    """Day-clustered SE of the mean: one cluster per session across ALL books."""
    if len(zs) < 3:
        return None
    by_day: Dict[str, float] = {}
    for z, d in zip(zs, days):
        by_day[d] = by_day.get(d, 0.0) + (z - z_bar)
    if len(by_day) < 3:
        return None
    ss = sum(v * v for v in by_day.values())
    se = math.sqrt(ss) / len(zs)
    return se if se > 0 else None


def pooled_band(bt_r_by_book: Mapping[str, Sequence[float]],
                sds: Mapping[str, float],
                per_book_n: Mapping[str, int],
                draws: int = band_mod.DRAWS,
                seed: int = band_mod.SEED) -> Optional[band_mod.Band]:
    """Bootstrap the SAME pooled statistic from the reference books.

    Draws `per_book_n[b]` trades with replacement from book b's BT R distribution, standardises by
    `sds[b]`, pools them and takes the mean — `draws` times. p5/p10/p90 of those means is the band
    the live pooled z has to sit in, exactly as the per-book band works today.
    """
    import numpy as np
    parts = []
    total = 0
    for book, n in per_book_n.items():
        if n <= 0:
            continue
        vals = bt_r_by_book.get(book) or []
        sd = sds.get(book)
        if not vals or sd is None or not math.isfinite(sd) or sd <= 0:
            logger.warning(f"pooled_band: {book} has no usable BT reference — excluded from the "
                           f"pooled band (its {n} live trade(s) still count toward n)")
            continue
        parts.append((np.asarray(vals, dtype=float) / sd, n))
        total += n
    if not parts or total <= 0:
        logger.error("pooled_band: no book contributed a reference distribution — NO-DATA")
        return None
    rng = np.random.default_rng(seed)
    acc = np.zeros(draws, dtype=float)
    n_ref = 0
    for arr, n in parts:
        acc += rng.choice(arr, size=(draws, n), replace=True).sum(axis=1)
        n_ref += len(arr)
    means = acc / total
    p5, p10, p90 = (float(x) for x in np.percentile(means, [5, 10, 90]))
    return band_mod.Band(p5=p5, p10=p10, p90=p90, n=total, draws=draws, n_ref=n_ref)


def classify_pooled(stat: PooledStat, band: Optional[band_mod.Band]) -> str:
    """BELOW-p5 / BELOW-p10 / IN-BAND / ABOVE-p90 / NO-DATA (under MIN_POOLED_N -> NO-DATA)."""
    if stat.n < MIN_POOLED_N:
        return band_mod.NO_DATA
    return band_mod.classify(stat.z_bar, band)


# --------------------------------------------------------------------------- the gate
def apply_pooled_gate(book_verdict: str, pooled_status: str) -> Tuple[str, Optional[str]]:
    """The ADVISORY pooled gate. Returns (verdict, why-it-changed or None).

    THE INVARIANT, asserted in tests: this function can only ever DOWNGRADE. A book whose own gate
    did not say ADVANCE can never leave here saying ADVANCE — the pool never lifts a losing book on
    a winning sibling's evidence, and the above-water rule is untouched by anything here.
    """
    if book_verdict != 'ADVANCE':
        return book_verdict, None
    if pooled_status in (band_mod.BELOW_P5, band_mod.BELOW_P10):
        return 'HOLD', (f"pooled z {pooled_status} — the portfolio is running below its backtest; "
                        f"this stage would have passed on noise")
    return book_verdict, None


def pooled_demote(stat: PooledStat, pooled_status: str) -> bool:
    """The pooled DEMOTE trigger: below p5 with n >= DEMOTE_MIN_N (every live book, one stage)."""
    return pooled_status == band_mod.BELOW_P5 and stat.n >= DEMOTE_MIN_N


def pooled_line(stat: PooledStat, band: Optional[band_mod.Band], status: str) -> str:
    """The single ADVISORY line both ramp checkers print beside their own BT-band line."""
    if stat.z_bar is None or band is None or status == band_mod.NO_DATA:
        return (f"  POOLED z: NO-DATA (n={stat.n}, need {MIN_POOLED_N}; "
                f"{stat.fmt_books() or 'no books'}) — blocks nothing | {POOL_RULE}")
    se = f"{stat.se:.3f}" if stat.se else 'n/a'
    return (f"  POOLED z = {stat.z_bar:+.3f} +/- {se} on n={stat.n} "
            f"({stat.fmt_books()}; dry {stat.dry_n} of {stat.n}, {stat.n_days} sessions) vs "
            f"pooled BT band {band.fmt()} [{band.n_ref} BT trades, {band.draws} draws] "
            f"-> {status} (ADVISORY, the per-book verdict decides) | {POOL_RULE}")


# --------------------------------------------------------------------------- the loaders
def load_live_trades(book: str, since: str, db_path: Path = TRADES_DB,
                     risk_base: Optional[float] = None) -> List[Trade]:
    """Closed live trades of `book` since `since`, as per-trade R — READ-ONLY.

    ORB divides by the fill's own `total_risk` (the live 1R); BF divides by `risk_base` (the stage
    base), the same normalization its BT reference uses. Rows whose R cannot be computed are
    dropped with an ERROR, never counted as zero.
    """
    strategy = {BOOK_ORB: 'orb', BOOK_BF: 'bull_flag'}.get(book)
    if strategy is None:
        logger.error(f"load_live_trades: unknown book {book!r} — no trades loaded")
        return []
    if not Path(db_path).exists():
        logger.error(f"load_live_trades: {db_path} missing — {book} contributes nothing to the pool")
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT trade_date, pnl, total_risk FROM trades "
            "WHERE strategy=? AND trade_date>=? AND pnl IS NOT NULL "
            "ORDER BY trade_date", (strategy, since)).fetchall()
        conn.close()
    except sqlite3.Error as e:  # noqa: BLE001 - a decision aid must not crash
        logger.error(f"load_live_trades: {db_path} unreadable ({e}) — {book} excluded from the pool")
        return []
    out: List[Trade] = []
    bad = 0
    for r in rows:
        d = dict(r)
        denom = risk_base if book == BOOK_BF else d.get('total_risk')
        try:
            denom = float(denom or 0)
            pnl = float(d['pnl'])
        except (TypeError, ValueError):
            bad += 1
            continue
        if denom <= 0:
            bad += 1
            continue
        out.append(Trade(book, str(d['trade_date'])[:10], pnl / denom))
    if bad:
        logger.error(f"load_live_trades: {bad} {book} trade(s) without a usable 1R — "
                     f"excluded from the pool (never counted as zero)")
    return out


def load_dry_trades(path: Path = DRY_POOL_PATH, since: Optional[str] = None) -> List[Trade]:
    """The HOD-break dry stream: (day, symbol, r) rows appended by the EOD check.

    A missing file is not an error — the dry run may not have written yet — but it IS logged, so a
    pooled line that is thinner than expected always says why.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"load_dry_trades: {p} missing — the HOD dry stream contributes nothing "
                       f"to the pooled band (the pooled reading is live-books only)")
        return []
    out: List[Trade] = []
    bad = 0
    try:
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                day = str(row.get('day') or '')[:10]
                try:
                    r = float(row.get('r'))
                except (TypeError, ValueError):
                    bad += 1
                    continue
                if not day or not math.isfinite(r):
                    bad += 1
                    continue
                if since and day < since:
                    continue
                out.append(Trade(BOOK_HOD_DRY, day, r))
    except OSError as e:  # noqa: BLE001
        logger.error(f"load_dry_trades: {p} unreadable ({e}) — dry stream excluded")
        return []
    if bad:
        logger.error(f"load_dry_trades: {bad} dry row(s) without a usable R — excluded")
    return out


def append_dry_trades(day: str, trades: Iterable[Tuple[str, float]],
                      path: Path = DRY_POOL_PATH) -> int:
    """Append one session's simulated dry trades to the pool file. Returns rows written.

    The producer is `scripts/hod_break_eod_check.py` (its EXECUTABLE would-be book). Writing the
    pool file is the ONLY write anything in this module performs, and it never touches trades.db.
    """
    p = Path(path)
    rows = [(day, str(s), float(r)) for s, r in trades if math.isfinite(float(r))]
    if not rows:
        return 0
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(DRY_POOL_HEADER)
        w.writerows(rows)
    return len(rows)


def reference_sds_and_r(books: Sequence[str],
                        orb_catalyst_veto: bool = False,
                        bf_min_daily_volume: int = 200_000
                        ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """The frozen BT reference R distribution and its SD, per book — ONE source for both."""
    sds: Dict[str, float] = {}
    rr: Dict[str, List[float]] = {}
    for b in books:
        vals = band_mod.pool_reference_r(b, orb_catalyst_veto=orb_catalyst_veto,
                                         bf_min_daily_volume=bf_min_daily_volume)
        if not vals:
            continue
        sd = band_mod.sd_of(vals)
        if sd is None:
            logger.warning(f"reference_sds_and_r: {b} reference has no usable SD — excluded")
            continue
        rr[b] = vals
        sds[b] = sd
    return sds, rr


def reading(since_by_book: Mapping[str, str],
            bf_risk_base: Optional[float] = None,
            orb_catalyst_veto: bool = False,
            bf_min_daily_volume: int = 200_000,
            dry_path: Path = DRY_POOL_PATH,
            db_path: Path = TRADES_DB) -> Tuple[PooledStat, Optional[band_mod.Band], str, str]:
    """Everything a ramp checker needs for its one advisory line.

    `since_by_book`: {'orb': stage_start, 'bf': stage_start, 'hod_dry': first dry session}.
    Returns (stat, band, status, line).
    """
    trades: List[Trade] = []
    if BOOK_ORB in since_by_book:
        trades += load_live_trades(BOOK_ORB, since_by_book[BOOK_ORB], db_path)
    if BOOK_BF in since_by_book:
        trades += load_live_trades(BOOK_BF, since_by_book[BOOK_BF], db_path,
                                   risk_base=bf_risk_base)
    trades += load_dry_trades(dry_path, since_by_book.get(BOOK_HOD_DRY))
    sds, rr = reference_sds_and_r([BOOK_ORB, BOOK_BF, BOOK_HOD_DRY],
                                  orb_catalyst_veto=orb_catalyst_veto,
                                  bf_min_daily_volume=bf_min_daily_volume)
    stat = pooled_z(trades, sds)
    band = pooled_band(rr, sds, stat.per_book_n) if stat.n else None
    status = classify_pooled(stat, band)
    return stat, band, status, pooled_line(stat, band, status)


#: The HOD-break dry run's first session (docs/scaling_plan_2026.md, CLAUDE.md Strategy 4).
HOD_DRY_SINCE = '2026-09-14'


def advisory_line(orb_since: Optional[str] = None, bf_since: Optional[str] = None,
                  dry_since: str = HOD_DRY_SINCE,
                  dry_path: Path = DRY_POOL_PATH,
                  db_path: Path = TRADES_DB) -> str:
    """The ONE line a ramp checker prints. Resolves the other book's stage itself; never raises.

    This is deliberately self-contained so each checker adds exactly one line and nothing else:
    the pooled reading is ADVISORY in this pass and must not be able to break a checker that the
    owner runs daily.
    """
    try:
        from trading import ramp_stage
        import yaml
        if orb_since is None:
            orb_since = ramp_stage.resolve('orb')[0]
        if bf_since is None:
            bf_since = ramp_stage.resolve('bf')[0]
        cfg = yaml.safe_load(open(ROOT / 'config.yaml'))
        bf_base = float(cfg['trading']['risk_per_trade'])
        adv = int(cfg.get('scanner', {}).get('min_daily_volume', 200_000))
        veto = False
        orb_yaml = ROOT / 'orb.yaml'
        if orb_yaml.exists():
            oc = yaml.safe_load(open(orb_yaml)) or {}
            veto = bool(oc.get('filter', {}).get('catalyst_veto', {}).get('enabled'))
        _, _, _, line = reading({BOOK_ORB: orb_since, BOOK_BF: bf_since,
                                 BOOK_HOD_DRY: dry_since},
                                bf_risk_base=bf_base, orb_catalyst_veto=veto,
                                bf_min_daily_volume=adv, dry_path=dry_path, db_path=db_path)
        return line
    except Exception as e:  # noqa: BLE001 - an advisory line must never break a daily checker
        logger.error(f"advisory_line: pooled reading unavailable ({e}) — printing NO-DATA")
        return f"  POOLED z: NO-DATA (pooled reading unavailable: {e}) — blocks nothing"
