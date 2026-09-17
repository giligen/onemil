#!/usr/bin/env python3
"""Book selection for the HOD-break reporting scripts (2026-09-17).

`HodBreakEngine` runs TWO books through one code path (`book: hod_break | red_to_green`): same fill/bracket/flat
machinery, different signal function, universe screen, DB strategy tag, client-order prefix and log tag. The EOD
check, the miss audit and the dead-man flat therefore need exactly one book-shaped seam each, and this module is it:

    book = load_book('red_to_green')          # or 'hod_break' (the default everywhere)
    book.simulate(sym, o, h, l, c, v, m, adv) # the spec's whole trade on one symbol-day
    book.detect(sym, o, h, l, v, m, adv)      # the spec's first signal

Nothing here reads the journal or the DB; the scripts own that. Pure selection + the two spec calls, so the default
(`hod_break`) path is the same functions the scripts always called and its output is byte-identical.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trading.hod_break import (HodBreakParams, HodBreakTrade, OPEN_MINUTE, STOP_FILL_SLIP,  # noqa: E402,F401
                               detect as hod_detect, entry_fill, run_book, simulate as hod_simulate, walk_exit)
from trading.red_to_green import (RedToGreenParams, detect as r2g_detect, level_for,  # noqa: E402
                                  prior_day_range_pct, r_ok)

BOOKS = ('hod_break', 'red_to_green')
DEFAULT_BOOK = 'hod_break'


@dataclass
class BookSpec:
    """One book's identity (log tag, DB strategy, config) plus its two spec entry points."""
    name: str                                   # 'hod_break' | 'red_to_green'
    tag: str                                    # 'HOD' | 'R2G' — the journal/log tag without brackets
    strategy: str                               # the DB `strategy` column value
    cfg: dict                                   # Config().hod_break_cfg / Config().red_to_green_cfg
    params: object                              # HodBreakParams | RedToGreenParams
    prev_day: Dict[str, tuple] = field(default_factory=dict)   # red-to-green: symbol -> (prior close, high, low)

    # ---------------------------------------------------------------- identity
    @property
    def is_r2g(self) -> bool:
        return self.name == 'red_to_green'

    @property
    def dry_tag(self) -> str:
        return f'[{self.tag} DRY]'

    @property
    def live_tag(self) -> str:
        return f'[{self.tag}]'

    @property
    def coid_prefix(self) -> str:
        return 'r2g' if self.is_r2g else 'hod'

    def journal_line(self, line: str) -> bool:
        """True for a journal line that belongs to this book (its tag, or its module's errors)."""
        return f'[{self.tag}' in line or (self.strategy in line and ('ERROR' in line or 'Traceback' in line))

    # ---------------------------------------------------------------- the book's prior-day inputs
    def prior(self, symbol: str) -> Tuple[float, Optional[float]]:
        """(prior close, prior-day range %) for the red-to-green book; (0.0, None) when unknown or not this book."""
        pc, ph, pl = self.prev_day.get(symbol, (0.0, None, None))
        return float(pc or 0.0), prior_day_range_pct(ph, pl)

    def eligible_prior_day(self, symbol: str) -> bool:
        """The 09:30-known half of the red-to-green precondition: a prior close and a prior-day range >= pdr_min_pct."""
        if not self.is_r2g:
            return True
        pc, pdr = self.prior(symbol)
        return pc > 0 and pdr is not None and pdr >= self.params.pdr_min_pct

    def level(self, symbol: str) -> Optional[float]:
        """The red-to-green break level (prior close x (1 + buffer)) — None for the HOD book (its level is the HOD)."""
        if not self.is_r2g:
            return None
        pc, _ = self.prior(symbol)
        return level_for(pc, self.params) if pc > 0 else None

    def screen_universe(self, adv: Dict[str, float], last_close: Dict[str, float]) -> set:
        """The symbols the ENGINE subscribes at session start — `HodBreakEngine._stream_the_universe`, same screen."""
        syms = {s for s, a in adv.items() if a >= float(self.cfg['min_adv20'])
                and last_close.get(s, 0.0) >= float(self.cfg.get('universe_min_prev_close', 0.0))}
        if self.is_r2g:
            syms = {s for s in syms if self.eligible_prior_day(s)}
        return syms

    # ---------------------------------------------------------------- the spec
    def detect(self, symbol: str, o: Sequence[float], h: Sequence[float], l: Sequence[float],
               v: Sequence[float], m: Sequence[int], adv20: float, start_idx: int = 0):
        """The book's first signal on one symbol-day of CLOSED 1-min bars, or None."""
        if not self.is_r2g:
            return hod_detect(o, h, l, v, m, adv20, self.params, start_idx=start_idx)
        pc, pdr = self.prior(symbol)
        return r2g_detect(o, h, l, v, m, adv20, pc, pdr, self.params, start_idx=start_idx)

    def simulate(self, symbol: str, o, h, l, c, v, m, adv20: float) -> Optional[HodBreakTrade]:
        """detect -> next-open capped fill -> the configured exit walk. Identical shape for both books; the HOD book
        delegates to `trading.hod_break.simulate` so its numbers are byte-identical to the pre-book-option scripts."""
        if not self.is_r2g:
            return hod_simulate(o, h, l, c, v, m, adv20, self.params)
        sig = self.detect(symbol, o, h, l, v, m, adv20)
        if sig is None or sig.bar_idx + 1 >= len(o):
            return None
        entry = entry_fill(float(o[sig.bar_idx + 1]), sig.level, self.params)
        if entry is None or sig.stop >= entry or not r_ok(entry, sig.stop, self.params):
            return None
        r = entry - sig.stop
        target = entry + self.params.target_r * r
        k, px, why = walk_exit(o, h, l, c, m, sig.bar_idx + 1, entry, sig.stop, target, self.params)
        return HodBreakTrade(entry_idx=sig.bar_idx + 1, entry=entry, stop=sig.stop, target=target, r_per_share=r,
                             exit_idx=k, exit_price=px, reason=why, rr=(px - entry) / r)


def load_prev_day(cache_path) -> Dict[str, tuple]:
    """The engine's OWN prior-session loader (daily_bars, read-only) — one definition, never a second query."""
    from trading.hod_break_engine import load_prev_day_from_daily_bars
    return load_prev_day_from_daily_bars(cache_path)


def load_book(name: str = DEFAULT_BOOK, cfg: Optional[dict] = None, prev_day: Optional[Dict[str, tuple]] = None,
              cache_path=None) -> BookSpec:
    """Build the BookSpec for `name`. `cfg`/`prev_day` are injectable (tests); otherwise config.yaml and daily_bars."""
    if name not in BOOKS:
        raise ValueError(f"unknown book {name!r} — expected one of {BOOKS}")
    if cfg is None:
        from config import Config
        c = Config()
        cfg = c.red_to_green_cfg if name == 'red_to_green' else c.hod_break_cfg
    if name == 'red_to_green':
        pd_ = prev_day if prev_day is not None else (load_prev_day(cache_path) if cache_path else {})
        return BookSpec('red_to_green', 'R2G', 'red_to_green', cfg, RedToGreenParams(**(cfg.get('params') or {})), pd_)
    return BookSpec('hod_break', 'HOD', 'hod_break', cfg, HodBreakParams(**(cfg.get('params') or {})), {})


def book_from_argv(argv) -> str:
    """Pop `--book X` out of `argv` (in place) and return the book name; default `hod_break`.
    Keeps the scripts' positional arguments (the day, --floor) exactly where they were."""
    if '--book' not in argv:
        return DEFAULT_BOOK
    i = argv.index('--book')
    if i + 1 >= len(argv):
        raise SystemExit(f"--book needs a value, one of {BOOKS}")
    name = argv[i + 1]
    if name not in BOOKS:
        raise SystemExit(f"unknown --book {name!r} — expected one of {BOOKS}")
    del argv[i:i + 2]
    return name
