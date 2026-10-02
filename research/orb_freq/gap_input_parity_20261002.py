#!/usr/bin/env python3
"""Step 1 measurement: which gap input matches the backtest's admissions?

For 2026-09-30 / 10-01 / 10-02 compares, per symbol of the liquid universe,
  (a) the gap the LIVE engine used (journal `[ORB] GAP_GATE` lines; only archived for 10/1,
      9/30 = the two live picks ASTX/AEHG; 10/2 had no decision),
  (b) the BT gap: daily_bars open (cache.db, read-only; 10/2 from Alpaca daily bars) vs prior close,
  (c) the gap from the 09:30 ET one-minute bar open (Alpaca SIP historical) vs prior daily close.
Admission classes use orb.yaml thresholds (read only): production gap>=min_gap, add-on P1 band.
Output: research/orb_freq/gap_input_parity_20261002.md.   READ-ONLY (cache.db mode=ro).
"""
import logging, os, re, sqlite3, sys
from datetime import date, datetime, timezone
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_sources.alpaca_client import AlpacaClient  # noqa: E402

log = logging.getLogger("gap_parity")
DAYS = [("2026-09-30", "2026-09-29"), ("2026-10-01", "2026-09-30"), ("2026-10-02", "2026-10-01")]
LIVE_PICKS = {"2026-09-30": {"ASTX", "AEHG"}}
ARCHIVE = {"2026-10-01": ROOT / "logs/session_archive/2026-10-01.log"}
RE_GG = re.compile(r"GAP_GATE (\S+) gap_input_open=([\d.]+) gap_input_prev_close=([\d.]+) gap_pct=(-?[\d.]+) "
                   r"source=(\S+) timestamp=(\S+)")


def classes(cfg):
    u = cfg["universe"]
    p1 = [p for p in u["addon_pools"]["pools"]][0]
    return u, p1


def admit(gap, price, u, p1):
    """Admission class of (gap, open price) by orb.yaml thresholds ('' when rejected)."""
    if gap is None or price is None:
        return ""
    if u["min_price"] <= price <= u["max_price"] and gap >= u["min_gap_pct"]:
        return "prod"
    if p1["min_price"] <= price <= p1["max_price"] and p1["min_gap_pct"] <= gap <= p1["max_gap_pct"]:
        return "P1"
    return ""


def live_gaps(day):
    """symbol -> (open, prev_close, gap, source) of the last GAP_GATE line at or before 13:36 UTC."""
    out = {}
    path = ARCHIVE.get(day)
    if not path or not path.exists():
        return out
    for line in path.open(errors="replace"):
        m = RE_GG.search(line)
        if m and m.group(6)[11:16] <= "13:36":
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5))
    return out


def main():
    load_dotenv()
    cfg = yaml.safe_load((ROOT / "orb.yaml").read_text())
    u, p1 = classes(cfg)
    cli = AlpacaClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))
    conn = sqlite3.connect(f"file:{ROOT/'data/cache.db'}?mode=ro", uri=True)
    lines = ["# Gap-input parity 2026-09-30 / 10-01 / 10-02 (Step 1)", "",
             f"Thresholds (orb.yaml, read only): prod gap>={u['min_gap_pct']} price {u['min_price']}-{u['max_price']} "
             f"prev_vol>={u['min_prev_volume']}; P1 gap {p1['min_gap_pct']}-{p1['max_gap_pct']}.", ""]
    for day, prev in DAYS:
        prevrows = conn.execute("SELECT symbol,open,close,volume FROM daily_bars WHERE bar_date=? AND volume>=? "
                                "AND close BETWEEN 1.5 AND 45", (prev, u["min_prev_volume"])).fetchall()
        syms = [r[0] for r in prevrows]
        pc = {r[0]: r[2] for r in prevrows}
        pday_open = {r[0]: r[1] for r in prevrows}
        d_b = {r[0]: r[1] for r in conn.execute("SELECT symbol,open FROM daily_bars WHERE bar_date=?", (day,))}
        if day == "2026-10-02" or not d_b:
            d_b = {}
            d = date.fromisoformat(day)
            for i in range(0, len(syms), 200):
                res = cli.get_daily_bars_range(syms[i:i + 200], d, d)
                for s, bars in res.items():
                    if bars:
                        d_b[s] = float(bars[0]["open"])
        c_open = {}
        s0 = datetime.fromisoformat(f"{day}T13:30:00+00:00")
        s1 = datetime.fromisoformat(f"{day}T13:31:00+00:00")
        for i in range(0, len(syms), 200):
            res = cli.get_1min_bars_range_multi(syms[i:i + 200], s0, s1)
            for s, df in res.items():
                if len(df):
                    c_open[s] = float(df.iloc[0]["open"])
        a = live_gaps(day)
        picks = LIVE_PICKS.get(day, set())
        gap = lambda o, s: (o - pc[s]) / pc[s] * 100 if (o and s in pc) else None
        rows, dis_ab, dis_cb, n_b, n_c = [], [], [], 0, 0
        cand = [s for s in syms if (gap(d_b.get(s), s) or -9) >= 2.5 or (gap(c_open.get(s), s) or -9) >= 2.5
                or (s in a and a[s][2] >= 2.5) or s in picks]
        for s in cand:
            gb, gc = gap(d_b.get(s), s), gap(c_open.get(s), s)
            ga = a[s][2] if s in a else None
            cb = admit(gb, d_b.get(s), u, p1)
            cc = admit(gc, c_open.get(s), u, p1)
            ca = admit(ga, a[s][0], u, p1) if s in a else ("prod/P1 (live pick)" if s in picks else None)
            n_b += bool(cb); n_c += bool(cc)
            if ca is not None and bool(ca) != bool(cb) or (ca and ca in ("prod", "P1") and ca != cb):
                dis_ab.append(s)
            if cc != cb:
                dis_cb.append(s)
            rows.append((s, ga, gb, gc, ca, cb, cc, pday_open.get(s), d_b.get(s), c_open.get(s), pc.get(s)))
        lines += [f"## {day} (prior session {prev}); universe {len(syms)} liquid symbols; "
                  f"daily_bars opens {len(d_b)}, 09:30 bars {len(c_open)}",
                  f"BT admits {n_b}; 09:30-bar admits {n_c}; (a) live gaps archived for {len(a)} symbols.",
                  f"Disagreements (c) vs (b): {len(dis_cb)} {dis_cb}",
                  f"Disagreements (a) vs (b): {len(dis_ab)} {dis_ab}" +
                  ("" if a or picks else "  [no (a) archive for this day]"), "",
                  "| sym | a gap | b gap | c gap | a cls | b cls | c cls | prevday open | b open | c open | prev close |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda x: "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
        for r in rows:
            if r[0] in dis_ab or r[0] in dis_cb or r[0] in picks:
                lines.append("| " + " | ".join(f(x) for x in r) + " |")
        lines.append("")
        log.info("%s: b admits %d c admits %d dis_cb %d dis_ab %d", day, n_b, n_c, len(dis_cb), len(dis_ab))
    (ROOT / "research/orb_freq/gap_input_parity_20261002.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
