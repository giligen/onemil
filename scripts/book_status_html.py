#!/usr/bin/env python3
"""Render ``book_status.py --json`` as ONE self-contained HTML page (the /books report; spec docs/book_status_html_spec_20261102.md).

Pure stdlib. The file follows the artifact page contract: it starts with ``<title>``, then the fonts link and the
``<style>``, then the content; there are no doctype / html / head / body tags (the publisher wraps the file).
Usage: ``python3 scripts/book_status_html.py [--json path] [--out reports/book_status_YYYYMMDD.html]``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SESSION_OPEN, SESSION_CLOSE, SQUEEZE_MIN, FLAT_MIN = 9 * 60 + 30, 16 * 60, 14 * 60, 15 * 60 + 55
HOD_ARM_START, HOD_ARM_END = 9 * 60 + 35, 14 * 60   # shaded HOD window on the timeline: first arm after the open to the squeeze
SEV_RANK = {"crit": 3, "warn": 2, "info": 1}
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600"
         "&family=IBM+Plex+Sans:wght@400;500;600&display=swap")

_DARK = """--bg:#0e1218;--panel:#151b24;--panel-2:#1b2330;--line:#263041;--fg:#e6e9ee;--muted:#8a94a6;--accent:#e3a33a;
--gain:#3fb67a;--loss:#e5604f;--warn:#e3a33a;--crit:#ff6b5b;--info:#5b9dd9;
--gain-ink:#3fb67a;--loss-ink:#e5604f;--focus:#e3a33a;"""
_LIGHT = """--bg:#f6f4ef;--panel:#ffffff;--panel-2:#f0ede6;--line:#ddd8cc;--fg:#1a1f27;--muted:#5f6775;--accent:#a8731a;
--gain:#39a46e;--loss:#ce5648;--warn:#cc9235;--crit:#e66052;--info:#528dc3;
--gain-ink:#23764c;--loss-ink:#b23d2e;--focus:#a8731a;"""

CSS = f""":root{{{_DARK}color-scheme:dark;
--f-display:'IBM Plex Sans Condensed','Arial Narrow',system-ui,sans-serif;--f-body:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
--f-mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:light){{:root:not([data-theme="dark"]){{{_LIGHT}color-scheme:light;}}}}
:root[data-theme="light"]{{{_LIGHT}color-scheme:light;}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--fg);font:400 15px/1.5 var(--f-body)}}
.page{{max-width:1100px;margin:0 auto;padding:28px 16px 56px}}
.page>section{{margin-top:36px}}
h1,h2,h3{{text-wrap:balance;margin:0}}
.eyebrow{{font:600 11px/1.2 var(--f-display);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
.mono,td,th,.num{{font-family:var(--f-mono);font-variant-numeric:tabular-nums}}
.muted{{color:var(--muted)}}
.gain{{color:var(--gain-ink)}}.loss{{color:var(--loss-ink)}}
:focus-visible{{outline:2px solid var(--focus);outline-offset:2px;border-radius:2px}}
.masthead{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:10px 24px;padding-bottom:14px;border-bottom:1px solid var(--line)}}
.masthead h1{{font:600 28px/1.1 var(--f-display)}}
.mast-right{{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;font:400 13px var(--f-mono);min-width:0}}
.pill{{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--line);background:var(--panel);padding:3px 10px;border-radius:999px;font:400 12px var(--f-mono)}}
.dot{{width:9px;height:9px;border-radius:50%;background:var(--gain);display:inline-block;flex:none}}
.dot.crit{{background:var(--crit)}}.dot.warn{{background:var(--warn)}}.dot.info{{background:var(--info)}}
.sessionline{{margin:10px 0 0;font:400 13px var(--f-body);color:var(--muted)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:6px;padding:14px 16px;min-width:0}}
.tile.sev-crit{{border-left-color:var(--crit)}}.tile.sev-warn{{border-left-color:var(--warn)}}.tile.sev-info{{border-left-color:var(--info)}}
.tile .big{{font:500 40px/1.1 var(--f-mono);margin:8px 0 6px;overflow-wrap:anywhere}}
.tile .figs{{display:flex;flex-wrap:wrap;gap:2px 14px;font:400 12px var(--f-mono);color:var(--muted)}}
.tile .words{{margin-top:8px;font-size:13px;color:var(--fg);overflow-wrap:anywhere}}
.tile.paused .big{{font-size:18px;font-family:var(--f-display)}}
.sec-head{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:12px}}
.sec-head h2{{font:600 18px/1.2 var(--f-display)}}
.flags{{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}}
.flags li{{display:flex;gap:12px;padding:10px 0;border-bottom:1px solid var(--line);min-width:0}}
.flags .dot{{margin-top:7px}}
.flags .txt{{min-width:0;overflow-wrap:anywhere}}
.flags .why{{display:block;font-size:12px;color:var(--muted)}}
.clear{{padding:14px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);color:var(--muted)}}
.tl{{width:100%;height:auto;display:block;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
.tl text{{font:400 12px var(--f-mono);fill:var(--muted)}}
.tl .rowlabel{{font:600 12px var(--f-display);fill:var(--fg);letter-spacing:.08em}}
.tl .band{{fill:var(--panel-2)}}.tl .band2{{fill:var(--panel)}}
.tl .grid{{stroke:var(--line);stroke-width:1}}
.tl .mark{{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}}
.tl .now{{stroke:var(--accent);stroke-width:2}}.tl .nowlbl{{fill:var(--accent)}}
.tl .fill-dot{{stroke-width:2}}.tl .fill-dot.win{{fill:var(--gain);stroke:var(--gain)}}.tl .fill-dot.lose{{fill:var(--loss);stroke:var(--loss)}}
.tl .fill-dot.open{{fill:none;stroke:var(--fg)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:6px}}
details.book{{border-top:1px solid var(--line)}}
details.book:last-of-type{{border-bottom:1px solid var(--line)}}
details.book>summary{{cursor:pointer;list-style:none;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 16px;padding:14px 0;min-width:0}}
details.book>summary::-webkit-details-marker{{display:none}}
details.book>summary::before{{content:"+";font:500 15px var(--f-mono);color:var(--accent);width:14px;flex:none}}
details.book[open]>summary::before{{content:"\\2212"}}
summary .name{{font:600 15px var(--f-display);letter-spacing:.04em;text-transform:uppercase}}
summary .head{{font:500 15px var(--f-mono)}}
summary .sub{{font:400 12px var(--f-mono);color:var(--muted)}}
.body{{padding:2px 0 22px 24px;min-width:0}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px 28px}}
.cols>*{{min-width:0}}
.kv{{display:flex;flex-wrap:wrap;gap:4px 22px;font:400 13px var(--f-mono);padding:8px 0 14px}}
.kv b{{font-weight:500;color:var(--fg)}}.kv span{{color:var(--muted)}}
.scroll{{overflow-x:auto;max-width:100%}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{font:600 11px var(--f-display);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
td{{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child,td.l,th.l{{text-align:left}}
tr.open td{{background:var(--panel-2)}}
tr.warnrow td{{color:var(--warn)}}
h3.sub-h{{font:600 11px var(--f-display);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:18px 0 6px}}
svg.chart{{width:100%;height:auto;display:block}}
svg.chart text{{font:400 12px var(--f-mono);fill:var(--muted)}}
svg.chart .zero{{stroke:var(--muted);stroke-width:1}}
svg.chart .bar.p{{fill:var(--gain)}}svg.chart .bar.n{{fill:var(--loss)}}
svg.chart .line{{fill:none;stroke:var(--fg);stroke-width:2}}
svg.chart .end{{fill:var(--accent)}}svg.chart text.endlbl{{fill:var(--fg)}}
.gauge{{position:relative;height:96px;margin:6px 0 4px}}
.gauge .track{{position:absolute;left:0;right:0;top:34px;height:14px;background:var(--panel-2);border-radius:3px;overflow:hidden}}
.gauge .z1{{position:absolute;top:0;bottom:0;background:var(--crit);opacity:.28}}
.gauge .z2{{position:absolute;top:0;bottom:0;background:var(--warn);opacity:.18}}
.gauge .tick{{position:absolute;top:28px;height:26px;width:2px;background:var(--muted)}}
.gauge .cur{{position:absolute;top:24px;height:34px;width:4px;background:var(--accent);margin-left:-2px;border-radius:1px}}
.gauge .lbl{{position:absolute;font:400 12px var(--f-mono);color:var(--muted);white-space:nowrap}}
.gauge .lbl.top{{top:2px;color:var(--fg)}}.gauge .lbl.bot{{top:62px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}}
.chip{{font:500 12px var(--f-mono);border:1px solid var(--line);padding:2px 8px;border-radius:3px}}
.chip.in{{border-color:var(--gain);color:var(--gain-ink)}}.chip.out{{border-color:var(--loss);color:var(--loss-ink)}}
.wbar{{display:inline-block;height:8px;vertical-align:middle;border-radius:1px}}
.wcell{{width:90px;text-align:left}}
.ctrl{{background:none;border:1px solid var(--line);color:var(--fg);font:400 12px var(--f-body);padding:4px 12px;border-radius:4px;cursor:pointer}}
.ctrl:hover{{border-color:var(--accent)}}
.pauses{{list-style:none;margin:0;padding:0;font:400 13px var(--f-mono)}}
.pauses li{{padding:6px 0;border-bottom:1px solid var(--line)}}
footer{{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}}
@media (max-width:560px){{.tile .big{{font-size:28px}}.body{{padding-left:0}}.masthead h1{{font-size:28px}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
"""

JS = """(function(){var K='onemil-books:';function g(k){try{return localStorage.getItem(K+k)}catch(e){return null}}
function s(k,v){try{localStorage.setItem(K+k,v)}catch(e){}}
var ds=document.querySelectorAll('details.book');
ds.forEach(function(d){var v=g(d.dataset.key);if(v==='1')d.open=true;else if(v==='0')d.open=false;
d.addEventListener('toggle',function(){s(d.dataset.key,d.open?'1':'0')})});
document.querySelectorAll('[data-all]').forEach(function(b){b.addEventListener('click',function(){
var o=b.dataset.all==='open';ds.forEach(function(d){d.open=o})})})})();"""


# --------------------------------------------------------------------------- formatting helpers
def esc(x) -> str:
    """HTML-escape any value (``None`` -> empty)."""
    return escape("" if x is None else str(x), quote=True)


def money(x: Optional[float], signed: bool = True) -> str:
    """``+$1,234`` / ``-$46`` (whole dollars); ``n/a`` for ``None``."""
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ("+" if signed else "")
    return f"{sign}${abs(x):,.0f}"


def price(x: Optional[float]) -> str:
    """Price with 2 decimals (3 under $1); ``-`` for ``None``."""
    if x is None:
        return "-"
    return f"{x:,.3f}" if abs(x) < 1 else f"{x:,.2f}"


def sign_cls(x: Optional[float]) -> str:
    """CSS class for the sign of a number (never applied to a whole tile)."""
    return "" if not x else ("gain" if x > 0 else "loss")


def num(x, fmt: str = "{:,.2f}") -> str:
    """Format a number or ``-`` when missing."""
    return "-" if x is None else fmt.format(x)


def hm(t: Optional[str]) -> int:
    """``HH:MM[:SS]`` -> minutes since midnight."""
    h, m = t.split(":")[:2]
    return int(h) * 60 + int(m)


def clock_minutes(t: str) -> float:
    """``HH:MM[:SS]`` -> fractional minutes since midnight."""
    parts = [int(p) for p in t.split(":")]
    return parts[0] * 60 + parts[1] + (parts[2] / 60 if len(parts) > 2 else 0.0)


def ok(sec) -> bool:
    """True when a data section is a populated dict (not the ``unavailable`` stub)."""
    return isinstance(sec, dict) and "unavailable" not in sec


def unavailable(sec, label: str) -> str:
    """Muted one-liner for a section that failed to build."""
    why = sec.get("unavailable", "no data") if isinstance(sec, dict) else "no data"
    return f'<p class="muted">{esc(label)} unavailable: {esc(why)}</p>'


def generated(data: dict) -> dt.datetime:
    """Report timestamp (ET, naive) parsed from ``generated_at_et``; falls back to the session date at 00:00."""
    raw = data.get("generated_at_et")
    try:
        return dt.datetime.fromisoformat(raw).replace(tzinfo=None)
    except (TypeError, ValueError):
        return dt.datetime.fromisoformat(data.get("session_date", "1970-01-01"))


def session_state(now: dt.datetime) -> str:
    """Words for where the clock is relative to the 09:30-16:00 ET session and the 15:55 flat."""
    minute = now.hour * 60 + now.minute
    if now.weekday() >= 5:
        return "no session (weekend)"
    if minute < SESSION_OPEN:
        return f"before the open, {SESSION_OPEN - minute} min to 09:30"
    if minute < SESSION_CLOSE:
        left = FLAT_MIN - minute
        return f"session open, {left} min to the flat" if left > 0 else "session open, positions flat"
    return "after the close"


def book_sev(flags: List[dict], book: str) -> str:
    """Worst flag severity for a tile's book key ('' when none)."""
    sevs = [f["severity"] for f in flags if f.get("book") == book]
    return max(sevs, key=lambda s: SEV_RANK[s]) if sevs else ""


# --------------------------------------------------------------------------- tiles
def tile(key: str, eyebrow: str, big: str, big_cls: str, figs: List[str], words: str, sev: str, extra: str = "") -> str:
    """One TL;DR tile: eyebrow, big number, two small figures, a line in words, severity stripe."""
    figs_html = "".join(f"<span>{esc(f)}</span>" for f in figs)
    return (f'<div class="tile {extra} sev-{sev}" data-tile="{key}"><div class="eyebrow">{esc(eyebrow)}</div>'
            f'<div class="big {big_cls}">{esc(big)}</div><div class="figs">{figs_html}</div>'
            f'<div class="words">{esc(words)}</div></div>')


def hod_orb_tile(data: dict, key: str, label: str, sev: str) -> str:
    """HOD / ORB tile: closed P&L today, WTD and since-9/29 figures, exit counts in words."""
    sec = data.get(key)
    acct = (sec or {}).get("account") if isinstance(sec, dict) else None
    badge = (acct or "paper").upper()
    risk = ((sec or {}).get("gate") or {}).get("risk_usd") if isinstance(sec, dict) else None
    eyebrow = f"{label} · {badge}" + (f" ${risk:,.0f} R" if risk else "")
    if not ok(sec):
        return tile(key, eyebrow, "n/a", "muted", [], (sec or {}).get("unavailable", "no data") if isinstance(sec, dict) else "no data", sev)
    today, wtd = sec["today"], sec["wtd"]
    if key == "hod":
        since = sec.get("since_0929", {})
        figs = [f"WTD {money(wtd['pnl'])}", f"since 9/29 {money(since.get('pnl'))} ({since.get('fills', 0)})"]
        ex = sec.get("exits", {})
        words = f"{today.get('open', 0)} open · {ex.get('stops', 0)} stop · {ex.get('squeeze', 0)} squeezed"
    else:
        cum = sum(d["pnl"] for d in sec.get("daily", []))
        figs = [f"WTD {money(wtd['pnl'])}", f"last {len(sec.get('daily', []))} sessions {money(cum)}"]
        words = f"{today.get('fills', 0)} fills today · {len(sec.get('vetoes', []))} vetoes"
    return tile(key, eyebrow, money(today["pnl"]), sign_cls(today["pnl"]), figs, words, sev)


def mom_tile(data: dict, which: str, sev: str) -> str:
    """MOM tile: equity % vs start, equity and pause-rail distance, this rotation's in/out names."""
    b = (data.get("mom") or {}).get(which)
    eyebrow = f"MOM · {which.upper()}"
    if not ok(b) or "equity" not in b:
        return tile(f"mom_{which}", eyebrow, "n/a", "muted", [], (b or {}).get("unavailable", "no data"), sev)
    pct = b.get("pct_vs_start")
    figs = [f"${b['equity']:,.0f}", f"pause rail {money(b.get('to_pause'))}"]
    words = f"in {' '.join(b['names_in']) or '-'} · out {' '.join(b['names_out']) or '-'}"
    return tile(f"mom_{which}", eyebrow, "n/a" if pct is None else f"{pct:+.1f}%", sign_cls(pct), figs, words, sev)


def paused_tile(data: dict) -> str:
    """Compact tile listing the paused books (BF and R2G) with their config state."""
    p = data.get("paused") or {}
    lines = [f"BF trading.enabled = {str(p.get('bf_enabled')).lower()}", f"R2G red_to_green.enabled = {str(p.get('r2g_enabled')).lower()}"]
    body = "".join(f"<div>{esc(x)}</div>" for x in lines)
    return (f'<div class="tile paused sev-" data-tile="paused"><div class="eyebrow">Paused</div>'
            f'<div class="big">BF · R2G</div><div class="words mono">{body}</div></div>')


# --------------------------------------------------------------------------- sections
def render_masthead(data: dict) -> str:
    """Masthead: title, date/time ET, service pill, one session-state line."""
    now = generated(data)
    ops = data.get("ops") or {}
    active = ops.get("service_active")
    errs = ops.get("errors_today", 0)
    dot = "dot" if (active and not errs) else "dot crit"
    pill = (f'<span class="pill"><span class="{dot}"></span>{esc(ops.get("service_state") or "unknown")} · boot '
            f'{esc(ops.get("boot_utc") or "?")} UTC · {errs} errors</span>')
    stamp = now.strftime("%a ") + str(now.day) + now.strftime(" %b %Y · %H:%M ET")
    return (f'<header><div class="masthead"><h1>OneMil books</h1><div class="mast-right"><span>{esc(stamp)}</span>{pill}</div></div>'
            f'<p class="sessionline">{esc(session_state(now))}</p></header>')


def render_tiles(data: dict) -> str:
    """The five TL;DR tiles in a responsive grid."""
    fl = data.get("flags", [])
    parts = [hod_orb_tile(data, "hod", "HOD", book_sev(fl, "hod")), hod_orb_tile(data, "orb", "ORB", book_sev(fl, "orb")),
             mom_tile(data, "paper", book_sev(fl, "mom_paper")), mom_tile(data, "live", book_sev(fl, "mom_live")), paused_tile(data)]
    return f'<section aria-label="Summary"><div class="tiles">{"".join(parts)}</div></section>'


def render_flags(data: dict) -> str:
    """Attention list: severity dot, text, muted rule; an explicit empty state."""
    fl = data.get("flags", [])
    head = '<div class="sec-head"><div><div class="eyebrow">Attention</div><h2>What needs a look</h2></div></div>'
    if not fl:
        return f'<section>{head}<div class="clear">Nothing needs action.</div></section>'
    items = "".join(f'<li><span class="dot {esc(f["severity"])}"></span><span class="txt">{esc(f["text"])}'
                    f'<span class="why">{esc(f["severity"].upper())} · {esc(f.get("book"))} · {esc(f.get("why"))}</span></span></li>'
                    for f in fl)
    return f'<section>{head}<ul class="flags">{items}</ul></section>'


def render_timeline(data: dict) -> str:
    """Inline SVG: 09:30-16:00 ET to scale, HOD/ORB rows, arm band, 14:00 squeeze and 15:55 flat markers, now marker, fill dots."""
    W, H, X0, X1 = 1000, 190, 64, 976
    span = SESSION_CLOSE - SESSION_OPEN

    def x(minute: float) -> float:
        """Minute-of-day -> x pixel."""
        return X0 + (minute - SESSION_OPEN) / span * (X1 - X0)

    rows = {"HOD": 70, "ORB": 128}
    out = [f'<svg id="timeline" class="tl" viewBox="0 0 {W} {H}" role="img" aria-label="Session timeline of fills">']
    out.append(f'<rect class="band" x="{x(HOD_ARM_START):.1f}" y="38" width="{x(HOD_ARM_END) - x(HOD_ARM_START):.1f}" height="64"><title>HOD arm window 09:35-14:00</title></rect>')
    for h in range(10, 17):
        out.append(f'<line class="grid" x1="{x(h * 60):.1f}" x2="{x(h * 60):.1f}" y1="30" y2="150"/>')
    out.append(f'<line class="grid" x1="{x(SESSION_OPEN):.1f}" x2="{x(SESSION_OPEN):.1f}" y1="30" y2="150"/>')
    ticks = [(SESSION_OPEN, "09:30")] + [(h * 60, f"{h:02d}:00") for h in range(10, 17)]
    for m, lbl in ticks:
        anchor = "start" if m == SESSION_OPEN else ("end" if m == SESSION_CLOSE else "middle")
        out.append(f'<text x="{x(m):.1f}" y="170" text-anchor="{anchor}">{lbl}</text>')
    out.append(f'<line class="mark mark-squeeze" x1="{x(SQUEEZE_MIN):.1f}" x2="{x(SQUEEZE_MIN):.1f}" y1="30" y2="150"><title>14:00 squeeze</title></line>')
    out.append(f'<text x="{x(SQUEEZE_MIN):.1f}" y="24" text-anchor="middle">14:00 squeeze</text>')
    out.append(f'<line class="mark mark-flat" x1="{x(FLAT_MIN):.1f}" x2="{x(FLAT_MIN):.1f}" y1="30" y2="150"><title>15:55 flat</title></line>')
    out.append(f'<text x="{x(FLAT_MIN):.1f}" y="24" text-anchor="end">15:55 flat</text>')
    now = generated(data)
    nm = now.hour * 60 + now.minute
    if data.get("session_date") == now.strftime("%Y-%m-%d") and now.weekday() < 5 and SESSION_OPEN <= nm <= SESSION_CLOSE:
        out.append(f'<line class="now" x1="{x(nm):.1f}" x2="{x(nm):.1f}" y1="30" y2="150"><title>now {now:%H:%M} ET</title></line>')
        out.append(f'<text class="nowlbl" x="{x(nm) + 4:.1f}" y="40">now</text>')
    for label, y in rows.items():
        out.append(f'<text class="rowlabel" x="6" y="{y + 4}">{label}</text><line class="grid" x1="{X0}" x2="{X1}" y1="{y}" y2="{y}"/>')
    for key, label in (("hod", "HOD"), ("orb", "ORB")):
        sec = data.get(key)
        if not ok(sec):
            continue
        for f in sec.get("fills", []):
            if not f.get("entry_ts_et"):
                continue
            pnl = f.get("pnl")
            cls = "open" if pnl is None else ("win" if pnl > 0 else "lose")
            tip = f'{f["symbol"]} {f["entry_ts_et"]} @ {price(f.get("fill_price"))} ' + ("open" if pnl is None else money(pnl))
            out.append(f'<circle class="fill-dot {cls}" cx="{x(clock_minutes(f["entry_ts_et"])):.1f}" '
                       f'cy="{rows[label]}" r="6"><title>{esc(tip)}</title></circle>')
    out.append("</svg>")
    legend = ('<div class="legend"><span>filled dot: closed, green win / red loss</span><span>hollow dot: open</span>'
              '<span>shaded: HOD arm window</span></div>')
    return (f'<section><div class="sec-head"><div><div class="eyebrow">Session</div><h2>Fills on the clock</h2></div></div>'
            f'{"".join(out)}{legend}</section>')


def fills_table(fills: List[dict]) -> str:
    """Fills table (time, symbol, fill, stop, R$, shares, spread bps, exit, P&L); open rows marked."""
    if not fills:
        return '<p class="muted">No fills today.</p>'
    cols = ["Time", "Symbol", "Fill", "Stop", "R$", "Shares", "Spread bps", "Exit", "P&amp;L"]
    head = "".join('<th class="l">%s</th>' % h if i == 1 else "<th>%s</th>" % h for i, h in enumerate(cols))
    rows = []
    for f in fills:
        pnl = f.get("pnl")
        is_open = pnl is None and not f.get("exit_reason")
        pcell = '<td class="muted">open</td>' if is_open else f'<td class="{sign_cls(pnl)}">{esc(money(pnl))}</td>'
        rows.append(f'<tr class="{"open" if is_open else ""}"><td>{esc(f.get("entry_ts_et") or "-")}</td><td class="l">{esc(f["symbol"])}</td>'
                    f'<td>{esc(price(f.get("fill_price")))}</td><td>{esc(price(f.get("stop")))}</td><td>{esc(num(f.get("risk_usd"), "{:,.0f}"))}</td>'
                    f'<td>{esc(num(f.get("shares"), "{:,.0f}"))}</td><td>{esc(num(f.get("entry_quote_spread_bps"), "{:,.0f}"))}</td>'
                    f'<td>{esc(f.get("exit_reason") or "-")}</td>{pcell}</tr>')
    return f'<div class="scroll"><table class="fills"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def bar_chart(daily: List[dict]) -> str:
    """Last-N-sessions P&L bars to scale with a zero line; labels only at the extremes and every other date."""
    if not daily:
        return '<p class="muted">No closed sessions yet.</p>'
    W, H, top, bot = 400, 190, 22, 150
    vals = [d["pnl"] for d in daily]
    hi, lo = max(max(vals), 0.0), min(min(vals), 0.0)
    rng = (hi - lo) or 1.0
    y = lambda v: top + (hi - v) / rng * (bot - top)  # noqa: E731 - tiny scale helper
    step = (W - 20) / len(daily)
    out = [f'<svg class="chart bars" viewBox="0 0 {W} {H}" role="img" aria-label="P&amp;L by session">',
           f'<line class="zero" x1="10" x2="{W - 10}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>']
    imax, imin = vals.index(max(vals)), vals.index(min(vals))
    for i, d in enumerate(daily):
        v = d["pnl"]
        x0 = 10 + i * step + step * 0.15
        y0, y1 = sorted((y(0), y(v)))
        out.append(f'<rect class="bar {"p" if v >= 0 else "n"}" x="{x0:.1f}" y="{y0:.1f}" width="{step * 0.7:.1f}" height="{max(y1 - y0, 1):.1f}">'
                   f'<title>{esc(d["date"])} {esc(money(v))} ({d["fills"]} fills)</title></rect>')
        if i % 2 == len(daily) % 2 or i == len(daily) - 1:
            out.append(f'<text x="{x0 + step * 0.35:.1f}" y="{H - 14}" text-anchor="middle">{esc(d["date"][5:].replace("-", "/"))}</text>')
        if i in (imax, imin) and v:
            ly = y(v) - 5 if v > 0 else y(v) + 14
            out.append(f'<text x="{x0 + step * 0.35:.1f}" y="{ly:.1f}" text-anchor="middle">{esc(money(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def sparkline(cum: List[dict]) -> str:
    """Cumulative P&L line since the book's start with the endpoint emphasised and labelled."""
    if len(cum) < 2:
        return '<p class="muted">Not enough closed sessions for a curve.</p>'
    W, H, top, bot = 400, 190, 22, 150
    vals = [c["cum_pnl"] for c in cum]
    hi, lo = max(max(vals), 0.0), min(min(vals), 0.0)
    rng = (hi - lo) or 1.0
    px = lambda i: 14 + i * (W - 28 - 56) / (len(vals) - 1)  # noqa: E731
    py = lambda v: top + (hi - v) / rng * (bot - top)  # noqa: E731
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    ex = px(len(vals) - 1)
    return (f'<svg class="chart spark" viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative P&amp;L since {esc(cum[0]["date"])}">'
            f'<line class="zero" x1="10" x2="{W - 10}" y1="{py(0):.1f}" y2="{py(0):.1f}"/><polyline class="line" points="{pts}"/>'
            f'<circle class="end" cx="{ex:.1f}" cy="{py(vals[-1]):.1f}" r="5"/>'
            f'<text class="endlbl" x="{ex + 9:.1f}" y="{py(vals[-1]) + 4:.1f}">{esc(money(vals[-1]))}</text>'
            f'<text x="14" y="{H - 14}">{esc(cum[0]["date"][5:].replace("-", "/"))}</text>'
            f'<text x="{ex:.1f}" y="{H - 14}" text-anchor="end">{esc(cum[-1]["date"][5:].replace("-", "/"))}</text></svg>')


def summary(name: str, head: str, sub: str, head_cls: str = "") -> str:
    """``<summary>`` repeating the tile's headline so a collapsed section still reads."""
    return (f'<summary><span class="name">{esc(name)}</span><span class="head {head_cls}">{esc(head)}</span>'
            f'<span class="sub">{esc(sub)}</span></summary>')


def details(key: str, summ: str, body: str, is_open: bool) -> str:
    """One collapsible drilldown block."""
    return f'<details class="book" data-key="{key}"{" open" if is_open else ""}>{summ}<div class="body">{body}</div></details>'


def render_hod(data: dict) -> str:
    """HOD drilldown: gate card, fills table, P&L bars beside the cumulative sparkline."""
    sec = data.get("hod")
    if not ok(sec):
        return details("hod", summary("HOD", "unavailable", ""), unavailable(sec, "HOD"), False)
    g, t = sec.get("gate", {}), sec["today"]
    kv = [("ARM SKIP r_pct", g.get("arm_skip_r_pct")), ("ARM SKIP min_price", g.get("arm_skip_min_price")),
          ("ARM SKIP symbols", g.get("arm_skip_symbols")), ("cap hits", g.get("cap_hits")),
          ("SQUEEZE lines", g.get("squeeze_lines")), ("squeeze enabled", g.get("squeeze_enabled")),
          ("concurrent cap", g.get("concurrency_cap")), ("per-day cap", g.get("per_day_cap")),
          ("risk $", g.get("risk_usd")), ("dry_run", g.get("dry_run")), ("exits", " ".join(f"{k} {v}" for k, v in sec.get("exits", {}).items()))]
    gate = '<div class="kv">' + "".join(f"<div><span>{esc(k)}</span> <b>{esc('n/a' if v is None else v)}</b></div>" for k, v in kv) + "</div>"
    body = (f'<h3 class="sub-h">Gates and caps</h3>{gate}<h3 class="sub-h">Fills today</h3>{fills_table(sec.get("fills", []))}'
            f'<div class="cols"><div><h3 class="sub-h">Last 12 sessions, P&amp;L</h3>{bar_chart(sec.get("daily", []))}</div>'
            f'<div><h3 class="sub-h">Cumulative since 9/29</h3>{sparkline(sec.get("cum_since_start", []))}</div></div>')
    return details("hod", summary("HOD paper", money(t["pnl"]), f'{t["fills"]} fills · {t.get("open", 0)} open', sign_cls(t["pnl"])),
                   body, t["fills"] > 0)


def render_orb(data: dict) -> str:
    """ORB drilldown: fills table, vetoes list, ramp state line."""
    sec = data.get("orb")
    if not ok(sec):
        return details("orb", summary("ORB", "unavailable", ""), unavailable(sec, "ORB"), False)
    t = sec["today"]
    vet = sec.get("vetoes", [])
    vrows = "".join(f'<tr><td class="l">{esc(v["symbol"])}</td><td class="l">{esc(v["reason"])}</td></tr>' for v in vet)
    vtab = (f'<div class="scroll"><table><thead><tr><th class="l">Symbol</th><th class="l">Reason</th></tr></thead><tbody>{vrows}</tbody></table></div>'
            if vet else '<p class="muted">No vetoes logged today.</p>')
    ramp = sec.get("ramp")
    rline = ("ramp: no freeze file" if ramp is None else
             f'ramp: {"FROZEN" if ramp.get("frozen") else "not frozen"}' + (f' — {ramp.get("reason")}' if ramp.get("reason") else ""))
    body = (f'<h3 class="sub-h">Fills today</h3>{fills_table(sec.get("fills", []))}<h3 class="sub-h">Vetoes</h3>{vtab}'
            f'<h3 class="sub-h">Ramp</h3><p class="mono" style="margin:0;font-size:13px">{esc(rline)}</p>'
            f'<h3 class="sub-h">Last 12 sessions, P&amp;L</h3><div style="max-width:520px">{bar_chart(sec.get("daily", []))}</div>')
    return details("orb", summary("ORB paper", money(t["pnl"]), f'{t["fills"]} fills · {len(vet)} vetoes', sign_cls(t["pnl"])),
                   body, t["fills"] > 0)


def gauge(b: dict) -> str:
    """Horizontal rail gauge from the halve rail (left) through the pause rail to start equity (right), current equity marked."""
    lo, start, eq = b["halve_rail"], b["start_equity"], b["equity"]
    hi = max(start, eq) * 1.02
    pos = lambda v: max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))  # noqa: E731
    pause = b["pause_rail"]
    return (f'<div class="gauge"><div class="track"><div class="z1" style="left:0;width:{pos(pause):.1f}%"></div>'
            f'<div class="z2" style="left:{pos(pause):.1f}%;width:{pos(start) - pos(pause):.1f}%"></div></div>'
            f'<div class="tick" style="left:{pos(pause):.1f}%"></div><div class="tick" style="left:{pos(start):.1f}%"></div>'
            f'<div class="cur" style="left:{pos(eq):.1f}%"></div>'
            f'<div class="lbl top" style="left:{pos(eq):.1f}%;transform:translateX(-{50 if 12 < pos(eq) < 88 else (0 if pos(eq) <= 12 else 100)}%)">${eq:,.0f}</div>'
            f'<div class="lbl bot" style="left:0">halve ${lo:,.0f}</div>'
            f'<div class="lbl bot" style="left:{pos(pause):.1f}%;transform:translateX(-50%)">pause ${pause:,.0f}</div>'
            f'<div class="lbl bot" style="right:0">start ${start:,.0f}</div></div>'
            f'<div class="kv"><div><span>above pause rail</span> <b>{esc(money(b.get("to_pause")))}</b></div>'
            f'<div><span>above halve rail</span> <b>{esc(money(b.get("to_halve")))}</b></div>'
            f'<div><span>vs start</span> <b class="{sign_cls(b.get("pct_vs_start"))}">{esc(num(b.get("pct_vs_start"), "{:+.1f}%"))}</b></div></div>')


def mom_column(which: str, b) -> str:
    """One MOM book column: gauge, rotation chips, weekly table with bars, parity table."""
    if not ok(b) or "equity" not in b:
        return f'<div><h3 class="sub-h">{which}</h3>{unavailable(b, "MOM " + which)}</div>'
    chips = ("".join(f'<span class="chip in">{esc(n)}</span>' for n in b["names_in"]) or '<span class="muted">none in</span>') + \
            '<span class="muted">/</span>' + ("".join(f'<span class="chip out">{esc(n)}</span>' for n in b["names_out"]) or '<span class="muted">none out</span>')
    weekly = b.get("weekly", [])
    mx = max([abs(w["week_pct"]) for w in weekly if w.get("week_pct") is not None] + [0.01])
    wr = "".join(
        f'<tr><td>{esc(w["date"])}</td><td>{esc(money(w["equity"], False))}</td>'
        f'<td class="{sign_cls(w.get("week_pct"))}">{esc(num(w.get("week_pct"), "{:+.1f}%"))}</td>'
        f'<td>{esc(num(w.get("spy_pct"), "{:+.1f}%"))}</td><td class="wcell">'
        + (f'<span class="wbar" style="width:{abs(w["week_pct"]) / mx * 80:.0f}px;background:var(--{"gain" if w["week_pct"] >= 0 else "loss"})"></span>' if w.get("week_pct") is not None else "")
        + "</td></tr>" for w in weekly)
    wtab = ('<div class="scroll"><table><thead><tr><th>Date</th><th>Equity</th><th>Week</th><th>SPY</th><th class="l"></th></tr></thead>'
            f'<tbody>{wr}</tbody></table></div>') if weekly else '<p class="muted">No weekly rows.</p>'
    par = b.get("parity", [])
    pr = "".join(f'<tr class="{"warnrow" if p["names_n"] < 18 or abs(p["resid_pp"]) > 2 else ""}"><td>{esc(p["date"])}</td>'
                 f'<td>{p["names_n"]}/20</td><td>{p["resid_pp"]:+.2f}</td></tr>' for p in par)
    ptab = ('<div class="scroll"><table><thead><tr><th>Date</th><th>Names</th><th>Resid pp</th></tr></thead>'
            f'<tbody>{pr}</tbody></table></div>') if par else '<p class="muted">No parity rows.</p>'
    return (f'<div><h3 class="sub-h">MOM {esc(which)} · rails</h3>{gauge(b)}<h3 class="sub-h">This rotation (in / out)</h3>'
            f'<div class="chips">{chips}</div><h3 class="sub-h">Weekly</h3>{wtab}<h3 class="sub-h">BT parity</h3>{ptab}</div>')


def render_mom(data: dict) -> str:
    """MOM drilldown: paper and live columns side by side (stacked on a phone)."""
    mom = data.get("mom")
    if not ok(mom):
        return details("mom", summary("MOM", "unavailable", ""), unavailable(mom, "MOM"), False)
    paper, live = mom.get("paper"), mom.get("live")
    heads = []
    for lbl, b in (("paper", paper), ("live", live)):
        if ok(b) and b.get("pct_vs_start") is not None:
            heads.append(f"{lbl} {b['pct_vs_start']:+.1f}%")
    date = data.get("session_date")
    active = any(ok(b) and b.get("last_rebalance") == date for b in (paper, live))
    body = f'<div class="cols">{mom_column("paper", paper)}{mom_column("live", live)}</div>'
    return details("mom", summary("MOM sleeve", " · ".join(heads) or "n/a", f"last rebalance {(paper or {}).get('last_rebalance', '?')}"), body, active)


def render_paused(data: dict) -> str:
    """Paused books: one line each with the config key and value."""
    p = data.get("paused")
    if not ok(p):
        return details("paused", summary("Paused books", "unavailable", ""), unavailable(p, "Paused"), False)
    body = (f'<ul class="pauses"><li>BF — trading.enabled = {esc(str(p.get("bf_enabled")).lower())}</li>'
            f'<li>R2G — red_to_green.enabled = {esc(str(p.get("r2g_enabled")).lower())}</li></ul>')
    return details("paused", summary("Paused books", "BF · R2G", "config.yaml"), body, False)


def render_ops(data: dict) -> str:
    """Operations: journal stats, boot time, crons table."""
    ops = data.get("ops")
    if not ok(ops):
        return details("ops", summary("Operations", "unavailable", ""), unavailable(ops, "Ops"), False)
    kv = (f'<div class="kv"><div><span>journal lines (filtered pass)</span> <b>{ops.get("journal_lines_today", 0):,}</b></div>'
          f'<div><span>ERROR</span> <b>{ops.get("errors_today", 0)}</b></div><div><span>WARNING</span> <b>{ops.get("warnings_today", 0)}</b></div>'
          f'<div><span>boot</span> <b>{esc(ops.get("boot_utc") or "?")} UTC</b></div></div>')
    rows = "".join(f'<tr class="{"warnrow" if c.get("overdue") or c.get("errors_today") else ""}"><td class="l">{esc(c["name"])}</td>'
                   f'<td class="l">{esc(c["schedule"])}</td><td class="l">{esc(c.get("log") or "-")}</td><td>{esc(c.get("last_write") or "-")}</td>'
                   f'<td>{c.get("errors_today", 0)}</td></tr>' for c in ops.get("crons", []))
    tab = ('<div class="scroll"><table><thead><tr><th class="l">Name</th><th class="l">Schedule</th><th class="l">Log</th>'
           f'<th>Last write</th><th>Errors today</th></tr></thead><tbody>{rows}</tbody></table></div>')
    flagged = any(f["severity"] != "info" and f.get("book") == "ops" for f in data.get("flags", [])) or ops.get("errors_today", 0) > 0
    return details("ops", summary("Operations", f'{ops.get("errors_today", 0)} errors · {ops.get("warnings_today", 0)} warnings',
                                  f'{len(ops.get("crons", []))} crons'), kv + tab, flagged)


def render_footer(data: dict) -> str:
    """Sources and generation stamp."""
    return (f'<footer>Sources: data/trades.db (read-only), onemil-trader journal (today, filtered), logs/momentum_sleeve_weekly.csv, '
            f'logs/momentum_sleeve_live_weekly.csv, logs/mom_bt_parity.csv, config.yaml, crontab and cron logs. '
            f'Generated by scripts/book_status_html.py at {esc(data.get("generated_at_et"))}.</footer>')


def render(data: dict) -> str:
    """The full page: title, fonts link, style, then content (no doctype / html / head / body tags)."""
    drill = (f'<section><div class="sec-head"><div><div class="eyebrow">Drilldown</div><h2>Book by book</h2></div>'
             f'<div><button class="ctrl" type="button" data-all="open">Expand all</button> '
             f'<button class="ctrl" type="button" data-all="close">Collapse all</button></div></div>'
             f'{render_hod(data)}{render_orb(data)}{render_mom(data)}{render_paused(data)}{render_ops(data)}</section>')
    return (f'<title>OneMil Books</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link href="{FONTS}" rel="stylesheet">\n'
            f'<style>{CSS}</style>\n<div class="page">{render_masthead(data)}{render_tiles(data)}{render_flags(data)}'
            f'{render_timeline(data)}{drill}{render_footer(data)}</div>\n<script>{JS}</script>\n')


def load_data(json_path: Optional[str]) -> dict:
    """Read the book_status JSON from a file, or run ``scripts/book_status.py --json`` (read-only) when no path is given."""
    if json_path:
        return json.loads(Path(json_path).read_text())
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "book_status.py"), "--json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: render to ``--out`` (default ``reports/book_status_<session date>.html``) and print path + size."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="book_status --json file (default: run book_status.py)")
    ap.add_argument("--out", default=None, help="output .html path")
    args = ap.parse_args(argv)
    data = load_data(args.json)
    out = Path(args.out) if args.out else ROOT / "reports" / f"book_status_{data.get('session_date', 'unknown').replace('-', '')}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render(data)
    out.write_text(html, encoding="utf-8")
    print(f"{out} {len(html.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
