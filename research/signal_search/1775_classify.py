"""PREREG 1,775 classifier: Claude Haiku 4.5 labels each pre-market headline (category, strength 1-5, why).

No regex anywhere. <=40 headlines per request, strict JSON, temperature 0, <=5 req/s, labels cached in
1775_labels.parquet (resume-safe, flushed every batch). `python3 1775_classify.py --selftest` runs the 20 asserts;
`python3 1775_classify.py` labels every unlabelled headline in 1775_cache/news/*.parquet then prints DONE.
"""
import glob, json, os, sys, time
import pandas as pd
from dotenv import load_dotenv

ROOT = "/home/ec2-user/onemil"; HERE = f"{ROOT}/research/signal_search"
LABELS = f"{HERE}/1775_labels.parquet"
MODEL, BATCH, MIN_GAP = "claude-haiku-4-5-20251001", 40, 0.2
CATS = ["MA", "DILUTION", "FDA", "CONTRACT", "EARNINGS", "ANALYST", "PR_OTHER"]

RUBRIC = """You label pre-market stock news headlines for a trading study. For EACH headline return one category and a strength.
Categories (descriptions, not keyword lists; pick the single best fit by meaning):
MA: mergers, acquisitions, buyouts, takeovers, company to be acquired, definitive merger agreement.
DILUTION: share offerings, pricing of an offering, registered direct, private placement, warrants, at-the-market program,
  dilution, reverse split, shelf registration, convertible notes.
FDA: FDA or regulatory decisions, clinical trial phases and results, approvals, breakthrough designations, PDUFA dates, topline data.
CONTRACT: commercial contracts, partnerships, collaborations, agreements with a named counterparty, awards, orders, licensing deals.
EARNINGS: earnings, quarterly results, fiscal period results, guidance, revenue reports.
ANALYST: analyst upgrades, downgrades, initiations, price target changes.
PR_OTHER: anything else (market-movers roundups, generic company announcements, product news, leadership, etc.).
Strength 1-5: 5 = binding, company-changing, verified (FDA approval, signed M&A, large contract with a named counterparty);
4 = substantial and specific but not final; 3 = moderate; 2 = minor; 1 = promotional, vague or recycled (e.g. generic movers lists).
Return ONLY a JSON array, one object per input in the same order: [{"i":<input index>,"category":"<one of the 7>","strength":<1-5>,"why":"<=12 words"}]. No other text."""


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_client():
    """Anthropic client using the valid key from .env (never printed)."""
    load_dotenv(f"{ROOT}/.env", override=True)
    import anthropic
    return anthropic.Anthropic(max_retries=3)


def parse(txt, n):
    """Strict JSON array -> list of (category, strength, why) or None if unparseable / wrong shape."""
    s = txt.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    out = [None] * n
    for o in arr:
        try:
            i, c, st = int(o["i"]), o["category"], int(o["strength"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= i < n) or c not in CATS or not 1 <= st <= 5 or out[i] is not None:
            return None
        out[i] = (c, st, str(o.get("why", ""))[:120])
    return out if all(out) else None


def classify_batch(cl, heads):
    """Label <=40 headlines; one re-ask on parse failure, then NONE labels (logged WARNING)."""
    body = "\n".join(f'{i}: {h}' for i, h in enumerate(heads))
    for att in range(2):
        time.sleep(MIN_GAP)
        r = cl.messages.create(model=MODEL, max_tokens=4000, temperature=0, system=RUBRIC,
                               messages=[{"role": "user", "content": body}])
        res = parse(r.content[0].text, len(heads))
        if res:
            return res
        log("WARN unparseable Haiku output, attempt", att + 1, "n", len(heads))
    log("ERROR giving up on batch -> NONE labels", len(heads))
    return [("NONE", 0, "unparseable")] * len(heads)


def selftest():
    """20 asserts on sample headlines (live Haiku call, one request)."""
    cases = [
        ("Acme Corp to be acquired by Big Co in $1.2B all-cash deal under definitive agreement", {"MA"}),
        ("XYZ Announces Pricing of $10 Million Public Offering of Common Stock and Warrants", {"DILUTION"}),
        ("ABC Announces Registered Direct Offering Priced At-the-Market", {"DILUTION"}),
        ("FDA Approves BioCo's Drug for Rare Disease", {"FDA"}),
        ("BioCo Reports Positive Topline Results from Phase 3 Trial", {"FDA"}),
        ("Tech Inc Awarded $50 Million Contract by US Department of Defense", {"CONTRACT"}),
        ("Firm Announces Strategic Collaboration with Pharma Giant", {"CONTRACT"}),
        ("Retailer Reports Q3 Earnings, Raises Full-Year Guidance", {"EARNINGS"}),
        ("Company Posts Fiscal Fourth Quarter Revenue of $12 Million", {"EARNINGS"}),
        ("Analyst Upgrades Stock to Buy, Raises Price Target to $20", {"ANALYST"}),
        ("Goldman Initiates Coverage of Chipmaker With Neutral Rating", {"ANALYST"}),
        ("Firm Announces 1-for-10 Reverse Stock Split", {"DILUTION"}),
        ("12 Industrials Stocks Moving In Thursday's Pre-Market Session", {"PR_OTHER"}),
        ("Company Names New Chief Financial Officer", {"PR_OTHER"}),
        ("Company Closes $25 Million Convertible Note Financing", {"DILUTION"}),
        ("Merger of Equals Creates Combined Company, Shareholders to Receive Shares", {"MA"}),
        ("Receives FDA Breakthrough Therapy Designation", {"FDA"}),
        ("Signs Exclusive License Agreement With University", {"CONTRACT"}),
        ("Launches New Mobile App Version 3.0", {"PR_OTHER"}),
        ("Receives Purchase Order from Major Retailer for 5,000 Units", {"CONTRACT"}),
    ]
    res = classify_batch(get_client(), [c[0] for c in cases])
    bad = 0
    for (h, ok), (c, st, why) in zip(cases, res):
        if c not in ok:
            bad += 1; log("ASSERT FAIL", h, "->", c, st)
    assert res[0][1] >= 4 and res[3][1] >= 4, "strength of binding M&A / FDA approval must be >= 4"
    assert res[12][1] <= 2, "generic movers list must be strength <= 2"
    assert bad == 0, f"{bad} category asserts failed"
    log("SELFTEST PASS 20/20")


def main():
    """Label every unlabelled headline id found in the news cache."""
    fs = sorted(glob.glob(f"{HERE}/1775_cache/news/*.parquet"))
    news = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    news = news[news.id.notna() & ~news.id.isin(["", "FAIL"])].drop_duplicates("id")
    have = pd.read_parquet(LABELS) if os.path.exists(LABELS) else pd.DataFrame(columns=["id", "category", "strength", "why"])
    todo = news[~news.id.isin(set(have.id))]
    log("headlines", len(news), "labelled", len(have), "todo", len(todo))
    cl = get_client(); rows = have.to_dict("records")
    for b in range(0, len(todo), BATCH):
        ch = todo.iloc[b:b + BATCH]
        res = classify_batch(cl, ch.headline.fillna("").tolist())
        rows += [dict(id=i, category=c, strength=s, why=w) for i, (c, s, w) in zip(ch.id, res)]
        pd.DataFrame(rows).to_parquet(LABELS + ".tmp"); os.replace(LABELS + ".tmp", LABELS)
        if (b // BATCH) % 10 == 0:
            log("labelled", len(rows), "/", len(have) + len(todo))
    log("DONE")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
