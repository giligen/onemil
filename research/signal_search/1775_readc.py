"""PREREG 1,775 read (c): same two vetoes on the 1,774 book (10:30 cover at the measured ask), VAL. Net = r_ask1030 - 4 fees - 9 locate bp."""
import importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("s", "/home/ec2-user/onemil/research/signal_search/1775_score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
H = S.HERE
ev = pd.read_parquet(f"{H}/1775_events.parquet")[["day", "symbol", "category", "strength"]]
e = pd.read_parquet(f"{H}/1774_events.parquet"); e = e[e.gap_ok & e.r_ask1030.notna()].merge(ev, on=["day", "symbol"], how="inner")
e["net"] = e.r_ask1030 - 13.0; out = []
for sp in ("TRAIN", "VAL"):
    d = e[e.split == sp].reset_index(drop=True)
    for lab, v in (("cat_PR_OTHER", d.category == "PR_OTHER"), ("strength_ge4", d.strength >= 4)):
        v = v.values; o, lo, hi = S.or_ci(d, v); x = d.net.values; k = int(np.ceil(.05 * len(x)))
        xa = x[~v]; ka = int(np.ceil(.05 * len(xa))); rng = np.random.default_rng(1)
        dw = [np.sort(x[np.delete(np.arange(len(x)), rng.choice(len(x), v.sum(), replace=False))])[:ka].mean() - np.sort(x)[:k].mean() for _ in range(500)]
        out.append(dict(split=sp, veto=lab, n=len(d), dropped=int(v.sum()), mean=x.mean(), mean_after=xa.mean(), worst5=np.sort(x)[:k].mean(),
                        worst5_after=np.sort(xa)[:ka].mean(), placebo_dworst5_p95=np.percentile(dw, 95), OR=o, lo=lo, hi=hi))
r = pd.DataFrame(out); r.to_csv(f"{H}/1775_readc.csv", index=False); print(r.round(1).to_string())
