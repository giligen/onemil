# Cell 1,700y: Volatility-Scaled Momentum

## GREF Reference (e=1.0)
- CAGR: 29.4% (target 29.3%) ✓
- Max DD: −38.3% (target −38.3%) ✓  
- Ratio: 0.77 (target 0.77) ✓

## Cell Performance vs GREF

| Target | Cap | CAGR | Max DD | Ratio | Exp into Worst 10 | Pass |
|--------|-----|------|--------|-------|-------------------|------|
| 20% | 1.0 | 15.2% | −31.3% | 0.48 | 0.45 | FAIL |
| 20% | 1.5 | 14.8% | −34.7% | 0.43 | 0.45 | OWNER |
| 30% | 1.0 | 20.1% | −37.1% | 0.54 | 0.68 | FAIL |
| 30% | 1.5 | 20.2% | −44.9% | 0.45 | 0.68 | OWNER |
| 40% | 1.0 | 25.5% | −37.1% | 0.69 | 0.86 | FAIL |
| 40% | 1.5 | 23.7% | −50.5% | 0.47 | 0.91 | OWNER |

## Episode Drawdowns
| Cell | 2021-02..05 | 2020-02..03 | 2025-02..04 |
|------|-------------|-------------|-------------|
| t20c10 | −13.6% | −31.3% | −20.3% |
| t30c10 | −20.3% | −37.1% | −29.2% |
| t40c10 | −26.9% | −37.1% | −33.0% |

## Verdict (Cap 1.0 only)
**All cells FAIL**: vol scaling cannot improve max DD ≥ 8pt over GREF (−38.3%), and achieves ratio < 0.87 target (max 0.69 at t40c10). De-risking sacrifices CAGR without reducing crash DD; crashes occur during high-vol regimes when exposure scales down. Cap 1.5 cells worsen both metrics (leverage amplifies negative tail). Defect: the sleeve's DD is structural, unrepaired by its own vol.
