"""Audit published MemoryAgentBench claims (papers/search/mab_claims.csv) against item-sampling noise.

For every row with numeric system/baseline scores and n, compute:
  z_unpaired = gap / sqrt(p1 q1/n + p2 q2/n)            (independent samples; conservative for same items)
  z_paired   = gap / sqrt(d/n), d = |gap| + 0.10        (same items; discordant fraction assumed = net gap
                                                         + 10 points, i.e. 5 % of items flip each way beyond
                                                         the net gap; a low-discordance, claim-friendly choice)
A claim is "within noise" if |z| < 1.96 under the given model. Item noise only: rerun, rebuild, order and
version drift add to it (see version_diff.py: up to 8 pts in MAB's own tables), so this undercounts.
Usage: python claims_audit.py papers/search/mab_claims.csv reports/data/claims_audit.json
"""
import csv, json, math, sys


def f(x):
    try:
        return float(str(x).replace("%", "").strip())
    except ValueError:
        return None


rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
# Reproduction checks, not improvement claims (checked by hand 2026-10-15)
NOT_CLAIMS = [("2610.09193", "re-implementation"), ("2609.10263", "Table 1 batch")]
rows = [r for r in rows if not any(r["source_id"] == s and k in r["system"] for s, k in NOT_CLAIMS)]
out = []
for r in rows:
    a, b, n = f(r["system_score"]), f(r["baseline_score"]), f(r["n_questions"])
    if a is None or b is None or not n or n < 5:
        continue
    if a <= 1 and b <= 1:
        a, b = a * 100, b * 100
    if not (0 <= a <= 100 and 0 <= b <= 100):
        continue
    gap = a - b
    if gap <= 0:
        continue  # only audit claimed improvements
    p1, p2 = a / 100, b / 100
    se_u = math.sqrt(max(p1 * (1 - p1), 1e-4) / n + max(p2 * (1 - p2), 1e-4) / n)
    d = abs(gap) / 100 + 0.10
    se_p = math.sqrt(d / n)
    out.append({"source": r["source_id"], "dataset": r["dataset"], "system": r["system"], "baseline": r["baseline"],
                "a": round(a, 2), "b": round(b, 2), "gap": round(gap, 2), "n": int(n),
                "copied": r["baseline_copied"], "unc": r["reports_uncertainty"], "sota": r["sota_claim"],
                "z_u": round(gap / 100 / se_u, 2), "z_p": round(gap / 100 / se_p, 2)})
tot = len(out)


def frac(sel, key):
    s = [o for o in out if sel(o)]
    return len(s), sum(abs(o[key]) < 1.96 for o in s)


summ = {}
for name, sel in [("all", lambda o: True), ("sota", lambda o: o["sota"].lower().startswith("y")),
                  ("copied", lambda o: o["copied"].lower() in ("yes", "mixed")),
                  ("no_uncertainty", lambda o: o["unc"].lower().startswith("n")),
                  ("gap_lt_5", lambda o: o["gap"] < 5), ("n_le_100", lambda o: o["n"] <= 100)]:
    k, wu = frac(sel, "z_u")
    _, wp = frac(sel, "z_p")
    summ[name] = {"n_claims": k, "within_noise_unpaired": wu, "within_noise_paired": wp}
    print(f"{name:15s} claims={k:4d}  within noise: unpaired {wu} ({wu/max(k,1):.0%}), paired-optimistic {wp} ({wp/max(k,1):.0%})")
gaps = sorted(o["gap"] for o in out)
print("median claimed gap:", gaps[len(gaps) // 2], "pts; sources:", len({o['source'] for o in out}))
by = {}
for o in out:
    by.setdefault(o["source"], []).append(o)
by_source = sorted(({"source": s, "n_claims": len(cs), "within": sum(abs(c["z_u"]) < 1.96 for c in cs),
                     "sota": any(c["sota"].lower().startswith("y") for c in cs),
                     "copied": any(c["copied"].lower() in ("yes", "mixed") for c in cs),
                     "unc": any(c["unc"].lower().startswith("y") for c in cs)} for s, cs in by.items()),
                   key=lambda x: (-x["within"] / x["n_claims"], -x["n_claims"]))
print(len(by_source), "sources;", sum(2 * x["within"] >= x["n_claims"] for x in by_source), "with >= half of claims within noise")
import random, statistics
fr = [x["within"] / x["n_claims"] for x in by_source]
rng = random.Random(0)
draws = sorted(statistics.mean(abs(rng.choice(cs)["z_u"]) < 1.96 for cs in by.values()) for _ in range(5000))
heads = []
for cs in by.values():
    cs2 = [c for c in cs if c["sota"].lower().startswith("y")] or cs
    heads.append(max(cs2, key=lambda c: (c["n"], c["gap"])))
robust = {"source_weighted": round(statistics.mean(fr), 3), "one_per_source_mean": round(statistics.mean(draws), 3),
          "one_per_source_90": [draws[250], draws[4750]],
          "headline_within": sum(abs(c["z_u"]) < 1.96 for c in heads), "headline_n": len(heads)}
print("robustness:", robust)
json.dump({"robustness": robust, "summary": summ, "claims": out, "median_gap": gaps[len(gaps) // 2], "by_source": by_source},
          open(sys.argv[2], "w"), indent=1)
