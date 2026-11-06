"""Per-split resolution limits for MemoryAgentBench: minimum detectable difference between two systems.

MDD95 (50% power, two-sided .05) and MDD80 (80% power) for an unpaired comparison at accuracy p, and for a paired
comparison with discordant fraction d (both systems answer the same items). Clustering on one context is ignored,
so these are optimistic lower bounds. Writes reports/data/resolution_limits.json and paper/resolution_table.tex.
"""
import json, math
SPLITS = [("FactConsolidation (each length)", 100, 1), ("RULER-QA1 / QA2", 100, 1), ("ICL (each set)", 100, 1),
          ("DetectiveQA", 71, 10), ("InfBench-Sum", 100, 100), ("Recsys", 200, 1), ("LME(S*)", 300, 5), ("EventQA (one length)", 500, 5)]
rows = []
for name, n, ctx in SPLITS:
    p = 0.5
    se_u = math.sqrt(2 * p * (1 - p) / n)
    r = {"split": name, "n": n, "contexts": ctx,
         "mdd95_unpaired": round(100 * 1.96 * se_u, 1), "mdd80_unpaired": round(100 * 2.80 * se_u, 1)}
    for d in (0.1, 0.2):
        r[f"mdd80_paired_d{int(d*100)}"] = round(100 * 2.80 * math.sqrt(d / n), 1)
    rows.append(r)
json.dump(rows, open("reports/data/resolution_limits.json", "w"), indent=1)
lines = [f"{r['split']} & {r['n']} & {r['contexts']} & {r['mdd95_unpaired']} & {r['mdd80_unpaired']} & {r['mdd80_paired_d10']}--{r['mdd80_paired_d20']} \\\\" for r in rows]
tex = ("\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n"
       "Split & $n$ & contexts & MDD$_{95}$ & MDD$_{80}$ & MDD$_{80}$ paired \\\\\\midrule\n" + "\n".join(lines) +
       "\n\\bottomrule\\end{tabular}\n\\caption{Smallest accuracy gap (points) that MAB splits can resolve between two single runs at "
       "$p=0.5$: MDD$_{95}$ is significant half the time; MDD$_{80}$ gives 80\\% power. The paired column assumes 10--20\\% "
       "discordant items. Binomial approximation (InfBench-Sum and Recsys are graded metrics); clustering on few contexts makes all of these optimistic.}\n\\label{tab:mdd}\n\\end{table}\n")
open("paper/resolution_table.tex", "w").write(tex)
for r in rows: print(r)
