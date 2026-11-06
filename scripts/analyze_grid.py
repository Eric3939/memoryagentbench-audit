"""Analyse Part B grid output (results/partB/<agent>/<dataset>/c{c}_o{o}_k{k}.jsonl) -> variance components.

Per (agent, dataset, context): correctness tensor y[item, rebuild, order]; reports
  - score mean, SD of single-cell scores, ever-flip fraction, Always-correct / Always-wrong / Flip counts
  - variance components (scripts/gtheory.components) and SD_run vs SD_item
  - null arm: |score(k=0,o=0) - score(k=1,o=0)| (two independent builds, same order)
  - MDD95 for comparing two single runs: 1.96 * sqrt(2) * SD_run_total, SD_run_total includes item noise
Writes <out>.json and prints a markdown table. `--synthetic DIR` writes fake JSONL for a pipeline test.
Usage: python analyze_grid.py results/partB reports/data/partB_summary
"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gtheory import components, simulate_once  # noqa: E402

CORRECT_KEYS = ("substring_exact_match", "exact_match", "Recall@5")


def correct_of(row):
    m = row.get("metrics", {})
    for k in CORRECT_KEYS:
        if k in m:
            return float(m[k])
    return None


def load(root):
    cells = {}
    for p in glob.glob(os.path.join(root, "*", "*", "c*_o*_k*.jsonl")):
        agent, ds = p.split(os.sep)[-3:-1]
        c, o, k = map(int, re.search(r"c(\d+)_o(\d+)_k(\d+)", os.path.basename(p)).groups())
        for line in open(p):
            r = json.loads(line)
            v = correct_of(r)
            if v is not None:
                cells.setdefault((agent, ds, c), {})[(r["question_id"], k, o)] = v
    return cells


def tensor(d):
    items = sorted({i for i, _, _ in d}); ks = sorted({k for _, k, _ in d}); os_ = sorted({o for _, _, o in d})
    y = np.full((len(items), len(ks), len(os_)), np.nan)
    for (i, k, o), v in d.items():
        y[items.index(i), ks.index(k), os_.index(o)] = v
    keep = ~np.isnan(y).any(axis=(1, 2))  # balanced design: drop items missing any cell
    return y[keep]


def summarise(y):
    n, K, O = y.shape
    res = components(y) if K > 1 or O > 1 else {"mean": float(y.mean())}
    cell_scores = y.mean(0).ravel()
    always_c = int((y.min((1, 2)) == 1).sum()); always_w = int((y.max((1, 2)) == 0).sum())
    res.update({"n_items": n, "K": K, "O": O, "cell_sd": float(cell_scores.std(ddof=1)) if cell_scores.size > 1 else 0.0,
                "always_correct": always_c, "always_wrong": always_w, "flip": n - always_c - always_w})
    if K > 1:
        res["null_arm_gap"] = float(abs(y[:, 0, 0].mean() - y[:, 1, 0].mean()))
    if "sd_run" in res:
        res["mdd95_single_runs"] = float(1.96 * np.sqrt(2) * np.sqrt(res["sd_run"] ** 2 + res["sd_item"] ** 2))
    return res


def write_synthetic(root):
    rng = np.random.default_rng(1)
    for agent, sd_b, sd_o, flip in [("Structure_rag_mem0", 0.5, 0.3, 0.8), ("Simple_rag_bm25", 0.0, 0.05, 0.1)]:
        y = simulate_once(rng, 100, 5, 5, sd_b, sd_o, flip)
        d = os.path.join(root, agent, "factconsolidation_sh_6k"); os.makedirs(d, exist_ok=True)
        for k in range(5):
            for o in range(5):
                with open(os.path.join(d, f"c0_o{o}_k{k}.jsonl"), "w") as f:
                    for i in range(100):
                        f.write(json.dumps({"question_id": i, "metrics": {"substring_exact_match": y[i, k, o]}}) + "\n")


def main():
    if sys.argv[1] == "--synthetic":
        write_synthetic(sys.argv[2]); print("synthetic grid written"); return
    root, out = sys.argv[1], sys.argv[2]
    rows = []
    for (agent, ds, c), d in sorted(load(root).items()):
        s = summarise(tensor(d)); s.update({"agent": agent, "dataset": ds, "context": c}); rows.append(s)
    json.dump(rows, open(out + ".json", "w"), indent=1, default=float)
    print("| agent | dataset | n | K×O | mean | cell SD | SD_run | SD_item | null-arm gap | MDD95 | flip items |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda k: f"{100 * r[k]:.1f}" if k in r else "–"
        print(f"| {r['agent']} | {r['dataset']} | {r['n_items']} | {r['K']}×{r['O']} | {f('mean')} | {f('cell_sd')} | "
              f"{f('sd_run')} | {f('sd_item')} | {f('null_arm_gap')} | {f('mdd95_single_runs')} | {r['flip']} |")


if __name__ == "__main__":
    main()
