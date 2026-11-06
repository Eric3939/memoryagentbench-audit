"""Three-arm Mem0 report: accuracy per arm (public harness / fixed keying / closed-book) and dataset, with exact
paired McNemar tests between arms on the same questions.

Reads results/threearm/<agent>{_buggykey,,_closedbook}/<dataset>/c0_o0_k0.jsonl (written by run_grid.py).
Usage: python threearm_report.py results/threearm [--synthetic]
"""
import glob, json, math, os, sys

ARMS = {"_buggykey": "public", "": "fixed", "_closedbook": "closed-book"}
KEYS = ("substring_exact_match", "exact_match", "Recall@5")


def score(row):
    m = row.get("metrics", {})
    for k in KEYS:
        if k in m:
            return float(m[k])
    return None


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 vectors; returns (b10, b01, p)."""
    b10 = sum(1 for x, y in zip(a, b) if x > y); b01 = sum(1 for x, y in zip(a, b) if y > x); n = b10 + b01
    if n == 0:
        return b10, b01, 1.0
    k = min(b10, b01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return b10, b01, min(1.0, p)


def load(root):
    data = {}
    for p in glob.glob(os.path.join(root, "*", "*", "c0_o0_k0.jsonl")):
        agent_dir, ds = p.split(os.sep)[-3:-1]
        arm = next((v for s, v in ARMS.items() if s and agent_dir.endswith(s)), "fixed")
        rows = {}
        for line in open(p):
            r = json.loads(line); v = score(r)
            if v is not None:
                rows[r["question_id"]] = v
        data.setdefault(ds, {})[arm] = rows
    return data


def synthetic(root):
    import random
    rng = random.Random(0)
    for ds, base in [("factconsolidation_sh_6k", 0.18), ("longmemeval_s*", 0.3)]:
        for suf, lift in [("_buggykey", 0.0), ("", 0.25), ("_closedbook", -0.02)]:
            d = os.path.join(root, "Structure_rag_mem0" + suf, ds); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "c0_o0_k0.jsonl"), "w") as f:
                for i in range(100):
                    v = 1.0 if rng.random() < base + lift else 0.0
                    f.write(json.dumps({"question_id": i, "metrics": {"substring_exact_match": v}}) + "\n")


def main():
    root = sys.argv[1]
    if "--synthetic" in sys.argv:
        synthetic(root)
    data = load(root)
    print("| dataset | n | public | fixed | closed-book | fixed vs public (b10/b01, p) | fixed vs closed-book (p) |")
    print("|---|---|---|---|---|---|---|")
    out = []
    for ds, arms in sorted(data.items()):
        common = sorted(set.intersection(*(set(v) for v in arms.values())))
        vec = {a: [arms[a][q] for q in common] for a in arms}
        acc = {a: 100 * sum(v) / len(v) for a, v in vec.items() if v}
        fp = mcnemar(vec.get("fixed", []), vec.get("public", [])) if {"fixed", "public"} <= set(vec) else None
        fc = mcnemar(vec.get("fixed", []), vec.get("closed-book", [])) if {"fixed", "closed-book"} <= set(vec) else None
        f = lambda a: f"{acc[a]:.1f}" if a in acc else "–"
        print(f"| {ds} | {len(common)} | {f('public')} | {f('fixed')} | {f('closed-book')} | "
              f"{'%d/%d, %.3g' % fp if fp else '–'} | {'%.3g' % fc[2] if fc else '–'} |")
        out.append({"dataset": ds, "n": len(common), "acc": acc, "fixed_vs_public": fp, "fixed_vs_closed": fc})
    json.dump(out, open(os.path.join(root, "threearm_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
