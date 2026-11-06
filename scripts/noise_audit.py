"""Zero-cost noise audit of MemoryAgentBench Table 3 (v4).

Treats each reported cell as a proportion estimated from n items and asks which reported
gaps / rankings / headline claims survive item-sampling noise alone. Item-sampling noise
is a LOWER bound: it ignores run-to-run, build, order and prompt variance and the
clustering of questions on one context. `deff` inflates the variance to mimic those.

Usage: python noise_audit.py reports/data/mab_table3.csv reports/data/noise_audit.json
"""
import csv, json, math, sys, random

# Items behind each column (from HF data + paper; FC assumed 100 = one context length).
N = {"SH-QA": 100, "MH-QA": 100, "LME(S*)": 300, "EventQA": 500, "MCC": 500,
     "Recom.": 200, "Summ.": 100, "DetQA": 71, "FC-SH": 100, "FC-MH": 100}
COMP = {"AR": ["SH-QA", "MH-QA", "LME(S*)", "EventQA"], "TTL": ["MCC", "Recom."],
        "LRU": ["Summ.", "DetQA"], "CR": ["FC-SH", "FC-MH"]}
SD_CONT = {"Summ.": 0.25}  # per-item SD for judge-F1 summarisation (assumed)


def load(path):
    rows, seen = [], set()
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        for row in r:
            if row[0] in seen or row[0] == "GPT-4o-mini":  # duplicate reference row of 4o-mini (128K)
                continue
            seen.add(row[0])
            cols = {}
            # header has repeated "Avg." so map by position
            names = ["SH-QA", "MH-QA", "LME(S*)", "EventQA", "AR", "MCC", "Recom.", "TTL",
                     "Summ.", "DetQA", "LRU", "FC-SH", "FC-MH", "CR", "Overall"]
            for k, v in zip(names, row[2:]):
                cols[k] = float(v)
            rows.append({"agent": row[0], "group": row[1], **cols})
    return rows


def se(col, p, deff=1.0):
    p = p / 100
    sd = SD_CONT.get(col, math.sqrt(max(p * (1 - p), 1e-4)))
    return 100 * sd * math.sqrt(deff / N[col])


def draw(row, deff, rng):
    d = {}
    for col in N:
        d[col] = min(100, max(0, rng.gauss(row[col], se(col, row[col], deff))))
    for c, cols in COMP.items():
        d[c] = sum(d[x] for x in cols) / len(cols)
    d["Overall"] = sum(d[c] for c in COMP) / 4
    return d


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0] * len(vals)
    for k, i in enumerate(order):
        r[i] = k + 1
    return r


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1
    return s / (n * (n - 1) / 2)


CLAIMS = [  # (label, metric, A, B): claim "A > B on metric"
    ("BM25 RAG beats its GPT-4o-mini long-context backbone on AR", "AR", "BM25", "GPT-4o-mini (128K)"),
    ("Long context (4o-mini) beats BM25 on TTL", "TTL", "GPT-4o-mini (128K)", "BM25"),
    ("Long context (4o-mini) beats BM25 on LRU", "LRU", "GPT-4o-mini (128K)", "BM25"),
    ("HippoRAG-v2 beats 4o-mini on AR", "AR", "HippoRAG-v2", "GPT-4o-mini (128K)"),
    ("4o-mini LC beats Mem0 overall", "Overall", "GPT-4o-mini (128K)", "Mem0"),
    ("MemGPT beats Mem0 overall", "Overall", "MemGPT", "Mem0"),
    ("Zep beats Cognee overall", "Overall", "Zep", "Cognee"),
    ("Mem0 beats Cognee overall", "Overall", "Mem0", "Cognee"),
    ("HippoRAG-v2 beats BM25 overall", "Overall", "HippoRAG-v2", "BM25"),
    ("BM25 beats 4o-mini LC on FC-SH", "FC-SH", "BM25", "GPT-4o-mini (128K)"),
    ("HippoRAG-v2 beats 4o-mini LC on FC-SH", "FC-SH", "HippoRAG-v2", "GPT-4o-mini (128K)"),
    ("Claude-3.7 beats GPT-4o overall", "Overall", "Claude-3.7-Sonnet (200K)", "GPT-4o (128K)"),
    ("GPT-4.1-mini beats Claude-3.7 on AR", "AR", "GPT-4.1-mini (1M)", "Claude-3.7-Sonnet (200K)"),
    ("Text-Embed-3-Large beats Text-Embed-3-Small overall", "Overall", "Text-Embed-3-Large", "Text-Embed-3-Small"),
    ("Qwen3-Embedding-4B beats Text-Embed-3-Large overall", "Overall", "Qwen3-Embedding-4B", "Text-Embed-3-Large"),
]


def main(inp, out, B=4000, seed=0):
    rows = load(inp)
    names = [r["agent"] for r in rows]
    idx = {n: i for i, n in enumerate(names)}
    res = {"n_items": N, "agents": names, "reported": rows, "scenarios": {}}
    # adjacent-pair gaps within each column (unpaired 95%)
    adj = {}
    for col in list(N) + ["Overall"]:
        srt = sorted(rows, key=lambda r: -r[col])
        tot = ins = 0
        for a, b in zip(srt, srt[1:]):
            if col == "Overall":
                continue
            tot += 1
            s = math.sqrt(se(col, a[col]) ** 2 + se(col, b[col]) ** 2)
            ins += (a[col] - b[col]) < 1.96 * s
        if tot:
            adj[col] = {"adjacent_pairs": tot, "within_noise": ins}
    res["adjacent_within_noise"] = adj
    rep_rank = ranks([r["Overall"] for r in rows])
    for deff in (1.0, 2.0, 4.0):
        rng = random.Random(seed)
        rank_counts = [[0] * len(rows) for _ in rows]
        taus, same_top3, claim_hits = [], 0, [0] * len(CLAIMS)
        for _ in range(B):
            ds = [draw(r, deff, rng) for r in rows]
            rk = ranks([d["Overall"] for d in ds])
            for i, k in enumerate(rk):
                rank_counts[i][k - 1] += 1
            taus.append(kendall(rk, rep_rank))
            top3 = {i for i, k in enumerate(rk) if k <= 3}
            same_top3 += top3 == {i for i, k in enumerate(rep_rank) if k <= 3}
            for c, (_, m, a, b) in enumerate(CLAIMS):
                claim_hits[c] += ds[idx[a]][m] > ds[idx[b]][m]
        taus.sort()
        res["scenarios"][str(deff)] = {
            "rank_prob": {names[i]: [x / B for x in rank_counts[i]] for i in range(len(rows))},
            "p_rank_unchanged": {names[i]: rank_counts[i][rep_rank[i] - 1] / B for i in range(len(rows))},
            "kendall_tau_median": taus[B // 2], "kendall_tau_p05": taus[int(0.05 * B)],
            "p_same_top3": same_top3 / B,
            "claims": [{"claim": l, "metric": m, "a": a, "b": b,
                        "reported_gap": round(rows[idx[a]][m] - rows[idx[b]][m], 1),
                        "p_holds": claim_hits[c] / B} for c, (l, m, a, b) in enumerate(CLAIMS)],
        }
    res["reported_rank"] = dict(zip(names, rep_rank))
    json.dump(res, open(out, "w"), indent=1)
    s = res["scenarios"]
    print("adjacent within noise:", adj)
    for d in s:
        print(f"deff={d} tau_med={s[d]['kendall_tau_median']:.3f} tau_p05={s[d]['kendall_tau_p05']:.3f} "
              f"top3_same={s[d]['p_same_top3']:.2f}")
        for c in s[d]["claims"]:
            print(f"   {c['p_holds']:.2f}  gap={c['reported_gap']:+.1f}  {c['claim']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
