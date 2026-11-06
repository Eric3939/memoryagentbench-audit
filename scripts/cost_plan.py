"""Back-of-envelope API cost for the proposed crossed design on MemoryAgentBench.

Token counts come from the real HF contexts (chars/4). Prices are per 1M tokens (gpt-4o-mini list
price $0.15 in / $0.60 out, cached input $0.075; text-embedding-3-small $0.02) - verify before running.
Assumptions per system (rough, documented inline) are deliberately conservative (no caching for LC).
Usage: python cost_plan.py papers/search/mab_dataset_stats.csv
"""
import csv, sys

P_IN, P_OUT, P_EMB = 0.15, 0.60, 0.02
rows = {r["source"]: r for r in csv.DictReader(open(sys.argv[1]))}


def ctx_tokens(src):
    return float(rows[src]["ctx_kchars_mean"]) * 1000 / 4


def n_q(src):
    return int(rows[src]["n_q"]) / int(rows[src]["n_ctx"])  # questions per context


def lc(src):  # long context: every question re-reads the full context (no caching), short answers
    t = min(ctx_tokens(src), 128_000)
    return n_q(src) * (t * P_IN + 20 * P_OUT) / 1e6


def rag(src, k=10, chunk=512):  # BM25/embedding RAG: top-k chunks per question + embeddings of context once
    return (n_q(src) * (k * chunk * P_IN + 20 * P_OUT) + ctx_tokens(src) * P_EMB) / 1e6


def mem0(src, chunk=4096):  # Mem0 build: per chunk ~1 extraction call (chunk in, ~1.5K out) + ~1 update call (~3K in, 1K out)
    chunks = max(1, ctx_tokens(src) / chunk)
    build = chunks * ((chunk + 1000) * P_IN + 1500 * P_OUT + 3000 * P_IN + 1000 * P_OUT) / 1e6
    query = n_q(src) * (1500 * P_IN + 20 * P_OUT) / 1e6
    return build + query


DATASETS = ["factconsolidation_sh_6k", "factconsolidation_mh_6k", "factconsolidation_sh_32k",
            "factconsolidation_mh_32k", "eventqa_65536"]
K, O = 5, 5  # repeats per order, orders
print(f"{'dataset':28s} {'LC':>8s} {'RAG':>8s} {'Mem0':>8s}   (USD per single build+eval of one context)")
tot = 0
for d in DATASETS:
    c = (lc(d), rag(d), mem0(d))
    print(f"{d:28s} {c[0]:8.3f} {c[1]:8.4f} {c[2]:8.3f}")
    tot += sum(c) * (2 if d.startswith("fact") else 1)  # FC: LC+BM25+embedding(≈RAG)+Mem0 ≈ sum with RAG twice
grid = tot * K * O
print(f"\nOne pass of all systems over these contexts: ${tot:.2f}")
print(f"Crossed design K={K} repeats x O={O} orders: ${grid:.2f}  (x1.5 safety: ${grid*1.5:.2f})")
print("Note: EventQA-64K counts one book; LC cost scales with context length x questions.")
