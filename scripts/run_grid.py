"""Crossed rebuild × order × item runner on top of the MemoryAgentBench harness (Part B).

For each context c, ingestion order o (o=0 native; o>0 = seeded chunk permutation) and rebuild k, build a fresh
agent (no cache reuse), memorize the chunks, answer every question, and append one JSONL row per question to
  <out>/<agent>/<sub_dataset>/c{c}_o{o}_k{k}.jsonl
Completed cells are skipped (checkpointing). Run from anywhere; the MAB repo path is --repo.

Fix vs. public harness: chunks are memorized WITH context_id (the public code passes None, so Mem0/Cognee/Zep
write and read different keys). --buggy-keying reproduces the public behaviour for the buggy-vs-fixed check.

  python run_grid.py --repo external/MemoryAgentBench --agent_config <yaml> --dataset_config <yaml> \
      --rebuilds 5 --orders 5 --out results/partB [--max_queries N] [--dry-run] [--buggy-keying]
--dry-run makes no API calls: prints chunk counts, token estimates and the first chunk of each order.
"""
import argparse, json, os, random, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--agent_config", required=True)
    ap.add_argument("--dataset_config", required=True)
    ap.add_argument("--rebuilds", type=int, default=1)
    ap.add_argument("--orders", type=int, default=1)
    ap.add_argument("--order_seed_base", type=int, default=1000)
    ap.add_argument("--out", default="results/partB")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--buggy-keying", action="store_true")
    ap.add_argument("--contexts", default="", help="comma-separated context indices to run (default: all)")
    ap.add_argument("--closed-book", action="store_true", help="answer without memorizing anything (no-memory arm)")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    agent_cfg_path, data_cfg_path = os.path.abspath(a.agent_config), os.path.abspath(a.dataset_config)
    sys.path.insert(0, os.path.abspath(a.repo))
    os.chdir(os.path.abspath(a.repo))

    import importlib.metadata as md  # vendored mem0 lacks dist metadata
    _v = md.version
    md.version = lambda n: "0.0-vendored" if n == "mem0ai" else _v(n)
    import yaml, tiktoken
    from conversation_creator import ConversationCreator

    acfg, dcfg = yaml.safe_load(open(agent_cfg_path)), yaml.safe_load(open(data_cfg_path))
    cc = ConversationCreator(acfg, dcfg)
    all_chunks, all_qas = cc.get_chunks(), cc.get_query_and_answers()
    enc = tiktoken.get_encoding("o200k_base")
    tag = "closedbook" if a.closed_book else ("buggy" if a.buggy_keying else "fixed")
    cell_dir = os.path.join(out_root, acfg["agent_name"] + {"closedbook": "_closedbook", "buggy": "_buggykey", "fixed": ""}[tag],
                            dcfg["sub_dataset"])
    os.makedirs(cell_dir, exist_ok=True)

    keep = {int(x) for x in a.contexts.split(",") if x.strip()}
    for c, (chunks, qas) in enumerate(zip(all_chunks, all_qas)):
        if keep and c not in keep:
            continue
        if a.max_queries:
            qas = qas[: a.max_queries]
        for o in range(a.orders):
            seed = a.order_seed_base + o
            ordered = list(chunks)
            if o > 0:
                random.Random(seed).shuffle(ordered)
            for k in range(a.rebuilds):
                path = os.path.join(cell_dir, f"c{c}_o{o}_k{k}.jsonl")
                if os.path.exists(path) and sum(1 for _ in open(path)) >= len(qas):
                    continue
                if a.dry_run:
                    toks = sum(len(enc.encode(x)) for x in ordered)
                    print(f"[dry] c{c} o{o}(seed {seed}) k{k}: {len(ordered)} chunks, {toks} tokens, {len(qas)} questions; "
                          f"first chunk: {ordered[0][:70]!r}")
                    continue
                run_cell(acfg, dcfg, [] if a.closed_book else ordered, qas, c, o, seed, k, path, a.buggy_keying, tag)


def run_cell(acfg, dcfg, ordered, qas, c, o, seed, k, path, buggy, tag):
    from agent import AgentWrapper
    from utils.eval_other_utils import metrics_summarization
    from collections import defaultdict
    save = f"/tmp/run_grid_{os.getpid()}_{c}_{o}_{k}"  # never reused -> no cached memory
    agent = AgentWrapper(acfg, dcfg, load_agent_from=save)
    t0 = time.time()
    for ch in ordered:
        if buggy:
            agent.send_message(ch, memorizing=True)                 # public harness behaviour
        else:
            agent.send_message(ch, memorizing=True, context_id=c)   # fixed: same key as queries
    build_s = time.time() - t0
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    with open(path, "a") as f:
        for qi, qa in enumerate(qas):
            if qi < done:
                continue
            q, ans = qa[0], qa[1]
            qid = qa[2] if len(qa) > 2 else None
            outp = agent.send_message(q, memorizing=False, query_id=qi, context_id=c)
            metrics, results = defaultdict(list), []
            metrics, results = metrics_summarization(outp, q, ans, dcfg, metrics, results, qi, qid)
            rec = results[-1] if results else {"output": None}
            row = {"agent": acfg["agent_name"], "dataset": dcfg["sub_dataset"], "keying": tag, "context_id": c,
                   "order_o": o, "order_seed": seed, "rebuild_k": k, "question_id": qi, "qa_pair_id": qid,
                   "metrics": {m: v[-1] for m, v in metrics.items() if v}, "build_seconds": round(build_s, 1),
                   "record": rec, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
