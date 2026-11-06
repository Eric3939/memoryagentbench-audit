"""Offline (no-API) checks backing reports/harness_audit.md.

Drives the real MemoryAgentBench harness code (ConversationCreator, AgentWrapper memorize paths,
LC truncation, post_process scoring) on the cached HF data. No LLM / embedding / network calls:
OPENAI_API_KEY is a dummy, HF is offline, and only code paths that do not call an API are run.

Run:  HF_HUB_DISABLE_PROGRESS_BARS=1 external/mabenv/bin/python -I env/check_harness_audit.py <MAB repo root> [--fast]
"""
import ast, os, re, sys, json, asyncio, types
from collections import Counter

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                  OPENAI_API_KEY="sk-dummy-no-calls", TOKENIZERS_PARALLELISM="false")
ROOT = os.path.abspath(sys.argv[1]); FAST = "--fast" in sys.argv
sys.path.insert(0, ROOT); os.chdir(ROOT)
import yaml, tiktoken
from conversation_creator import ConversationCreator
from agent import AgentWrapper
from utils.eval_other_utils import post_process, normalize_answer
ENC = tiktoken.encoding_for_model("gpt-4o-mini")
ntok = lambda s: len(ENC.encode(s, disallowed_special=()))
Y = lambda p: yaml.safe_load(open(p))
A = "configs/agent_conf/RAG_Agents/gpt-4o-mini/"; L = "configs/agent_conf/Long_Context_Agents/"
D = "configs/data_conf/"

MAIN = {  # main-table tiers (bash_files/configs/rag_agents.txt)
    "RULER-QA1": D + "Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml",
    "RULER-QA2": D + "Accurate_Retrieval/Ruler/QA/Ruler_qa2_421k.yaml",
    "EventQA-full": D + "Accurate_Retrieval/EventQA/Eventqa_full.yaml",
    "LME-S*": D + "Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml",
    "FC-SH-262K": D + "Conflict_Resolution/Factconsolidation_sh_262k.yaml",
    "FC-MH-262K": D + "Conflict_Resolution/Factconsolidation_mh_262k.yaml",
    "ICL-banking77": D + "Test_Time_Learning/ICL/ICL_banking77.yaml",
    "DetQA": D + "Long_Range_Understanding/Detective_QA.yaml",
    "InfBench-Sum": D + "Long_Range_Understanding/InfBench_sum.yaml",
}
if not FAST:
    MAIN["Recsys"] = D + "Test_Time_Learning/Recsys/Recsys_redial_full.yaml"

out = {}
def section(t): print("\n" + "=" * 8 + " " + t + " " + "=" * 8, flush=True)

# ---------------------------------------------------------------- 1. RAG chunk-store cap
section("1. RAG chunk store: chunks kept by _handle_rag_agent (oldest dropped when n*chunk_size > limit)")
RAG_CFGS = ["Embedding_rag_gpt-4o-mini-text_embedding_3_small.yaml", "Structure_rag_gpt-4o-mini-raptor.yaml",
            "Agentic_memory_gpt-4o-mini-self_rag.yaml", "Structure_rag_gpt-4o-mini-graph-rag.yaml",
            "Structure_rag_gpt-4o-mini-memo_rag.yaml", "Simple_rag_gpt-4o-mini-bm25.yaml"]
res1 = []
for dname, dpath in MAIN.items():
    dcfg = Y(dpath)
    for cs in (4096, 512):
        dc = dict(dcfg, chunk_size=cs, max_test_samples=min(dcfg["max_test_samples"], 10))
        acfg0 = Y(A + RAG_CFGS[0])
        cc = ConversationCreator(acfg0, dc)
        ctx_chunks = cc.get_chunks()
        nctx = len(ctx_chunks)
        tokc = [{c: ntok(c) for c in cl} for cl in ctx_chunks]
        for cfgname in RAG_CFGS:
            acfg = Y(A + cfgname)
            kept_tok = tot_tok = kept_n = tot_n = 0
            for ci in range(nctx):
                ag = AgentWrapper(acfg, dc, load_agent_from="/nonexistent")
                for ch in ctx_chunks[ci]:
                    ag.send_message(ch, memorizing=True)
                tot_n += len(ctx_chunks[ci]); kept_n += len(ag.chunks)
                tot_tok += sum(tokc[ci].values())
                kept_tok += sum(tokc[ci][c] for c in ctx_chunks[ci][len(ctx_chunks[ci]) - len(ag.chunks):])  # kept = suffix
            row = dict(dataset=dname, chunk=cs, system=acfg["agent_name"], limit=ag.input_length_limit,
                       chunks_total=tot_n, chunks_kept=kept_n, frac_tokens_kept=round(kept_tok / tot_tok, 3))
            res1.append(row)
            if row["frac_tokens_kept"] < 1:
                print(row, flush=True)
    print(f"[{dname}] done", flush=True)
out["rag_cap"] = res1

# ---------------------------------------------------------------- 2. LC truncation
section("2. Long-context truncation (_truncate_context_if_needed keeps the TAIL)")
res2 = []
for dname, dpath in MAIN.items():
    dcfg = dict(Y(dpath), max_test_samples=1)
    for lc in ["Long_context_agent_gpt-4o-mini.yaml", "Long_context_agent_gpt-4.1-mini.yaml",
               "Long_context_agent_claude-3-7-sonnet-20250219.yaml", "Long_context_agent_gemini-2.0-flash.yaml"]:
        acfg = Y(L + lc)
        cc = ConversationCreator(acfg, dcfg)
        chunks = cc.get_chunks()[0]
        ag = AgentWrapper.__new__(AgentWrapper)  # avoid creating SDK clients for claude/gemini
        ag.agent_name = acfg["agent_name"]; ag.sub_dataset = dcfg["sub_dataset"]
        ag.context_max_length = dcfg["context_max_length"]; ag.context = ""
        ag.input_length_limit = acfg["input_length_limit"] - acfg["buffer_length"] - dcfg["generation_max_length"]
        for ch in chunks:
            ag._handle_long_context_agent(ch, memorizing=True)
        full = ntok(ag.context); head = ag.context[:200]
        if ag.input_length_limit <= ag.context_max_length + 50000:   # same guard as _query_long_context_agent
            ag._truncate_context_if_needed(ENC)
        kept = ntok(ag.context)
        row = dict(dataset=dname, model=acfg["model"], ctx_tokens=full, limit=ag.input_length_limit,
                   kept_tokens=kept, frac_kept=round(kept / full, 3), head_survives=ag.context.startswith(head[:100]))
        res2.append(row); print(row, flush=True)
out["lc_trunc"] = res2

# ---------------------------------------------------------------- 3. Zep ingestion coverage
section("3. Zep: share of each memorized chunk that reaches graph.add (content[:9998]) and thread (content[:2400])")
from utils.templates import get_template
res3 = []
for dname, dpath in MAIN.items():
    dcfg = dict(Y(dpath), max_test_samples=1)
    acfg = Y(A + "Structure_rag_gpt-4o-mini-zep.yaml")
    cc = ConversationCreator(acfg, dcfg)
    tmpl = get_template(dcfg["sub_dataset"], "memorize", acfg["agent_name"])
    tot = g = t = 0
    for ch in cc.get_chunks()[0]:
        content = tmpl.format(context=ch, **({"time_stamp": "2025-01-01 00:00:00"} if "{time_stamp}" in tmpl else {}))
        tot += len(content); g += min(len(content), 9998); t += min(len(content), 2400)
    row = dict(dataset=dname, chars_total=tot, graph_frac=round(g / tot, 3), thread_frac=round(t / tot, 3))
    res3.append(row); print(row, flush=True)
out["zep"] = res3

# ---------------------------------------------------------------- 4. Scoring
section("4. Scoring behaviour of post_process on constructed outputs")
def score(sub, output, gold):
    m, _ = post_process({"output": output}, gold, {"sub_dataset": sub})
    return {k: (round(float(v), 3)) for k, v in m.items() if k in ("exact_match", "substring_exact_match", "eventqa_recall")}
cases = [
    ("icl_banking77_5900shot_balance", "label: 43", ["43"], "ICL, instructed format"),
    ("icl_banking77_5900shot_balance", "43", ["43"], "ICL, bare number"),
    ("icl_banking77_5900shot_balance", "label: 17", ["7"], "ICL, wrong label"),
    ("detective_qa", '{"answer":"C. The Brandt couple", "reasoning":"..."}', ["C. The Brandt couple"], "DetQA, demo JSON format"),
    ("detective_qa", '```json\n{"answer": "C. The Brandt couple"}\n```', ["C. The Brandt couple"], "DetQA, fenced JSON"),
    ("detective_qa", "C. The Brandt couple", ["C. The Brandt couple"], "DetQA, bare option"),
    ("detective_qa", "B. Hila Weber", ["B.Hila Weber"], "DetQA, correct option, gold lacks space"),
    ("factconsolidation_sh_262k", "Armenia (previously Argentina)", ["Argentina"], "FC, hedged old+new"),
    ("ruler_qa2_421K", "I don't know", ["no"], "RULER yes/no gold, abstention"),
    ("ruler_qa2_421K", "Not mentioned in the documents", ["no"], "RULER yes/no gold, abstention"),
    ("ruler_qa2_421K", "Yes", ["no"], "RULER, opposite answer"),
    ("eventqa_full", "Debbie waited for her father at the end of the driveway", ["Debbie waited for her father at the end of the driveway."], "EventQA, no final period"),
]
res4 = []
for sub, o, g, note in cases:
    r = dict(note=note, output=o, gold=g, **score(sub, o, g)); res4.append(r); print(r)
out["scoring_cases"] = res4

# ---------------------------------------------------------------- 5. Gold-label statistics
section("5. Gold-label statistics (HF data)")
from datasets import load_dataset
res5 = {}
for split in ["Accurate_Retrieval", "Conflict_Resolution", "Long_Range_Understanding", "Test_Time_Learning"]:
    ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
    for r in ds:
        src = r["metadata"]["source"]
        st = res5.setdefault(src, Counter())
        for q, a in zip(r["questions"], r["answers"]):
            golds = a if isinstance(a, list) else [a]
            golds = [x for sub in golds for x in (sub if isinstance(sub, list) else [sub])]
            norm = [normalize_answer(x) for x in golds]
            st["questions"] += 1
            if any(n == "" for n in norm): st["gold_normalizes_to_empty"] += 1
            if any(n in ("yes", "no") for n in norm): st["gold_yes_no"] += 1
            if any(len(n) <= 3 for n in norm): st["gold_len_le3"] += 1
            if any(n and n in normalize_answer(q) for n in norm): st["gold_substring_of_question"] += 1
            if min(ntok(x) for x in golds) > 10: st["all_golds_gt10_tokens"] += 1
            if src == "detective_qa" and any(re.match(r"^[A-D]\.\S", x) for x in golds): st["detqa_gold_no_space_after_letter"] += 1
            if src.startswith("eventqa"):
                m = re.search(r"Below is a list of possible subsequent events:\s*(\[.*\])", q, re.S)
                if m:
                    try:
                        opts = ast.literal_eval(m.group(1))
                    except Exception:
                        st["eventqa_options_unparsable"] += 1; continue
                    gn = normalize_answer(golds[0])
                    st["eventqa_n_options_%d" % len(opts)] += 1
                    if any(gn in normalize_answer(o) and normalize_answer(o) != gn for o in opts):
                        st["eventqa_gold_substring_of_distractor"] += 1
                    if not any(normalize_answer(o) == gn for o in opts):
                        st["eventqa_gold_not_among_options"] += 1
for k, v in res5.items():
    print(k, dict(v))
out["gold_stats"] = {k: dict(v) for k, v in res5.items()}

# ---------------------------------------------------------------- 6. Cognee dataset filter
section("6. Cognee search(): is the `datasets` filter able to drop GRAPH_COMPLETION results?")
src = open("cognee/modules/search/methods/search.py").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "search")
ns = {"json": json, "JSONEncoder": json.JSONEncoder, "parse_id": lambda x: x}
calls = {}
async def log_query(*a, **k): return types.SimpleNamespace(id=1)
async def log_result(*a, **k): return None
async def get_document_ids_for_user(uid, datasets):   # mirrors the vendored function for a dataset that does not exist:
    calls["datasets"] = datasets; return []             # dataset_id=None -> no DatasetData rows -> []
async def specific_search(qt, q, user, **k):           # GRAPH_COMPLETION returns [completion_string]
    calls["specific_search_kwargs"] = sorted(k); return ["Argentina"]
ns.update(log_query=log_query, log_result=log_result, get_document_ids_for_user=get_document_ids_for_user,
          specific_search=specific_search, User=object, SearchType=object)
exec(compile(ast.Module(body=[fn], type_ignores=[]), "cognee_search", "exec"), ns)
r = asyncio.run(ns["search"]("q", types.SimpleNamespace(value="GRAPH_COMPLETION"), ["default_dataset_x_context_0"], types.SimpleNamespace(id=0)))
print("filtered result:", r, "| datasets reached specific_search?", "datasets" in calls.get("specific_search_kwargs", []), calls)
out["cognee_filter"] = dict(result=r, calls=calls)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "data", "harness_audit_checks.json"), "w"), indent=1, default=str)
print("\nAUDIT CHECKS DONE")
