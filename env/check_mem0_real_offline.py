"""Offline check with the REAL vendored Mem0 (local Qdrant, real user_id filtering); only the LLM and the
embedder are stubbed (no network, no API). Drives the unmodified harness memorize path, then queries:
  public path : send_message(q, context_id=0)           (what main.py does)
  fixed path  : memorize with context_id=0, query with context_id=0
Prints how many memories Mem0 actually returns in each case.
Usage (from anywhere): python check_mem0_real_offline.py <MemoryAgentBench repo root>
"""
import hashlib, json, os, sys, types
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1])
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-no-calls")
os.environ["MEM0_TELEMETRY"] = "False"
import importlib.metadata as md
_v = md.version
md.version = lambda n: "0.0-vendored" if n == "mem0ai" else _v(n)
import numpy as np, yaml
import agent as agent_mod
import initialization
from conversation_creator import ConversationCreator


def fake_embed(text, *a, **k):
    h = hashlib.sha256(text.encode()).digest()
    v = np.frombuffer((h * 48)[:1536 * 1], dtype=np.uint8).astype(float)[:1536]
    v = np.resize(v, 1536)
    return list(v / (np.linalg.norm(v) + 1e-9))


def fake_llm(messages, response_format=None, **k):
    # valid for both the fact-extraction and the memory-update parsers
    txt = " ".join(m["content"] for m in messages if isinstance(m.get("content"), str))
    fact = "fact-" + hashlib.md5(txt.encode()).hexdigest()[:8]
    return json.dumps({"facts": [fact], "memory": [{"id": "0", "text": fact, "event": "ADD"}]})


class StubCompletions:
    def create(self, **k):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="stub"))],
                                     usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))


def build(acfg, dcfg, chunks, memorize_ctx):
    ag = agent_mod.AgentWrapper(acfg, dcfg, load_agent_from=f"/tmp/claude_check_{os.getpid()}")
    ag.memory.llm.generate_response = fake_llm
    ag.memory.embedding_model.embed = fake_embed
    ag.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=StubCompletions()))
    if memorize_ctx is None:
        initialization._memorize_context_chunks(ag, chunks, 0, 1)            # public harness
    else:
        for ch in chunks:
            ag.send_message(ch, memorizing=True, context_id=memorize_ctx)   # fixed
    return ag


acfg = yaml.safe_load(open("configs/agent_conf/RAG_Agents/gpt-4o-mini/Structure_rag_gpt-4o-mini-mem0.yaml"))
dcfg = yaml.safe_load(open("configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml"))
cc = ConversationCreator(acfg, dcfg)
chunks, qas = cc.get_chunks()[0], cc.get_query_and_answers()[0]
for label, mctx in [("public harness", None), ("fixed (context_id passed)", 0)]:
    ag = build(acfg, dcfg, chunks, mctx)
    stored = {u: len(ag.memory.get_all(user_id=u)["results"]) for u in
              [f"context_None_{dcfg['sub_dataset']}", f"context_0_{dcfg['sub_dataset']}"]}
    got = ag.memory.search(query=qas[0][0], user_id=f"context_0_{dcfg['sub_dataset']}", limit=acfg["retrieve_num"])
    print(f"{label}: stored per user_id {stored}; search(user_id=context_0) returned {len(got['results'])} memories")
