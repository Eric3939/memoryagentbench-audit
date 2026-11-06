"""No-API check: which user_id does the MemoryAgentBench harness use when Mem0 memorizes vs. when it queries?

Replaces mem0's Memory with a recorder and the OpenAI client with a stub, then drives the *real* harness path:
initialization._memorize_context_chunks (as main.py does) followed by agent.send_message(query, context_id=0)
(as main.process_single_query does). Prints the user_ids seen by add() and search().
Usage (from repo root): python check_mem0_keying.py <repo_root>
"""
import os, sys, types
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1])
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-no-calls")
import yaml
import importlib.metadata as _md
_orig_version = _md.version
_md.version = lambda name: "0.0-vendored" if name == "mem0ai" else _orig_version(name)  # vendored mem0 has no dist metadata

calls = []


class RecorderMemory:
    def __init__(self, *a, **k):
        self.store = {}

    def add(self, messages, user_id=None, **k):
        calls.append(("add", user_id))
        self.store.setdefault(user_id, []).append("x")
        return {"results": []}

    def search(self, query, user_id=None, limit=100, **k):
        calls.append(("search", user_id, len(self.store.get(user_id, []))))
        return {"results": [{"memory": m} for m in self.store.get(user_id, [])]}


import mem0.memory.main as mm
mm.Memory = RecorderMemory
import agent as agent_mod
from conversation_creator import ConversationCreator
import initialization


class StubCompletions:
    def create(self, **k):
        msg = types.SimpleNamespace(content="stub")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))


acfg = yaml.safe_load(open("configs/agent_conf/RAG_Agents/gpt-4o-mini/Structure_rag_gpt-4o-mini-mem0.yaml"))
dcfg = yaml.safe_load(open("configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml"))
cc = ConversationCreator(acfg, dcfg)
chunks, qas = cc.get_chunks()[0], cc.get_query_and_answers()[0]
ag = agent_mod.AgentWrapper(acfg, dcfg, load_agent_from="/tmp/claude_check_no_save")
ag.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=StubCompletions()))
initialization._memorize_context_chunks(ag, chunks, 0, 1)        # exactly as main.py memorizes
q = qas[0][0]
try:
    ag.send_message(q, memorizing=False, query_id=0, context_id=0)  # exactly as main.py queries
except Exception as e:  # downstream bookkeeping may fail on the stub; the search call is already recorded
    print("post-search exception (expected with stub):", type(e).__name__, e)
adds = sorted({c[1] for c in calls if c[0] == "add"})
searches = [c for c in calls if c[0] == "search"]
print("add() user_ids:", adds)
print("search() calls (user_id, #memories under that id):", searches)
print("KEY MISMATCH" if searches and all(s[2] == 0 for s in searches) else "keys match")
