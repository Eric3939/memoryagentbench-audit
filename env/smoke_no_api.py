"""No-API smoke test: build MAB FactConsolidation chunk streams (native + one shuffled order).

Run from the MemoryAgentBench repo root:  python smoke_no_api.py <repo_root>
Checks: imports of the harness, HF data loading, chunking, and that a chunk permutation keeps every
serial-numbered fact (so FC gold labels stay valid under reordering).
"""
import os, random, re, sys
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1])
import yaml
import agent  # noqa: F401  (module-level imports: torch, transformers, tiktoken, openai, langchain)
from conversation_creator import ConversationCreator

for cfg in ["Factconsolidation_sh_6k.yaml", "Factconsolidation_sh_32k.yaml"]:
    dcfg = yaml.safe_load(open(f"configs/data_conf/Conflict_Resolution/{cfg}"))
    for name, acfg in [("long_context", {"agent_name": "Long_context_agent_gpt-4o-mini", "agent_chunk_size": None}),
                       ("mem0", {"agent_name": "Structure_rag_mem0", "agent_chunk_size": 4096})]:
        cc = ConversationCreator(acfg, dcfg)
        chunks = cc.get_chunks()[0] if hasattr(cc, "get_chunks") else None
        attrs = [a for a in vars(cc) if not a.startswith("_")]
        if chunks is None:
            for a in attrs:
                v = getattr(cc, a)
                if isinstance(v, list) and v and isinstance(v[0], list) and isinstance(v[0][0], str):
                    chunks = v[0]
                    break
        if chunks is None:
            print(cfg, name, "attrs:", attrs)
            continue
        serial = lambda cs: sorted(int(m) for c in cs for m in re.findall(r"(?m)^(\d+)\. ", c))
        perm = chunks[:]
        random.Random(1).shuffle(perm)
        print(f"{cfg} {name}: {len(chunks)} chunks; facts native={len(serial(chunks))} shuffled={len(serial(perm))} "
              f"identical_fact_set={serial(chunks) == serial(perm)}")
print("SMOKE OK")
