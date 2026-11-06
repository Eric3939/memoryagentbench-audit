"""Per-source stats for the MemoryAgentBench HF dataset (contexts, questions, context length)."""
import sys, glob, os, json
import pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.parquet"))):
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        md = r["metadata"]; src = md.get("source", "?") if isinstance(md, dict) else "?"
        rows.append(dict(split=os.path.basename(f)[:-8], source=src, n_q=len(r["questions"]),
                         ctx_chars=len(r["context"])))
d = pd.DataFrame(rows)
g = d.groupby(["split", "source"]).agg(n_ctx=("n_q", "size"), n_q=("n_q", "sum"),
        q_min=("n_q", "min"), q_max=("n_q", "max"), ctx_kchars_mean=("ctx_chars", lambda x: round(x.mean()/1000)))
pd.set_option("display.width", 200); print(g.to_string())
g.reset_index().to_csv(sys.argv[2], index=False)
