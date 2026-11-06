# What Does a MemoryAgentBench Score Measure?

Code, data and paper source for a harness and claims audit of
[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (MAB), an agent-memory
leaderboard. Prepared for the Agent Evaluation Science (AES) symposium, Fall 2026.

The audit asks what a MAB score can support, in two parts:

- **Harness audit.** An offline review of the public evaluation harness. The main finding is
  that Mem0 stores memories under one key and queries another, so it answers without its own
  memory. Related isolation problems affect Cognee and Zep, and RAG agents with small input
  limits silently drop the oldest history. See `reports/harness_audit.md` and
  `reports/other_harness_audit.md`.
- **Claims audit.** 268 quantitative comparisons extracted from 41 papers, pull requests and
  issues that report MAB numbers, checked against the item-sampling noise of each question set.
  See `papers/search/mab_claims.csv` and `reports/data/claims_audit.json`.

## Layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source (AES template), generated tables and figures |
| `scripts/` | Analysis scripts (see below) |
| `env/` | Environment setup and offline harness checks |
| `reports/` | Audit write-ups and the data they produce (`reports/data/`) |
| `papers/search/` | Extracted claims, hand-verification sample, reproductions, dataset sizes |

## Reproducing the analyses

The zero-cost analyses need only Python 3 with `numpy`, `scipy`, `pandas` and `matplotlib`.
Run from the repo root:

```
python scripts/claims_audit.py papers/search/mab_claims.csv reports/data/claims_audit.json
python scripts/noise_audit.py reports/data/mab_table3.csv reports/data/noise_audit.json
python scripts/resolution_limits.py          # -> reports/data/resolution_limits.json, paper/resolution_table.tex
python scripts/make_harness_table.py         # -> paper/harness_table.tex
python scripts/make_figs.py .                # -> paper/figs/*.pdf
```

The offline harness checks drive the real MAB code on cached Hugging Face data without any
model calls. Clone MAB into `external/MemoryAgentBench`, create the environment with
`env/setup_and_smoke.sh`, then:

```
HF_HUB_DISABLE_PROGRESS_BARS=1 external/mabenv/bin/python -I env/check_harness_audit.py external/MemoryAgentBench
external/mabenv/bin/python -I env/check_mem0_keying.py external/MemoryAgentBench
external/mabenv/bin/python -I env/check_mem0_real_offline.py external/MemoryAgentBench
```

Their recorded outputs are in `reports/data/harness_audit_checks.{json,log}` and
`reports/data/mem0_real_offline_check.log`.

`scripts/run_grid.py`, `scripts/analyze_grid.py`, `scripts/gtheory.py` and
`scripts/threearm_report.py` implement the paid experiments (memory rebuild and ingestion-order
variance, and the three-arm Mem0 check in `env/run_threearm.sh`). They require an OpenAI API key.
`scripts/version_diff.py` needs the arXiv PDFs of the MAB paper versions, which are not included.

## Paper

`paper/main.tex` compiles with XeLaTeX (for example on Overleaf) using the AES template in the
same folder. Placeholders marked `\tbd{}` are results that depend on the paid experiments.

## License

Code and data in this repository are released under the MIT License (see `LICENSE`).
The AES LaTeX template in `paper/` carries its own license; see `paper/LICENSE` and
`paper/THIRD-PARTY-NOTICES.md`.
