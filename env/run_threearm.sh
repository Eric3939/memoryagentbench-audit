#!/bin/bash
#SBATCH -J mab-threearm
#SBATCH -o logs/threearm-%j.out
# Three-arm Mem0 check (public harness / fixed keying / closed-book) on 4 MAB splits, one context each.
# Needs an OpenAI API key. Submit with:
#   OPENAI_API_KEY=... sbatch -p short -c 2 --mem=8G -t 06:00:00 env/run_threearm.sh
# Mem0 uses a shared /tmp/qdrant path, so arms run sequentially in one job. Estimated cost ~$5-10 (gpt-4o-mini).
set -euo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY before submitting}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 MEM0_TELEMETRY=False
P="$(cd "$(dirname "$0")/.." && pwd)"; R=$P/external/MemoryAgentBench; C=$R/configs; PY=$P/external/mabenv/bin/python
export TMPDIR=$P/external/tmp
M=$C/agent_conf/RAG_Agents/gpt-4o-mini/Structure_rag_gpt-4o-mini-mem0.yaml
for D in Conflict_Resolution/Factconsolidation_sh_6k Accurate_Retrieval/LongMemEval/Longmemeval_s_star \
         Accurate_Retrieval/EventQA/Eventqa_64k Test_Time_Learning/ICL/ICL_banking77; do
  for ARM in --buggy-keying "" --closed-book; do
    echo "=== $D arm=${ARM:-fixed} $(date)"
    $PY -I $P/scripts/run_grid.py --repo $R --agent_config $M --dataset_config $C/data_conf/$D.yaml \
        --out $P/results/threearm --contexts 0 $ARM
  done
done
python3 -I $P/scripts/threearm_report.py $P/results/threearm | tee $P/reports/data/threearm_report.md
