#!/bin/bash
#SBATCH -J mab-env
#SBATCH -o logs/mab-env-%j.out
# Create the MemoryAgentBench environment and run the no-API smoke test.
# Expects the MemoryAgentBench repo cloned at external/MemoryAgentBench.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$(cd "$(dirname "$0")/.." && pwd)"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
ENV=$P/external/mabenv
if [ ! -d "$ENV" ]; then conda create -y -q -p "$ENV" python=3.10; fi
source activate "$ENV" 2>/dev/null || conda activate "$ENV"
pip install -q -r $P/env/requirements-min.txt
pip freeze > $P/env/requirements-min.lock.txt
cd $P/external/MemoryAgentBench
python -I $P/env/smoke_no_api.py $P/external/MemoryAgentBench
