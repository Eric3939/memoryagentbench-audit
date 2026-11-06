#!/bin/bash
#SBATCH -J aes-paper
#SBATCH -o logs/aes-paper-%j.out
# Compile the paper with tectonic (needs a recent TeX snapshot; Overleaf XeLaTeX also works).
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
T=$P/external/texenv
if [ ! -d "$T" ]; then conda create -y -q -p "$T" -c conda-forge tectonic; fi
cd $P/paper
$T/bin/tectonic -X compile main.tex --keep-logs 2>&1 | tail -40
ls -la main.pdf
