#!/bin/bash
#SBATCH --job-name=hwcheck
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-%j.log    # relative to the folder sbatch is run from
# Add your cluster's GPU request here or on the command line, e.g.
#   sbatch --gres=gpu:1 run_hwcheck.sbatch
#   sbatch -p <partition> -w <gpu-node> run_hwcheck.sbatch

# Usage (from this folder): sbatch [slurm options] run_hwcheck.sbatch [hwcheck.py run options]
#   e.g. sbatch run_hwcheck.sbatch --model Qwen/Qwen3-8B --load-in-4bit
# Python: $HWCHECK_PYTHON, else the active venv, else ./.venv, else python3.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p results
if [ -n "$HWCHECK_PYTHON" ]; then PY="$HWCHECK_PYTHON"
elif [ -n "$VIRTUAL_ENV" ]; then PY="$VIRTUAL_ENV/bin/python"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
nvidia-smi --query-gpu=name,memory.used --format=csv
"$PY" -u hwcheck.py run "$@"
