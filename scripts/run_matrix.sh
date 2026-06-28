#!/usr/bin/env bash
# Resumable overnight matrix runner (run INSIDE WSL/Ubuntu).
#
# Cache-first: every (model x method x bit x dataset) run is keyed by run_id;
# re-running skips finished runs, so this is safe to stop/restart. One failing
# run is logged but does NOT abort the batch (no `set -e`).
#
# Launch detached so it survives your shell/my session closing:
#   cd ~/abstention-quant
#   nohup bash scripts/run_matrix.sh > results/logs/nohup.out 2>&1 &
#   tail -f results/logs/nohup.out      # watch progress
#
# Override scope via env vars, e.g.:  N=150 DATASETS="abstention_bench" bash scripts/run_matrix.sh
set -uo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

# bitsandbytes needs torch's bundled CUDA libs (libnvJitLink.so.13) on the path.
export LD_LIBRARY_PATH="$(python -c 'import glob,site;print(":".join(glob.glob(site.getsitepackages()[0]+"/nvidia/*/lib")))'):${LD_LIBRARY_PATH:-}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODELS="${MODELS:-qwen25_15b phi3_mini qwen25_3b}"
DATASETS="${DATASETS:-abstention_bench squad2}"
METHODS="${METHODS:-bnb gguf}"   # "bnb", "gguf", or "bnb gguf"
N="${N:-120}"
MNT="${MNT:-64}"
SIGNAL="${SIGNAL:-token_prob}"
BNB_BITS="${BNB_BITS:-16,8,4}"

_want() { case " $METHODS " in *" $1 "*) return 0;; *) return 1;; esac; }

# Per-model GGUF repo + 'bits:file' list (all open, full q8/q4/q3/q2 range).
GGUF_REPO_qwen25_15b="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILES_qwen25_15b="8:qwen2.5-1.5b-instruct-q8_0.gguf,4:qwen2.5-1.5b-instruct-q4_k_m.gguf,3:qwen2.5-1.5b-instruct-q3_k_m.gguf,2:qwen2.5-1.5b-instruct-q2_k.gguf"
GGUF_REPO_qwen25_3b="Qwen/Qwen2.5-3B-Instruct-GGUF"
GGUF_FILES_qwen25_3b="8:qwen2.5-3b-instruct-q8_0.gguf,4:qwen2.5-3b-instruct-q4_k_m.gguf,3:qwen2.5-3b-instruct-q3_k_m.gguf,2:qwen2.5-3b-instruct-q2_k.gguf"
GGUF_REPO_phi3_mini="bartowski/Phi-3.1-mini-4k-instruct-GGUF"
GGUF_FILES_phi3_mini="8:Phi-3.1-mini-4k-instruct-Q8_0.gguf,4:Phi-3.1-mini-4k-instruct-Q4_K_M.gguf,3:Phi-3.1-mini-4k-instruct-Q3_K_M.gguf,2:Phi-3.1-mini-4k-instruct-Q2_K.gguf"

mkdir -p results/logs
LOG="results/logs/matrix.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "START matrix: models=[$MODELS] datasets=[$DATASETS] N=$N MNT=$MNT"
for model in $MODELS; do
  for ds in $DATASETS; do
    split=test
    [ "$ds" = "squad2" ] && split=validation

    if _want bnb; then
      say "RUN bnb  $model x $ds"
      python -m aq.cli slice --model "$model" --dataset "$ds" --bits "$BNB_BITS" \
        --split "$split" --max-samples "$N" --max-new-tokens "$MNT" --signal "$SIGNAL" \
        2>&1 | tee -a "$LOG" || say "FAILED bnb $model x $ds (continuing)"
    fi

    repo_var="GGUF_REPO_${model}"; files_var="GGUF_FILES_${model}"
    if _want gguf && [ -n "${!repo_var:-}" ]; then
      say "RUN gguf $model x $ds"
      python -m aq.cli gguf --model "$model" --dataset "$ds" \
        --gguf-repo "${!repo_var}" --files "${!files_var}" \
        --split "$split" --max-samples "$N" --max-new-tokens "$MNT" --signal "$SIGNAL" \
        2>&1 | tee -a "$LOG" || say "FAILED gguf $model x $ds (continuing)"
    fi
  done
done
say "ALL DONE. Summaries in results/*.json, figures in results/figures/."
