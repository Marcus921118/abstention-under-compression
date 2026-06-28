#!/usr/bin/env bash
# One-shot WSL2/Ubuntu environment build for the GPTQ/AWQ quantization track.
# Run INSIDE Ubuntu (WSL2) after `wsl --install`. See docs/wsl-setup.md.
#
# Why this is needed: GPTQ/AWQ kernel backends have no Windows wheels and need a
# CUDA compiler the Windows host lacks. On Linux they have wheels. BUT Ubuntu
# 26.04 ships Python 3.14, for which those backends ALSO lack wheels -- so we use
# `uv` to provision Python 3.12 and pin a known-good torch/transformers combo.
# Designed to be sudo-free (uv is user-level; repo is git-cloned).
set -euo pipefail

REPO_WIN="/mnt/c/RD_work/abstention-quant"
REPO_DST="$HOME/abstention-quant"
PYVER="3.12"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # adjust to match driver

echo "== 0. prerequisites (git/curl) =="
for t in git curl; do
  command -v "$t" >/dev/null || { echo "MISSING $t -> run: sudo apt-get update && sudo apt-get install -y $t"; exit 1; }
done

echo "== 1. install uv (user-level, no sudo) =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090,SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== 2. provision Python $PYVER =="
uv python install "$PYVER"

echo "== 3. clone repo into native Linux FS (committed code only) =="
if [ ! -d "$REPO_DST/.git" ]; then
  git clone "$REPO_WIN" "$REPO_DST"
else
  git -C "$REPO_DST" pull --ff-only || true
fi
cd "$REPO_DST"

echo "== 4. venv on Python $PYVER =="
uv venv --python "$PYVER" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 5. torch (CUDA) — known-good for GPTQ/AWQ wheels =="
uv pip install "torch==2.6.*" --index-url "$TORCH_INDEX"

echo "== 6. HF stack + bnb (transformers 4.x targets GPTQ/AWQ integrations) =="
uv pip install "transformers==4.46.*" accelerate datasets bitsandbytes sentencepiece

echo "== 7. GPTQ + AWQ backends (Linux wheels) =="
uv pip install gptqmodel optimum || echo "WARN: gptqmodel failed — report the error to pin versions"
uv pip install autoawq || echo "WARN: autoawq failed — try a pinned autoawq / autoawq-kernels"

echo "== 8. install the aq package =="
uv pip install -e ".[dev]"

echo "== 9. reuse Windows-extracted AbstentionBench data =="
mkdir -p "$REPO_DST/data/processed"
if [ -f "$REPO_WIN/data/processed/abstention_bench.jsonl" ]; then
  cp "$REPO_WIN/data/processed/abstention_bench.jsonl" "$REPO_DST/data/processed/"
  echo "  reused abstention_bench.jsonl"
fi

echo "== 10. sanity checks =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m aq.cli selftest
pytest -q || true

cat <<'EOF'

== DONE ==
Env ready at ~/abstention-quant/.venv (Python 3.12). Next: wire GPTQ/AWQ into
src/aq/models/quantize.py, then:

  cd ~/abstention-quant && source .venv/bin/activate
  aq slice --model qwen25_15b --dataset abstention_bench --bits 16,8,4 --max-samples 200

If gptqmodel/autoawq failed to install, paste the exact error and we'll pin a
compatible (torch, transformers, backend) triple.
EOF
