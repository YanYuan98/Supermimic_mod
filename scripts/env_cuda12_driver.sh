# Source on machines whose NVIDIA driver is older than 580 (CUDA 12.x), e.g. driver 570 / CUDA 12.8:
#   source scripts/env_cuda12_driver.sh
# One-time venv setup on such a machine (the lockfile's torch 2.14.1+cu130 and warp-lang 1.18.0 need driver >= 580;
# torch 2.14 has no cu128/cu129 wheels and its cu126 wheels lack sm_120):
#   uv sync --group dev
#   uv pip install --python .venv/bin/python "torch==2.13.0+cu129" --index-url https://download.pytorch.org/whl/cu129 \
#       --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
#   uv pip install --python .venv/bin/python --no-deps \
#       https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_NO_SYNC=1  # keep `uv run` from re-syncing the venv back to the cu130 lockfile
unset PYTHONPATH
# Triton compiles sm_100+ with its bundled CUDA 13 ptxas-blackwell, whose cubins a 12.x driver rejects
# ("device kernel image is invalid"); the bundled CUDA 12.8 ptxas supports sm_120.
export TRITON_PTXAS_BLACKWELL_PATH="$_repo/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/sm_inductor_cu129"
unset _repo
# Data-parallel runs pin one GPU per rank (scripts/train.py); NCCL's cuMem-based SHM transport then needs
# peer access between the GPUs, which these RTX 5090s do not support ("peer access is not supported").
export NCCL_CUMEM_ENABLE=0
