#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# /workspaces must already be a writable PVC mount.
bash .devcontainer/configure-trust.sh
install -d -m 700 "${CODEX_HOME:-/workspaces/.codex}"
python .devcontainer/doctor.py --static

# The assembly tools and Codex remain usable while troubleshooting an AITER
# dependency/build issue: set AITER_DEV_INSTALL=0 in containerEnv and rebuild.
if [[ "${AITER_DEV_INSTALL:-1}" == "0" ]]; then
    echo "AITER installation skipped (AITER_DEV_INSTALL=0)."
    exit 0
fi

git submodule update --init --recursive
python -m pip install -r requirements.txt
# Keep build hooks in this environment so they can see bundled Torch/Triton.
# Build requirements come from the image and requirements.txt above.
AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 python -m pip install --no-build-isolation -e .

echo "AITER editable installation finished."
echo "Run: python .devcontainer/doctor.py"
echo "Then sign into Codex with: codex login --device-auth"
