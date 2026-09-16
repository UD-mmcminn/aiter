# Build context is only .devcontainer: repo contents and credentials are not
# copied into the image. Envbuilder clones the repo onto the workspace PVC.
ARG ROCM_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
FROM ${ROCM_IMAGE}

USER root
ARG CODEX_VERSION=0.154.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl gdb git git-lfs jq \
        less nodejs npm openssh-client openssl ripgrep shellcheck sudo tmux vim \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# Development sessions run as root. Codex state lives on the workspace PVC.
RUN mkdir -p /workspaces

# Keep the ROCm image's PyTorch and Triton. Editable AITER/development packages
# go into a writable venv that is rebuilt with the image, not persisted across
# changes to Python or ROCm versions.
RUN python3 -c 'import site; print("\n".join(site.getsitepackages()))' > /tmp/rocm-python-sites \
    && python3 -m venv --system-site-packages /opt/aiter-venv \
    && cp /tmp/rocm-python-sites "$(/opt/aiter-venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/rocm-base.pth" \
    && rm /tmp/rocm-python-sites \
    && /opt/aiter-venv/bin/python -m pip install --no-cache-dir \
        'setuptools>=77,<81' 'setuptools_scm>=8,<11' vcs_versioning \
        packaging wheel 'msgpack==1.1.1' 'pyelftools==0.32'

ENV PATH=/opt/aiter-venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    CODEX_HOME=/workspaces/.codex \
    GPU_ARCHS=gfx908 \
    PYTORCH_ROCM_ARCH=gfx908 \
    AITER_USE_SYSTEM_TRITON=1 \
    PREBUILD_KERNELS=0 \
    MAX_JOBS=8

# These checks need no GPU and fail image creation if essential tools are absent.
RUN command -v hipcc && command -v llvm-objdump && command -v llvm-mc \
    && command -v ld.lld && command -v rocminfo && codex --version \
    && python -c 'import torch; assert torch.version.hip, "ROCm PyTorch required"'

WORKDIR /workspaces
USER root
CMD ["bash"]
