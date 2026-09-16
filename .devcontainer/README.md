# AITER on MI100 with Coder / Kubernetes / Envbuilder

This environment runs as **root**, uses `/workspaces` for persistent storage,
sets `CODEX_HOME=/workspaces/.codex`, and includes ROCm LLVM/HIP, PyTorch and
Codex CLI. Its ROCm 7.2.4 / PyTorch 2.10.0 / Python 3.12 base is digest-pinned.
Existing AITER ASM kernels still need the gfx908 port; installing the environment
is not a claim of kernel support.

## Coder template

[The revised Terraform template](coder-template/main.tf) is based on the supplied
organization template. [Its deployment notes](coder-template/README.md) explain
the changes and the choices to make when creating a workspace.

Select **MI100 (reserve one AMD GPU)** and enter the MI100 node's
`kubernetes.io/hostname` label. General workspaces can still use the same template
without a GPU reservation. Selection is the deployer's responsibility; the
Terraform template does not discover or validate GPU models. The node must have
the AMD device plugin advertising `amd.com/gpu`.

The existing PVC stays mounted at `/workspaces`; no data migration or additional
Codex volume is needed. Envbuilder's default repo layout puts this checkout at
`/workspaces/aiter` (unless its repository basename differs). The agent and
editors open `/workspaces` so this template also works with other repositories.

```text
/workspaces/                <- per-workspace PVC
  .codex/                   <- Codex configuration, auth and session state
  aiter/                    <- this Git checkout
    .codex/                 <- separate project-specific configuration
```

Root is used for building, lifecycle commands and development sessions. There
is no UID 10001 or fsGroup requirement. The NFS export still needs to permit
writes: server-side root squashing can restrict container root too.

For another template, [coder-pod.example.yaml](coder-pod.example.yaml) shows the
required mount/device configuration. Envbuilder does not implement devcontainer
`mounts` or `workspaceFolder`; configure these through Kubernetes and Envbuilder
options. Preserve `/workspaces` in any custom ignored-path configuration.

## Bootstrap and verification

The post-create hook installs the optional organization CA supplied through
`CUSTOM_CA_CERT_B64`, checks tooling/storage, initializes submodules, installs
`requirements.txt`, then runs:

```bash
python -m pip install --no-build-isolation -e .
```

Build dependencies are provided by the image and requirements file. Build hooks
can see the base image's Torch/Triton, and bulk AITER prebuilding is disabled.
The writable development venv is rebuilt with the image instead of being
persisted across Python/ROCm upgrades.

In the workspace, run:

```bash
cd /workspaces/aiter
python .devcontainer/doctor.py
codex login --device-auth
```

The doctor checks the PVC mount, writable Codex state, gfx908 assembly, GPU
visibility, and FP32/FP16/BF16 matrix multiplication. It does not import AITER or
launch unported kernels. Hardware/driver compatibility needs validation on the
actual MI100 node.

If AITER installation fails while working on the port, set `AITER_DEV_INSTALL`
to `0` in devcontainer `containerEnv` and rebuild to use the assembly tools and
Codex. Remove the override and rerun `bash .devcontainer/bootstrap.sh` when ready.

Codex state survives rebuilds and stop/start while the same PVC is retained.
No local credentials are copied into the image or repository. Existing Codex
history on another machine is not migrated automatically.

## Shared memory and image settings

The Terraform template mounts an 8 GiB RAM-backed `emptyDir` at `/dev/shm`.
This is an upper bound, not memory reserved at startup. Actual usage counts
against the container's memory limit. It is temporary and should not store
Codex state or source code.

Change `build.args.ROCM_IMAGE` to use another validated ROCm/PyTorch base;
`CODEX_VERSION` is also a build arg. OS/Python dependency installation is not a
fully reproducible lockfile build. Envbuilder's VS Code customization support
is partial; use Coder's editor configuration if the listed extensions do not
install automatically. Codex CLI is independently installed in the image.

## References

- [Envbuilder supported properties](https://github.com/coder/envbuilder/blob/main/docs/devcontainer-spec-support.md)
- [Envbuilder variables](https://github.com/coder/envbuilder/blob/main/docs/env-variables.md)
- [AMD GPU allocation](https://instinct.docs.amd.com/projects/k8s-device-plugin/en/latest/user-guide/configuration.html)
- [Kubernetes emptyDir](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir)
- [Codex state location](https://developers.openai.com/codex/config-advanced/)
