"""Check the porting environment without importing AITER or launching its kernels."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="Skip GPU execution")
    args = parser.parse_args()
    failures = []

    def check(ok, message):
        print(f"{'OK' if ok else 'FAIL'}: {message}", flush=True)
        if not ok:
            failures.append(message)

    for tool in ("hipcc", "llvm-mc", "llvm-objdump", "llvm-readobj", "ld.lld", "codex"):
        check(shutil.which(tool) is not None, f"{tool} is on PATH")

    codex_dir = Path(os.environ.get("CODEX_HOME", "/workspaces/.codex"))
    check(str(codex_dir) == "/workspaces/.codex", "CODEX_HOME=/workspaces/.codex")
    try:
        with tempfile.TemporaryFile(dir=codex_dir):
            pass
        check(True, "Codex state directory is writable")
    except OSError as exc:
        check(False, f"Codex state directory: {exc}")
    # A mount is necessary but not sufficient to prove durability; the template
    # must use a PVC rather than emptyDir or a container filesystem directory.
    check(os.path.ismount("/workspaces"), "/workspaces is a mount (confirm it is a PVC in Coder)")

    if shutil.which("llvm-mc"):
        probe = subprocess.run(
            ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx908",
             "-filetype=obj", "-o", os.devnull],
            input="s_endpgm\n", text=True, capture_output=True,
        )
        check(probe.returncode == 0, "LLVM can assemble gfx908")
        if probe.returncode:
            print(probe.stderr)

    try:
        import torch

        check(bool(torch.version.hip), f"PyTorch {torch.__version__}, HIP {torch.version.hip}")
        if not args.static:
            check(Path("/dev/kfd").exists(), "/dev/kfd is exposed")
            check(torch.cuda.is_available(), "PyTorch sees an accessible AMD GPU")
            matched = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                arch = props.gcnArchName.split(":")[0]
                print(f"GPU {i}: {props.name}, {props.gcnArchName}", flush=True)
                check(arch == "gfx908", f"GPU {i} is gfx908")
                if arch != "gfx908":
                    continue
                matched += 1
                # Exercise the actual bundled libraries, not just detection.
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    x = torch.ones((64, 64), device=f"cuda:{i}", dtype=dtype)
                    y = x @ x
                    torch.cuda.synchronize(i)
                    check(bool((y == 64).all().item()), f"GPU {i} {dtype} GEMM")
            check(matched > 0, "At least one MI100 was tested")
    except Exception as exc:
        check(False, f"PyTorch check: {exc}")

    print("AITER gfx908 kernel support is a separate porting task.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
