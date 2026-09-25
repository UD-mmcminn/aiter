# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from contextlib import nullcontext
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
from aiter.dist.parallel_state import set_custom_all_reduce
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_custom(
    tp_size,
    rankID,
    x,
    backend,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(backend != "rccl")
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        world_size=tp_size,
        rank=rankID,
        device_id=device,
    )
    cpu_group = None
    communicator = None
    try:
        ranks = list(range(tp_size))
        cpu_group = dist.new_group(ranks=ranks, backend="gloo")
        device_group = dist.group.WORLD
        communicator = CudaCommunicator(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name="custom_ar_test",
        )
        x = x.to(device)

        # Warm up the device process group and align all ranks before timing.
        dist.all_reduce(torch.zeros(1, device=device), group=device_group)
        torch.cuda.synchronize()

        quick = communicator.qr_comm
        if quick is not None and not quick.disabled:
            raise RuntimeError(
                "quick all-reduce would mask the requested backend; set "
                "AITER_QUICK_REDUCE_QUANTIZATION=NONE"
            )
        if backend == "custom":
            if communicator.ca_comm is None or communicator.ca_comm.disabled:
                raise RuntimeError(
                    "custom all-reduce was requested but its communicator is disabled"
                )
            if not communicator.ca_comm.should_custom_ar(x):
                raise RuntimeError(
                    "custom all-reduce was requested but rejected the input "
                    f"shape={tuple(x.shape)}, dtype={x.dtype}"
                )
            selected_backend = "custom"
        elif backend == "rccl":
            if communicator.ca_comm is not None:
                raise RuntimeError("custom all-reduce remained enabled for the RCCL run")
            if communicator.pynccl_comm is None or communicator.pynccl_comm.disabled:
                raise RuntimeError(
                    "RCCL was requested but its communicator is disabled"
                )
            selected_backend = "rccl"
        elif backend == "auto":
            ca_comm = communicator.ca_comm
            if (
                ca_comm is not None
                and not ca_comm.disabled
                and ca_comm.should_custom_ar(x)
            ):
                selected_backend = "custom"
            else:
                if (
                    communicator.pynccl_comm is None
                    or communicator.pynccl_comm.disabled
                ):
                    raise RuntimeError(
                        "automatic routing selected RCCL but its communicator "
                        "is disabled"
                    )
                selected_backend = "rccl"
        else:
            raise ValueError(f"unsupported all-reduce backend: {backend}")

        if withGraph:
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            capture_context = (
                communicator.ca_comm.capture()
                if selected_backend == "custom"
                else nullcontext()
            )
            with torch.cuda.stream(capture_stream), capture_context:
                with torch.cuda.graph(graph, stream=capture_stream):
                    out = communicator.all_reduce(x)
                    if out is None:
                        raise RuntimeError(f"{backend} rejected the requested input")
            torch.cuda.current_stream().wait_stream(capture_stream)
            out.fill_(0)

            @perftest(use_cuda_event=True)
            def run_ca():
                graph.replay()

            _, us = run_ca()
        else:

            @perftest(use_cuda_event=True)
            def run_ca(x):
                out = communicator.all_reduce(x)
                if out is None:
                    raise RuntimeError(f"{backend} rejected the requested input")
                return out

            out, us = run_ca(x)

        # Multiprocessing return values must not retain device storage after
        # this rank tears its process groups down.
        out_cpu = out.cpu()
        return out_cpu, us, selected_backend
    finally:
        if communicator is not None:
            communicator.destroy()
        if cpu_group is not None:
            dist.destroy_process_group(cpu_group)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


@benchmark()
def test_allreduce_custom(
    tp_size,
    shape,
    dtype,
    backend="custom",
    withGraph=False,
    distributed_init_method: str | None = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    generator = torch.Generator().manual_seed(0)
    ref = torch.zeros(shape, dtype=torch.float32)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype, generator=generator)
        ref += x.float()
        rets.append(
            pool.apply_async(
                allreduce_custom,
                args=(
                    tp_size,
                    i,
                    x,
                    backend,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [us for _, us, _ in rets]
    selected_backends = {selected for _, _, selected in rets}
    if len(selected_backends) != 1:
        raise AssertionError(
            f"ranks selected different all-reduce backends: {selected_backends}"
        )
    selected_backend = selected_backends.pop()
    rank0_out = rets[0][0]
    for rank, (out, _, _) in enumerate(rets[1:], start=1):
        if not torch.equal(rank0_out, out):
            max_rank_delta = (rank0_out.float() - out.float()).abs().max().item()
            raise AssertionError(
                f"{backend} all-reduce output differs between rank 0 and rank "
                f"{rank}: max_abs={max_rank_delta}"
            )

    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 3e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    max_err = 0.0
    for out, us, _ in rets:
        msg = (
            f"test_allreduce_custom: {shape=} {dtype=} {backend=} "
            f"{withGraph=} {us:>8.2f}"
        )
        err = checkAllclose(
            ref,
            out.float(),
            rtol=rtol,
            atol=atol,
            tol_err_ratio=0,
            msg=msg,
        )
        if err:
            raise AssertionError(
                f"{backend} all-reduce failed its FP32-reference check: "
                f"mismatch_ratio={err}"
            )
        max_err = max(max_err, err)
    return {
        "selected_backend": selected_backend,
        "grid_cap": os.environ.get("AITER_CUSTOM_AR_GRID_CAP", "auto"),
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


# Custom-AR size cutoff (bytes). Mirrors _DEFAULT_CAR_MAX_SIZE in
# aiter/dist/device_communicators/custom_all_reduce.py and honors the same
# AITER_CUSTOM_AR_MAX_SIZE override. Inputs at or below this run on the custom
# kernels (larger ones fall back to RCCL), so the swept size list stops here.
_DEFAULT_CAR_MAX_BYTES = 8192 * 8192
_GFX908_TP4_CAR_MAX_BYTES = 6 * 1024 * 1024


def _car_max_bytes(tp_size: int) -> int:
    e = os.environ.get("AITER_CUSTOM_AR_MAX_SIZE", "")
    try:
        v = int(e)
        if v > 0:
            return v
    except ValueError:
        pass
    try:
        props = torch.cuda.get_device_properties(0)
        if tp_size == 4 and "gfx908" in getattr(props, "gcnArchName", ""):
            return _GFX908_TP4_CAR_MAX_BYTES
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_CAR_MAX_BYTES


def gen_sizes(dtype, tp_size: int) -> list[int]:
    """Element counts to sweep: [1024, 2048, 4096] then 7168*k / 8192*k for
    k = 1, 2, 4, 8, ... up to the largest size custom_all_reduce serves."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    max_numel = _car_max_bytes(tp_size) // itemsize
    sizes = [n for n in (1024, 2048, 4096) if n <= max_numel]
    k = 1
    while True:
        added = False
        for base in (7168, 8192):
            n = base * k
            if n <= max_numel:
                sizes.append(n)
                added = True
        if not added:
            break
        k *= 2
    return sizes


l_dtype = ["bf16", "fp16", "fp32"]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    default="bf16",
    help="data type (default: bf16)",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4, 6, 8],
    default=8,
    help="number of GPUs / tensor-parallel size (default: 8)",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["graph", "eager"],
    default="graph",
    help="execution mode (default: graph)",
)
parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=["auto", "custom", "rccl", "both"],
    default="custom",
    help="all-reduce implementation to benchmark (default: custom)",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="single shape override, e.g. -s 128,8192 (default: swept size list)",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    with_graph = args.mode == "graph"
    if args.shape is not None:
        l_shape = [args.shape]
    else:
        l_shape = gen_sizes(dtype, args.tp_size)
    df = []
    for shape in l_shape:
        backends = ["custom", "rccl"] if args.backend == "both" else [args.backend]
        for backend in backends:
            ret = test_allreduce_custom(
                args.tp_size,
                shape,
                dtype,
                backend=backend,
                withGraph=with_graph,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
            df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "backend",
        "selected_backend",
        "grid_cap",
        "withGraph",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "custom allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
    if args.backend == "both":
        comparison = df.pivot_table(
            index=["tp_size", "shape", "dtype", "withGraph"],
            columns="backend",
            values="max_us",
            aggfunc="first",
        ).reset_index()
        comparison["custom speedup vs RCCL"] = (
            comparison["rccl"] / comparison["custom"]
        )
        logger.info(
            "custom vs RCCL summary (markdown):\n%s",
            comparison.to_markdown(index=False),
        )
