# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest


@perftest()
def run_torch(input, weight, bias, eps, residual=None, x_bias=None):
    if x_bias is not None:
        input = input + x_bias
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    return output, residual_out


@perftest()
def run_ck(input, weight, bias, eps, residual=None, x_bias=None):
    if residual is None:
        residual_out = None
        output = aiter.layer_norm(input, weight, bias, eps, x_bias)
        # output = torch.empty_like(input)
        # aiter.layernorm2d_fwd(
        #     output,
        #     input,
        #     weight,
        #     bias,
        #     eps
        # )
    else:
        residual_out = torch.empty_like(input)
        output = torch.empty_like(input)
        aiter.layernorm2d_fwd_with_add(
            output, input, residual, residual_out, weight, bias, eps, x_bias
        )
    return output, residual_out


def test_layernorm2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    hidden_stats = torch.randn(m, n * 8, dtype=dtype, device="cuda")
    _q, k, _v = torch.split(hidden_stats, [6 * n, n, n], dim=1)
    input = k
    (a, *_), avg_a = run_torch(input, weight, bias, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, bias, 1e-5)
    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


def test_layernorm2d_fuseAdd(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    # x_bias = torch.randn(n, dtype=dtype, device="cuda")
    # x_bias = None
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    hidden_stats = torch.randn(m, n * 8, dtype=dtype, device="cuda")
    _q, _k, _v = torch.split(hidden_stats, [6 * n, n, n], dim=1)
    # input = k
    (a, res_a, *_), avg_a = run_torch(input, weight, bias, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, bias, 1e-5, residual=res)

    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, atol=0.03, msg=msg)
    checkAllclose(res_a, res_b, msg="res check")


parser = argparse.ArgumentParser(
    description="Test layernorm2d performance and correctness",
)
parser.add_argument(
    "--mode",
    choices=("plain", "add", "both"),
    default="both",
    help="LayerNorm variant to test.",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="+",
    default=[128],
    help="""Number of rows in the input tensor.
    e.g.: -m 1 32 128""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="+",
    default=[8192],
    help="""Number of columns in the input tensor.
    e.g.: -n 1024 4096 8192""",
)
args = parser.parse_args()
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            if args.mode in ("plain", "both"):
                test_layernorm2d(dtype, m, n)
            if args.mode in ("add", "both"):
                test_layernorm2d_fuseAdd(dtype, m, n)
