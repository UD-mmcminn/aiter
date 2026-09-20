# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness coverage for the public CK INT8 A8W8 GEMM path."""

import argparse

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime


DEFAULT_CASES = (
    # M, N, K, bias
    (1, 64, 32, False),
    (17, 65, 80, False),
    (17, 65, 80, True),
    (256, 256, 256, False),
    (32, 1024, 1024, True),
)

OUTPUT_DTYPES = {"bf16": dtypes.bf16, "fp16": dtypes.fp16}


def run_case(
    m: int,
    n: int,
    k: int,
    use_bias: bool,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> float:
    x = torch.randn((m, k), dtype=dtype, device="cuda") * 0.25
    weight = torch.randn((n, k), dtype=dtype, device="cuda") * 0.25
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.i8)
    wq, w_scale = aiter.pertoken_quant(weight, quant_dtype=dtypes.i8)
    bias = (
        torch.randn((1, n), dtype=dtype, device="cuda") * 0.25
        if use_bias
        else None
    )

    # Match the CK epilogue: integer GEMM accumulation, rowwise FP32 scales,
    # optional bias, then a single conversion to the output dtype. FP32
    # accumulates these test cases exactly: 127**2 * max(K) is below 2**24.
    expected = F.linear(xq.float(), wq.float()) * w_scale.T * x_scale
    if bias is not None:
        expected = expected + bias.float()
    expected = expected.to(dtype)

    # gfx9 public dispatch is the CK path. Testing the public entry point also
    # catches a future routing regression that a direct gemm_a8w8_CK call would
    # miss.
    actual = aiter.gemm_a8w8(
        xq,
        wq,
        x_scale,
        w_scale,
        bias=bias,
        dtype=dtype,
    )
    torch.cuda.synchronize()

    assert actual.shape == (m, n)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual.float() - expected.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the public INT8 A8W8 CK GEMM path"
    )
    parser.add_argument(
        "-d", "--dtype", nargs="+", choices=OUTPUT_DTYPES, default=["bf16"]
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    args = parser.parse_args()

    gfx = get_gfx_runtime()
    if not gfx.startswith("gfx9"):
        parser.error(f"the public CK A8W8 route requires gfx9, got {gfx}")

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} ({gfx})")
    passed = 0
    failures = []

    for dtype_name in args.dtype:
        dtype = OUTPUT_DTYPES[dtype_name]
        atol = (
            args.atol
            if args.atol is not None
            else (0.02 if dtype == dtypes.bf16 else 0.002)
        )
        rtol = (
            args.rtol
            if args.rtol is not None
            else (0.01 if dtype == dtypes.bf16 else 0.002)
        )
        for m, n, k, use_bias in DEFAULT_CASES:
            case = f"dtype={dtype_name} M={m} N={n} K={k} bias={use_bias}"
            try:
                max_abs = run_case(m, n, k, use_bias, dtype, atol, rtol)
            except Exception as error:
                failures.append((case, error))
                print(f"FAIL {case}: {error}")
            else:
                passed += 1
                print(f"PASS {case} max_abs={max_abs:.6g}")

    print(f"CK INT8 A8W8 GEMM summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"CK INT8 A8W8 GEMM correctness failures: {len(failures)}")


if __name__ == "__main__":
    main()
