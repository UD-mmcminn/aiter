# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for AITER's generated binary operators.

The legacy filename is retained because CI and test-splitting metadata reference it.
"""

import argparse
from dataclasses import dataclass

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import aiter_operator as operator_impl


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

PUBLIC_OUT = {
    "add": aiter.add,
    "sub": aiter.sub,
    "mul": aiter.mul,
    "div": aiter.div,
}

PUBLIC_INPLACE = {
    "add": aiter.add_,
    "sub": aiter.sub_,
    "mul": aiter.mul_,
    "div": aiter.div_,
}

PRIVATE_OUT = {
    "add": operator_impl._add_kernel,
    "sub": operator_impl._sub_kernel,
    "mul": operator_impl._mul_kernel,
    "div": operator_impl._div_kernel,
}

PRIVATE_INPLACE = {
    "add": operator_impl._add_kernel_,
    "sub": operator_impl._sub_kernel_,
    "mul": operator_impl._mul_kernel_,
    "div": operator_impl._div_kernel_,
}

TORCH_OUT = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
}


@dataclass(frozen=True)
class Case:
    name: str
    input_shape: tuple[int, ...]
    other_shape: tuple[int, ...]
    native: bool
    reverse: bool = False
    transpose_input: bool = False
    transpose_other: bool = False


OUT_CASES = (
    Case("contiguous", (32, 128), (32, 128), True),
    Case("contiguous_tail", (17, 65), (17, 65), True),
    Case("vector_aligned", (256,), (256,), True),
    Case("broadcast_dim0", (4, 16, 64), (1, 16, 64), True),
    Case("broadcast_dim1", (4, 16, 64), (4, 1, 64), True),
    Case("broadcast_dim1_reverse", (4, 16, 64), (4, 1, 64), True, reverse=True),
    Case("broadcast_dim1_unrolled", (4, 32, 128), (4, 1, 128), True),
    Case(
        "broadcast_dim1_unrolled_reverse",
        (4, 32, 128),
        (4, 1, 128),
        True,
        reverse=True,
    ),
    Case("broadcast_dim2_tail", (4, 16, 65), (4, 16, 1), True),
    Case("broadcast_n1", (4, 16, 64), (16, 1), True),
    Case("broadcast_m11", (4, 16, 64), (4, 1, 1), True),
    Case("broadcast_k", (4, 32, 128), (128,), True),
    Case("broadcast_scalar", (4, 32, 128), (1,), True),
    Case(
        "transpose_tail",
        (2, 17, 65),
        (2, 17, 65),
        True,
        transpose_input=True,
    ),
    Case("broadcast_4d_middle", (2, 3, 4, 5), (2, 1, 4, 5), True),
    Case("vector_tail_fallback", (127,), (127,), False),
    Case("multi_broadcast_fallback", (2, 3, 4), (1, 3, 1), False),
)

INPLACE_CASES = (
    Case("contiguous_tail", (17, 65), (17, 65), True),
    Case("vector_aligned", (256,), (256,), True),
    Case("broadcast_dim0", (4, 16, 64), (1, 16, 64), True),
    Case("broadcast_dim1", (4, 16, 64), (4, 1, 64), True),
    Case("broadcast_dim1_unrolled", (4, 32, 128), (4, 1, 128), True),
    Case("broadcast_dim2_tail", (4, 16, 65), (4, 16, 1), True),
    Case("broadcast_n1", (4, 16, 64), (16, 1), True),
    Case("broadcast_m11", (4, 16, 64), (4, 1, 1), True),
    Case(
        "transpose_tail",
        (2, 17, 65),
        (2, 17, 65),
        True,
        transpose_other=True,
    ),
    Case("vector_tail_fallback", (127,), (127,), False),
    Case("multi_broadcast_fallback", (2, 3, 4), (1, 3, 1), False),
)


def make_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, transpose: bool
) -> torch.Tensor:
    if transpose:
        assert len(shape) == 3
        m, n, k = shape
        tensor = (torch.rand((m, k, n), dtype=dtype, device="cuda") + 0.5).transpose(
            1, 2
        )
        assert tensor.shape == shape
        assert not tensor.is_contiguous() and tensor.stride(1) == 1
        return tensor
    return torch.rand(shape, dtype=dtype, device="cuda") + 0.5


def make_operands(
    case: Case, input_dtype: torch.dtype, other_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    input = make_tensor(case.input_shape, input_dtype, case.transpose_input)
    other = make_tensor(case.other_shape, other_dtype, case.transpose_other)
    if case.reverse:
        input, other = other, input
    return input, other


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerance = (
        1e-2
        if actual.dtype in (torch.float16, torch.bfloat16)
        else 1e-5
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def run_out_case(
    op_name: str,
    input_dtype: torch.dtype,
    other_dtype: torch.dtype,
    case: Case,
) -> float:
    input, other = make_operands(case, input_dtype, other_dtype)
    expected = TORCH_OUT[op_name](input, other)

    native_output = torch.empty(
        expected.shape, dtype=expected.dtype, device=expected.device
    )
    native = bool(PRIVATE_OUT[op_name](input, other, native_output))
    assert native == case.native, (
        f"routing mismatch for {case.name}: expected native={case.native}, got {native}"
    )
    if native:
        assert_close(native_output, expected)

    actual = PUBLIC_OUT[op_name](input, other)
    assert_close(actual, expected)
    return float((actual.float() - expected.float()).abs().max().item())


def run_inplace_case(
    op_name: str,
    input_dtype: torch.dtype,
    other_dtype: torch.dtype,
    case: Case,
) -> float:
    input, other = make_operands(case, input_dtype, other_dtype)
    expected = input.clone()
    getattr(expected, f"{op_name}_")(other)

    native_output = input.clone()
    native = bool(PRIVATE_INPLACE[op_name](native_output, other))
    assert native == case.native, (
        f"routing mismatch for {case.name}: expected native={case.native}, got {native}"
    )
    if native:
        assert_close(native_output, expected)

    actual = input.clone()
    returned = PUBLIC_INPLACE[op_name](actual, other)
    assert returned.data_ptr() == actual.data_ptr()
    assert_close(actual, expected)
    return float((actual.float() - expected.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated AITER binary operators and their fallbacks"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        nargs="+",
        choices=DTYPES,
        default=["bf16"],
    )
    parser.add_argument(
        "--op",
        nargs="+",
        choices=PUBLIC_OUT,
        default=list(PUBLIC_OUT),
    )
    parser.add_argument(
        "--other-dtype",
        choices=DTYPES,
        help="use a different dtype for the second operand",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "out", "inplace"),
        default="all",
    )
    args = parser.parse_args()

    failures = []
    passed = 0
    print(f"GPU: {torch.cuda.get_device_name()} ({get_gfx()})")
    for dtype_name in args.dtype:
        input_dtype = DTYPES[dtype_name]
        other_dtype_name = args.other_dtype or dtype_name
        other_dtype = DTYPES[other_dtype_name]
        for op_name in args.op:
            modes = []
            if args.mode in ("all", "out"):
                modes.append(("out", OUT_CASES, run_out_case))
            if args.mode in ("all", "inplace"):
                modes.append(("inplace", INPLACE_CASES, run_inplace_case))
            for mode, cases, runner in modes:
                for case in cases:
                    try:
                        max_abs = runner(
                            op_name, input_dtype, other_dtype, case
                        )
                    except Exception as error:
                        failures.append((op_name, dtype_name, mode, case, error))
                        print(
                            f"FAIL op={op_name} dtypes={dtype_name}/{other_dtype_name} "
                            f"mode={mode} "
                            f"case={case.name} native={case.native}: {error}"
                        )
                    else:
                        passed += 1
                        print(
                            f"PASS op={op_name} dtypes={dtype_name}/{other_dtype_name} "
                            f"mode={mode} "
                            f"case={case.name} native={case.native} "
                            f"max_abs={max_abs:.6g}"
                        )

    print(f"binary operator summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"AITER binary correctness failures: {len(failures)}")


if __name__ == "__main__":
    main()
