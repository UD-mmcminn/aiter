# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness coverage for the unquantized KV-cache primitives."""

import argparse

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

SLOT_DTYPES = {
    "i32": torch.int32,
    "i64": torch.int64,
}


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return float((actual.float() - expected.float()).abs().max().item())


def _error_text(error: Exception) -> str:
    text = str(error)
    if len(text) <= 1000:
        return text
    return f"{text[:1000]}\n... error text truncated ..."


def run_reshape_case(
    dtype: torch.dtype, slot_dtype: torch.dtype, asm_layout: bool
) -> float:
    num_tokens, num_blocks, num_heads, head_size, block_size = 7, 4, 3, 80, 16
    pack = 16 // dtype.itemsize
    slots = (0, 17, -1, 31, 48, 62, 7)

    key = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device="cuda"
    )
    value = torch.randn_like(key)
    key_cache = torch.full(
        (num_blocks, num_heads, head_size // pack, block_size, pack),
        -3,
        dtype=dtype,
        device="cuda",
    )
    value_shape = (
        (num_blocks, num_heads, block_size // pack, head_size, pack)
        if asm_layout
        else (num_blocks, num_heads, head_size, block_size)
    )
    value_cache = torch.full(value_shape, -5, dtype=dtype, device="cuda")
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()

    for token_idx, slot in enumerate(slots):
        if slot < 0:
            continue
        block_idx, block_offset = divmod(slot, block_size)
        expected_key[block_idx, :, :, block_offset, :] = key[token_idx].reshape(
            num_heads, head_size // pack, pack
        )
        if asm_layout:
            expected_value[
                block_idx, :, block_offset // pack, :, block_offset % pack
            ] = value[token_idx]
        else:
            expected_value[block_idx, :, :, block_offset] = value[token_idx]

    slot_mapping = torch.tensor(slots, dtype=slot_dtype, device="cuda")
    aiter.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        asm_layout=asm_layout,
    )
    torch.cuda.synchronize()

    return max(
        _assert_exact(key_cache, expected_key),
        _assert_exact(value_cache, expected_value),
    )


def run_flash_case(dtype: torch.dtype, slot_dtype: torch.dtype) -> float:
    num_tokens, num_blocks, num_heads, head_size, block_size = 7, 4, 3, 80, 16
    slots = (0, 17, -1, 31, 48, 62, 7)

    key = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device="cuda"
    )
    value = torch.randn_like(key)
    cache_shape = (num_blocks, block_size, num_heads, head_size)
    key_cache = torch.full(cache_shape, -3, dtype=dtype, device="cuda")
    value_cache = torch.full(cache_shape, -5, dtype=dtype, device="cuda")
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()

    for token_idx, slot in enumerate(slots):
        if slot < 0:
            continue
        block_idx, block_offset = divmod(slot, block_size)
        expected_key[block_idx, block_offset] = key[token_idx]
        expected_value[block_idx, block_offset] = value[token_idx]

    slot_mapping = torch.tensor(slots, dtype=slot_dtype, device="cuda")
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    aiter.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        one,
        one,
    )
    torch.cuda.synchronize()

    return max(
        _assert_exact(key_cache, expected_key),
        _assert_exact(value_cache, expected_value),
    )


def run_copy_case(dtype: torch.dtype) -> float:
    cache_shape = (6, 2, 3, 5)
    key_caches = [
        torch.randn(cache_shape, dtype=dtype, device="cuda") for _ in range(2)
    ]
    value_caches = [
        torch.randn(cache_shape, dtype=dtype, device="cuda") for _ in range(2)
    ]
    expected_keys = [cache.clone() for cache in key_caches]
    expected_values = [cache.clone() for cache in value_caches]
    pairs = ((0, 4), (1, 5))
    for expected, original in zip(expected_keys, key_caches):
        for source, destination in pairs:
            expected[destination] = original[source]
    for expected, original in zip(expected_values, value_caches):
        for source, destination in pairs:
            expected[destination] = original[source]

    block_mapping = torch.tensor(pairs, dtype=torch.int64, device="cuda")
    aiter.copy_blocks(key_caches, value_caches, block_mapping)

    max_abs = 0.0
    for actual, expected in zip(
        key_caches + value_caches, expected_keys + expected_values
    ):
        max_abs = max(max_abs, _assert_exact(actual, expected))
    return max_abs


def run_swap_case(dtype: torch.dtype) -> float:
    cache_shape = (6, 2, 3, 5)
    source = torch.randn(cache_shape, dtype=dtype, device="cuda")
    destination = torch.full(cache_shape, -7, dtype=dtype, device="cuda")
    expected = destination.clone()
    pairs = ((0, 3), (2, 1), (5, 4))
    for source_block, destination_block in pairs:
        expected[destination_block] = source[source_block]

    # swap_blocks intentionally consumes a CPU mapping so it can enqueue the
    # individual asynchronous block copies without a device-side setup kernel.
    block_mapping = torch.tensor(pairs, dtype=torch.int64, device="cpu")
    aiter.swap_blocks(source, destination, block_mapping)
    torch.cuda.synchronize()
    return _assert_exact(destination, expected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate unquantized AITER KV-cache primitives"
    )
    parser.add_argument(
        "-d", "--dtype", nargs="+", choices=DTYPES, default=["bf16"]
    )
    parser.add_argument(
        "--slot-dtype",
        nargs="+",
        choices=SLOT_DTYPES,
        default=list(SLOT_DTYPES),
    )
    parser.add_argument(
        "--op",
        nargs="+",
        choices=("reshape", "flash", "copy", "swap"),
        default=("reshape", "flash", "copy", "swap"),
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} ({get_gfx()})")
    passed = 0
    failures = []

    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        if "reshape" in args.op:
            for slot_name in args.slot_dtype:
                for asm_layout in (False, True):
                    case = (
                        f"op=reshape dtype={dtype_name} slot={slot_name} "
                        f"layout={'asm' if asm_layout else 'standard'}"
                    )
                    try:
                        max_abs = run_reshape_case(
                            dtype, SLOT_DTYPES[slot_name], asm_layout
                        )
                    except Exception as error:
                        failures.append((case, error))
                        print(f"FAIL {case}: {_error_text(error)}")
                    else:
                        passed += 1
                        print(f"PASS {case} max_abs={max_abs:.6g}")

        if "flash" in args.op:
            for slot_name in args.slot_dtype:
                case = f"op=flash dtype={dtype_name} slot={slot_name} layout=NHD"
                try:
                    max_abs = run_flash_case(dtype, SLOT_DTYPES[slot_name])
                except Exception as error:
                    failures.append((case, error))
                    print(f"FAIL {case}: {_error_text(error)}")
                else:
                    passed += 1
                    print(f"PASS {case} max_abs={max_abs:.6g}")

        for op_name, runner in (("copy", run_copy_case), ("swap", run_swap_case)):
            if op_name not in args.op:
                continue
            case = f"op={op_name} dtype={dtype_name}"
            try:
                max_abs = runner(dtype)
            except Exception as error:
                failures.append((case, error))
                print(f"FAIL {case}: {_error_text(error)}")
            else:
                passed += 1
                print(f"PASS {case} max_abs={max_abs:.6g}")

    print(f"unquantized KV-cache summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"KV-cache correctness failures: {len(failures)}")


if __name__ == "__main__":
    main()
