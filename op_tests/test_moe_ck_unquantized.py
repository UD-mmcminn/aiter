# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness coverage for the unquantized CK fused-MoE path."""

import argparse
import functools
import os

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    ck_moe_stage1,
    fused_moe,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight


def _resolved_callable(func):
    return func.func if isinstance(func, functools.partial) else func


def _assert_ck_route(
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts: int,
    topk: int,
) -> int:
    metadata = get_2stage_cfgs(
        get_padded_M(tokens),
        hidden_dim,
        intermediate_dim,
        experts,
        topk,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.bf16,
        aiter.QuantType.No,
        True,
        aiter.ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
    )
    stage1 = _resolved_callable(metadata.stage1)
    stage2 = _resolved_callable(metadata.stage2)
    if stage1 is not ck_moe_stage1 or stage2 is not aiter.ck_moe_stage2_fwd:
        raise AssertionError(
            "expected untuned CK two-stage MoE route, got "
            f"stage1={stage1!r}, stage2={stage2!r}"
        )
    return metadata.block_m


def _routing(tokens: int, experts: int, topk: int):
    rows = torch.arange(tokens, dtype=torch.int32, device="cuda").view(-1, 1)
    columns = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, -1)
    topk_ids = (rows * 3 + columns * 5) % experts

    weights = torch.arange(1, topk + 1, dtype=torch.float32, device="cuda")
    weights = weights.view(1, -1).expand(tokens, -1).contiguous()
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.contiguous()


def run_case(
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts: int,
    topk: int,
    atol: float,
    rtol: float,
):
    block_m = _assert_ck_route(
        tokens, hidden_dim, intermediate_dim, experts, topk
    )

    hidden_states = (
        torch.randn(tokens, hidden_dim, dtype=dtypes.bf16, device="cuda") * 0.1
    )
    w1 = (
        torch.randn(
            experts,
            intermediate_dim * 2,
            hidden_dim,
            dtype=dtypes.bf16,
            device="cuda",
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            experts,
            hidden_dim,
            intermediate_dim,
            dtype=dtypes.bf16,
            device="cuda",
        )
        * 0.1
    )
    topk_weights, topk_ids = _routing(tokens, experts, topk)

    stage1_ref = torch_moe_stage1(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        doweight=False,
    )
    expected = torch_moe_stage2(
        stage1_ref,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=aiter.QuantType.No,
        doweight=True,
    )

    shuffled_w1 = shuffle_weight(w1, layout=(16, 16))
    shuffled_w2 = shuffle_weight(w2, layout=(16, 16))
    actual = fused_moe(
        hidden_states,
        shuffled_w1,
        shuffled_w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    return max_abs, block_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate unquantized BF16 CK fused MoE"
    )
    parser.add_argument("-t", "--tokens", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--intermediate-dim", type=int, default=512)
    parser.add_argument("-e", "--experts", type=int, default=8)
    parser.add_argument("-k", "--topk", type=int, default=2)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    if args.topk <= 0 or args.topk > args.experts:
        parser.error("topk must be in [1, experts]")
    if args.hidden_dim % 32 or args.intermediate_dim % 32:
        parser.error("hidden and intermediate dimensions must be multiples of 32")

    # This test is specifically for the portable CK defaults. Do not let a
    # locally tuned ASM, Opus, FlyDSL, or CK-Tile row replace either stage.
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
    os.environ["AITER_ONLINE_TUNE"] = "0"

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} ({get_gfx()})")
    passed = 0
    failures = []

    for tokens in args.tokens:
        case = (
            f"tokens={tokens} hidden={args.hidden_dim} "
            f"intermediate={args.intermediate_dim} experts={args.experts} "
            f"topk={args.topk}"
        )
        try:
            max_abs, block_m = run_case(
                tokens,
                args.hidden_dim,
                args.intermediate_dim,
                args.experts,
                args.topk,
                args.atol,
                args.rtol,
            )
        except Exception as error:
            failures.append((case, error))
            print(f"FAIL {case}: {error}")
        else:
            passed += 1
            print(f"PASS {case} block_m={block_m} max_abs={max_abs:.6g}")

    print(f"unquantized CK fused-MoE summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"CK fused-MoE correctness failures: {len(failures)}")


if __name__ == "__main__":
    main()
