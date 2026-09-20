# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Focused MI100 correctness probe for unquantized Triton MLA decode."""

import argparse
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.attention.mla import mla_decode_fwd


BLOCK_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CASES = (
    (16, (64,)),
    (16, (127,)),
    (128, (65, 191)),
)


def reference(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: tuple[int, ...],
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for seq_idx, seq_len in enumerate(seq_lens):
        num_pages = math.ceil(seq_len / BLOCK_SIZE)
        pages = block_tables[seq_idx, :num_pages].long()
        kv = kv_buffer[pages].reshape(-1, HEAD_SIZE)[:seq_len]

        q_lora = q[seq_idx, :, :KV_LORA_RANK].float()
        q_rope = q[seq_idx, :, KV_LORA_RANK:].float()
        k_lora = kv[:, :KV_LORA_RANK].float()
        k_rope = kv[:, KV_LORA_RANK:].float()
        scores = (q_lora @ k_lora.T + q_rope @ k_rope.T) * softmax_scale
        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        outputs.append(
            (probabilities.float() @ k_lora).to(torch.bfloat16)
        )
    return torch.stack(outputs)


@torch.inference_mode()
def run_case(
    num_query_heads: int,
    seq_lens: tuple[int, ...],
    atol: float,
    rtol: float,
) -> float:
    batch_size = len(seq_lens)
    max_pages = math.ceil(max(seq_lens) / BLOCK_SIZE)
    total_pages = sum(math.ceil(length / BLOCK_SIZE) for length in seq_lens)
    num_blocks = total_pages + 2

    q = torch.randn(
        (batch_size, num_query_heads, HEAD_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv_buffer = torch.randn(
        (num_blocks, BLOCK_SIZE, 1, HEAD_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_tables = torch.zeros(
        (batch_size, max_pages), dtype=torch.int32, device="cuda"
    )
    page_ids = torch.randperm(num_blocks, device="cuda")[:total_pages].int()
    page_start = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        num_pages = math.ceil(seq_len / BLOCK_SIZE)
        block_tables[seq_idx, :num_pages] = page_ids[
            page_start : page_start + num_pages
        ]
        page_start += num_pages

    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = torch.empty(
        (batch_size, num_query_heads, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device="cuda",
    )
    softmax_scale = 1.0 / math.sqrt(HEAD_SIZE)

    result = mla_decode_fwd(
        q=q,
        kv_buffer=kv_buffer,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_kv=max(seq_lens),
        block_tables=block_tables,
        softmax_scale=softmax_scale,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        causal=True,
        q_descale=None,
        kv_descale=None,
        shuffled_kv_cache=False,
    )
    torch.cuda.synchronize()

    expected = reference(q, kv_buffer, block_tables, seq_lens, softmax_scale)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
    return float((result.float() - expected.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate unquantized BF16 Triton MLA decode on gfx908"
    )
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    gfx = get_gfx_runtime()
    if gfx != "gfx908":
        parser.error(f"this focused probe requires gfx908, got {gfx}")

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} ({gfx})")
    passed = 0
    failures = []

    for num_query_heads, seq_lens in CASES:
        case = f"heads={num_query_heads} seq_lens={seq_lens}"
        try:
            max_abs = run_case(num_query_heads, seq_lens, args.atol, args.rtol)
        except Exception as error:
            failures.append((case, error))
            print(f"FAIL {case}: {error}")
        else:
            passed += 1
            print(f"PASS {case} max_abs={max_abs:.6g}")

    print(f"BF16 Triton MLA decode summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"BF16 Triton MLA decode failures: {len(failures)}")


if __name__ == "__main__":
    main()
