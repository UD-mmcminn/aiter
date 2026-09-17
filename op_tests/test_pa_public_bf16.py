# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import math

import torch

from aiter.paged_attn import PagedAttention


def test_public_paged_attention_bf16_gqa() -> None:
    torch.manual_seed(0)

    batch, num_q_heads, num_kv_heads, head_size = 2, 8, 1, 128
    block_size = 16
    seq_lens = torch.tensor([57, 49], dtype=torch.int32, device="cuda")
    blocks_per_seq = 4
    num_blocks = batch * blocks_per_seq
    scale = 1.0 / math.sqrt(head_size)

    query = torch.randn(
        batch,
        num_q_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key_tokens = torch.randn(
        num_blocks,
        num_kv_heads,
        block_size,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_tokens = torch.randn_like(key_tokens)

    pack = 16 // key_tokens.element_size()
    key_cache = (
        key_tokens.reshape(
            num_blocks,
            num_kv_heads,
            block_size,
            head_size // pack,
            pack,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = value_tokens.permute(0, 1, 3, 2).contiguous()
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).reshape(batch, blocks_per_seq)
    one = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    output = PagedAttention.forward_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        kv_cache_dtype="auto",
        num_kv_heads=num_kv_heads,
        scale=scale,
        alibi_slopes=None,
        k_scale=one,
        v_scale=one,
    )
    torch.cuda.synchronize()

    reference = torch.empty_like(output)
    heads_per_kv = num_q_heads // num_kv_heads
    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        physical_blocks = block_tables[batch_idx]
        key = (
            key_tokens[physical_blocks]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(heads_per_kv, dim=0)
            .float()
        )
        value = (
            value_tokens[physical_blocks]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(heads_per_kv, dim=0)
            .float()
        )
        scores = (
            torch.einsum("hd,hkd->hk", query[batch_idx].float(), key) * scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        reference[batch_idx] = torch.einsum(
            "hk,hkd->hd", probabilities, value
        ).to(torch.bfloat16)

    max_abs = (output.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    print(f"PASS public BF16 paged-attention GQA: max_abs={max_abs:.8g}")


if __name__ == "__main__":
    test_public_paged_attention_bf16_gqa()
