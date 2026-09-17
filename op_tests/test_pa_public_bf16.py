# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import math

import torch

import aiter
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


def _make_nhd_decode_case():
    torch.manual_seed(1)

    batch, num_q_heads, num_kv_heads, head_size = 2, 8, 1, 128
    block_size = 16
    context_lens = torch.tensor([57, 273], dtype=torch.int32, device="cuda")
    max_context_len = int(context_lens.max().item())
    max_blocks_per_seq = math.ceil(max_context_len / block_size)
    num_blocks = batch * max_blocks_per_seq
    scale = 1.0 / math.sqrt(head_size)

    query = torch.randn(
        batch, num_q_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).reshape(batch, max_blocks_per_seq)

    reference = torch.empty_like(query)
    heads_per_kv = num_q_heads // num_kv_heads
    blocks_used = []
    last_page_lens = []
    for batch_idx, seq_len_tensor in enumerate(context_lens):
        seq_len = int(seq_len_tensor.item())
        num_pages = math.ceil(seq_len / block_size)
        physical_blocks = block_tables[batch_idx, :num_pages]
        blocks_used.append(physical_blocks)
        last_page_lens.append((seq_len - 1) % block_size + 1)

        key = (
            key_cache[physical_blocks]
            .reshape(-1, num_kv_heads, head_size)[:seq_len]
            .permute(1, 0, 2)
            .repeat_interleave(heads_per_kv, dim=0)
            .float()
        )
        value = (
            value_cache[physical_blocks]
            .reshape(-1, num_kv_heads, head_size)[:seq_len]
            .permute(1, 0, 2)
            .repeat_interleave(heads_per_kv, dim=0)
            .float()
        )
        scores = torch.einsum("hd,hkd->hk", query[batch_idx].float(), key) * scale
        probabilities = torch.softmax(scores, dim=-1)
        reference[batch_idx] = torch.einsum(
            "hk,hkd->hd", probabilities, value
        ).to(torch.bfloat16)

    page_counts = torch.tensor(
        [blocks.numel() for blocks in blocks_used], dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            page_counts.cumsum(0, dtype=torch.int32),
        ]
    )
    kv_page_indices = torch.cat(blocks_used)
    kv_last_page_lens = torch.tensor(
        last_page_lens, dtype=torch.int32, device="cuda"
    )
    assert kv_indptr.dtype == kv_page_indices.dtype == kv_last_page_lens.dtype

    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        max_context_len,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        reference,
    )


def _make_workspace(query: torch.Tensor, max_context_len: int) -> torch.Tensor:
    partition_size = 256
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = math.ceil(max_context_len / partition_size)
    workspace_bytes = (
        num_seqs
        * num_heads
        * max_num_partitions
        * head_size
        * query.element_size()
        + 2 * num_seqs * num_heads * max_num_partitions * 4
    )
    return torch.empty(workspace_bytes, dtype=torch.uint8, device=query.device)


def test_public_paged_attention_v1_bf16_gqa() -> None:
    (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        max_context_len,
        scale,
        _,
        _,
        _,
        reference,
    ) = _make_nhd_decode_case()

    output = torch.empty_like(query)
    workspace = _make_workspace(query, max_context_len)
    one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    cu_query_lens = torch.arange(
        query.size(0) + 1, dtype=torch.int32, device=query.device
    )
    aiter.paged_attention_v1(
        output,
        workspace,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        context_lens,
        max_context_len,
        None,
        "auto",
        "NHD",
        0.0,
        one,
        one,
    )
    torch.cuda.synchronize()

    max_abs = (output.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    print(f"PASS public BF16 paged-attention v1 GQA: max_abs={max_abs:.8g}")


def test_public_paged_attention_ragged_bf16_gqa() -> None:
    (
        query,
        key_cache,
        value_cache,
        _,
        _,
        max_context_len,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        reference,
    ) = _make_nhd_decode_case()

    partition_size = 256
    max_num_partitions = math.ceil(max_context_len / partition_size)
    output = torch.empty_like(query)
    workspace = _make_workspace(query, max_context_len)
    one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    aiter.paged_attention_ragged(
        output,
        workspace,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        key_cache.size(1),
        max_num_partitions,
        None,
        "auto",
        "NHD",
        0.0,
        one,
        one,
        partition_size=partition_size,
    )
    torch.cuda.synchronize()

    max_abs = (output.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    print(f"PASS public BF16 paged-attention ragged GQA: max_abs={max_abs:.8g}")


if __name__ == "__main__":
    test_public_paged_attention_bf16_gqa()
    test_public_paged_attention_v1_bf16_gqa()
    test_public_paged_attention_ragged_bf16_gqa()
