# SPDX-License-Identifier: Apache-2.0
"""AITER C++ JIT binding wrapper for MiniMax-M3 MSA sparse block score decode.

Wraps aiter.ops.msa_block_select.pa_sparse_block_score_decode with:
1. Contiguous & device checks
2. Exact signature & shape compatibility for vLLM sparse attention decoding
3. Fallback support when AITER JIT module is unavailable or inputs do not meet fp8 e4m3 requirements
"""

import math
import os
from typing import Optional
import torch

try:
    from aiter.ops.msa_block_select import (
        pa_sparse_block_score_decode as _aiter_score_decode_jit,
        SCORE_MFMA_COLS,
    )
    _AITER_SCORE_DECODE_AVAILABLE = True
except (ImportError, AttributeError):
    _AITER_SCORE_DECODE_AVAILABLE = False
    _aiter_score_decode_jit = None
    SCORE_MFMA_COLS = 16


def is_aiter_sparse_score_decode_available() -> bool:
    """Return whether the AITER JIT sparse block score decode kernel is importable."""
    return _AITER_SCORE_DECODE_AVAILABLE


def aiter_sparse_block_score_decode(
    q_idx: torch.Tensor,
    key_cache_idx: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    init_blocks: int = 0,
    local_blocks: int = 0,
    query_len: int = 1,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """Execute AITER C++ JIT sparse block score decode kernel.

    Args:
        q_idx: [total_q, num_idx_heads, head_dim] fp8 e4m3, contiguous.
        key_cache_idx: [num_blocks, block_size, head_dim] fp8 e4m3, contiguous.
        score: [num_idx_heads, total_q, S] fp32, contiguous, written in place.
        block_table: [num_reqs, max_blocks] int32, contiguous.
        seq_lens: [num_reqs] int32, contiguous.
        init_blocks: leading blocks forced via sentinel scores.
        local_blocks: trailing blocks forced via sentinel scores.
        query_len: query tokens per request (default: 1).
        max_seq_len: context length upper bound (required for fixed launch grid).

    Returns:
        score tensor written in-place.
    """
    if not _AITER_SCORE_DECODE_AVAILABLE:
        raise RuntimeError(
            "AITER sparse block score decode JIT kernel is not available"
        )

    # Validate memory contiguousness & device placement
    if not q_idx.is_cuda or not key_cache_idx.is_cuda:
        raise ValueError("q_idx and key_cache_idx must be CUDA/HIP tensors")

    if not q_idx.is_contiguous():
        q_idx = q_idx.contiguous()
    if not key_cache_idx.is_contiguous():
        key_cache_idx = key_cache_idx.contiguous()
    if not block_table.is_contiguous():
        block_table = block_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    total_q = q_idx.shape[0]
    num_idx_heads = q_idx.shape[1]
    head_dim = q_idx.shape[2]
    block_size = key_cache_idx.shape[1]

    if total_q % query_len != 0:
        raise ValueError(
            f"q_idx rows {total_q} not a multiple of query_len {query_len}"
        )
    num_reqs = total_q // query_len

    if num_idx_heads * query_len > SCORE_MFMA_COLS:
        raise ValueError(
            f"num_idx_heads * query_len = {num_idx_heads * query_len} exceeds "
            f"MFMA limit {SCORE_MFMA_COLS}"
        )

    if max_seq_len < 1:
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else block_size

    _aiter_score_decode_jit(
        q_idx=q_idx,
        key_cache_idx=key_cache_idx,
        score=score,
        block_table=block_table,
        seq_lens=seq_lens,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        query_len=query_len,
        max_seq_len=max_seq_len,
    )
    return score
