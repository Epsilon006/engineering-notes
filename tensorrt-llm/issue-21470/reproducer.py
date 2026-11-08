# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone #21470 GPU regression; no model weights required.

Run against the unmodified issue-version wheel to reproduce the ValueError.
After the page-table sizing fix, the same test must pass.
The helper and test bodies are copied unchanged from the candidate repository test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX


@dataclass
class _ContextRequest:
    request_id: int
    tokens: list[int]
    context_remaining_length: int
    conversation_id: str
    py_request_id: int = field(init=False)
    py_conversation_params: ConversationParams | None = field(init=False)
    use_conversation_params: bool = True
    lora_task_id: int | None = None
    cache_salt: str | None = None
    kv_cache_retention_config: KvCacheRetentionConfig | None = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_disagg_generation_init_state: bool = False
    is_generation_only_request: bool = False
    is_dummy_request: bool = False
    is_context_only_request: bool = False
    return_perf_metrics: bool = False
    context_current_position: int = 0
    py_served_prefix_end: int = 0
    py_num_connector_matched_tokens: int = 0
    py_connector_allocation_reported: bool = False
    prepopulated_prompt: tuple[int, int] | None = None
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    def __post_init__(self) -> None:
        self.py_request_id = self.request_id
        if not self.use_conversation_params:
            self.py_conversation_params = None
            return
        self.py_conversation_params = ConversationParams(conversation_id=self.conversation_id)

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    @property
    def is_dummy(self) -> bool:
        return self.is_dummy_request

    @property
    def prepopulated_prompt_len(self) -> int:
        if self.prepopulated_prompt is None:
            return 0
        return self.prepopulated_prompt[0]

    def get_tokens(self, beam_id: int = DEFAULT_BEAM_INDEX) -> list[int]:
        assert beam_id == DEFAULT_BEAM_INDEX
        return self.tokens

    def get_tokens_view(self, beam_id: int = DEFAULT_BEAM_INDEX) -> np.ndarray:
        """Mirror LlmRequest.get_tokens_view, which the C++ backend takes on the reuse path.

        The real binding returns a zero-copy contiguous 1-D int32 view of the token buffer;
        the dtype matters because it selects the C++ int32 ingest fast path.
        """
        assert beam_id == DEFAULT_BEAM_INDEX
        return np.asarray(self.tokens, dtype=np.int32)

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.prepopulated_prompt = (length, tokens_per_block)
        self.context_current_position = length


def test_context_larger_than_kv_pool_returns_false_and_can_retry() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    max_seq_len = 8192
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            host_cache_size=0,
            max_gpu_total_bytes=4 << 20,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_num_tokens=max_seq_len,
        mapping=Mapping(),
        dtype=DataType.HALF,
    )
    request = _ContextRequest(1, list(range(max_seq_len)), max_seq_len, "small-pool")
    try:
        assert 0 < manager.max_seq_len < max_seq_len
        assert manager.prepare_context(request)
        cache = manager.kv_cache_map[request.py_request_id]
        capacity, history = cache.capacity, cache.history_length

        assert not manager.resize_context(request, max_seq_len)
        assert (cache.capacity, cache.history_length) == (capacity, history)
        assert not cache.is_active

        assert manager.prepare_context(request)
        assert manager.resize_context(request, manager.tokens_per_block)
        assert cache.capacity == manager.tokens_per_block + manager.num_extra_kv_tokens
    finally:
        manager.free_resources(request)
        manager.shutdown()
