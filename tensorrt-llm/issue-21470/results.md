# Results, provenance, and scope

These are archived observations from the initial single-GPU reduction. They are not new measurements of the current admission-based PR.

## Environment

- NVIDIA GeForce RTX 5080, 16303 MiB reported memory, compute capability 12.0.
- Windows/WSL2 Docker with NVIDIA driver 591.74.
- Official image: `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202610300406`.
- Pinned linux/amd64 image manifest: `sha256:c80740b3b3181286924313e9dc5fc1d3f40cd07895b11d766029b014900885b2`.
- Wheel source revision: `ac6fb72d9fa1caf2b2d730aa4797727e6b3f6d23`.
- Python 3.12.3; PyTorch `2.14.0a0+b2c75dd062.nv26.09`; CUDA runtime 13.4.
- No model weights, model forward pass, internal MoE kernel, or multi-rank communicator.

## Exact archived test

`reproducer.py` is byte-for-byte identical to the standalone GPU regression that produced the recorded one-test pass. SHA-256: `e4a56fe32004d35eaa420cad22497ecdbef8f5fcd5ecf2d0ec329a5c8722a933`.

Its helper and test bodies were exported unchanged from the original PR candidate test, based on upstream main `e62e13c29ffb01fab3cbd09862bac47128423c9a`. Their published historical source is [test_kvcm2_integration.py at 82ddb95](https://github.com/Epsilon006/TensorRT-LLM/blob/82ddb95a10f626d317dfc06532fe40e3c7a27f38/tests/unittest/_torch/pyexecutor/kv_cache/test_kvcm2_integration.py). Imports were made explicit so that the issue-version wheel could run the selected test without collecting the newer entire test module. Original copyright and license headers are retained.

The exact pytest arguments run in the prepared persistent container were `python -m pytest -c /dev/null --noconftest -q /results/test_21470_gpu.py`. The public file is renamed `reproducer.py`, with no byte changes. Pytest was installed separately in that container. The article's one-shot container wrapper is provided for convenience and has not been rerun as a wrapper.

## Baseline observations

The original wheel reported configured length 8192, allocatable length 2048, block size 32, table width 68, and page-table shape `[1, 3, 2, 68]`.

The backend type was `tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2.KVCacheManager`.

The recorded exception was `ValueError: User-provided base page indices is too short`, at the installed official module's `resize_context`, line 5265, calling native `kv_cache.resize(capacity)`.

The original baseline harness caught this specific exception and printed `EXPECTED BASELINE REPRODUCED`. When the archived pytest is used directly on the original image, the ValueError is an expected failing test, not a successful test run. An import error, missing CUDA, or a skipped test is not reproduction.

## Initial candidate observations

The experimental one-line change replaced `max_seq_capacity = self.max_seq_len + self._generation_kv_capacity_headroom` with `max_seq_capacity = max_seq_len + self._generation_kv_capacity_headroom` in the installed wheel. This is historical context, not a recommendation for the current fix.

- Allocatable length stayed at 2048; table width became 260; shape became `[1, 3, 2, 260]`.
- The 8192-token allocation returned `False`; capacity and history remained unchanged.
- The first-chunk failure suspended the cache. Preparing again and explicitly requesting 32 tokens succeeded.
- Archived pytest summary: `1 passed, 45 warnings in 6.93s`.
- Original installed Python module SHA-256: `f364e2d19b953b38ba9d0dc8196e1345ab1b2084417cf9fa1a8bd5177084b3ec`.
- Experimental patched module SHA-256: `77b10f2d5afe47be31272d808454ec8206cf55e11c6e736ee49dbda70f88d134`.

## Current proposal snapshot

As checked on November 8, 2026, [PR #21720 at 1726699](https://github.com/NVIDIA/TensorRT-LLM/pull/21720/commits/1726699312807e945875b7669692182a6cc05d65) keeps the table bounded and adds admission checks. Its description reports 39 targeted cases passing in an isolated harness using the issue-version native allocator. That harness includes compatibility adaptations and is not a full matching-main installation. The historical reproducer here does not package or rerun those 39 cases; use the PR's revision-specific tests and validation description for that evidence.

The physical-demand estimate is limited to supported dense configurations. The independent fixed page-index admission budget covers a broader set of flags, but specialized managers that rebase indices or override resizing require their own check. Startup chunk adjustment and general CUDA index-range hardening are outside the PR's stated scope.

## Limits

The initial experiment bypasses the real scheduler. Its 32-token retry is manual, and no inference about automatic chunk shrinking follows. It does not establish request-local failure, multi-rank survival, specialized-model behavior, or B200/GB300 performance. Full relevant modules with a matching build, official CI, and the original 16-rank deployment remained outstanding in the PR description when this note was published.

## Public references

- [Original issue](https://github.com/NVIDIA/TensorRT-LLM/issues/21470).
- [Initial single-GPU findings](https://github.com/NVIDIA/TensorRT-LLM/issues/21470#issuecomment-6635617984), posted November 5, 2026 at 00:10 China time (November 4 UTC).
- [Review identifying the request-lifecycle and table-growth gaps](https://github.com/NVIDIA/TensorRT-LLM/pull/21720#pullrequestreview-5812542254).
- [Current PR](https://github.com/NVIDIA/TensorRT-LLM/pull/21720), whose head and scope can evolve after this note.
