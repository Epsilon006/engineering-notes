# Reproducing TensorRT-LLM #21470's KV page-index error on a single RTX 5080

*From a reported 16-rank incident to an isolated native allocator regression.*

Published November 8, 2026. This is a reproduction and investigation note, not a complete fix report. The original failure was reported by another user; the single-GPU experiment described here is my reduction of its local allocation path. The linked PR is still under review.

## The original failure

[TensorRT-LLM issue #21470](https://github.com/NVIDIA/TensorRT-LLM/issues/21470) describes a deployment on 16 GB300 GPUs using attention data parallelism, expert parallelism, and an internal MoE kernel. The reporter observed that the kernel's persistent workspace left a KV pool slightly too small for one configured context chunk. Startup succeeded and short requests worked, but a full chunk reached an exception in rank 0's executor loop: `ValueError: User-provided base page indices is too short`.

The report then shows the executor terminating all ranks. Increasing the KV memory fraction or reducing the context token budget avoided the reported failure. Those observations were useful constraints: the internal kernel affected available memory, while the exception originated in the KV allocation path.

My starting question was narrower than reproducing that deployment: could the same local exception be reached without the model, the internal kernel, or the other fifteen GPUs?

## Reducing the scope

The traceback led from the scheduler's context allocation into `KVCacheManagerV2.resize_context`, then into the native cache's `resize`. The native code validates the caller-provided page-index span before attempting physical page allocation. That ordering matters: a buffer-capacity error and a normal refusal to allocate pages are different outcomes.

The first experiment therefore kept the real Python manager, CUDA buffers, and C++ allocator, while replacing the surrounding deployment with a small request helper. No model weights were loaded and no model forward pass was required. The helper supplied request fields used by the allocation path; the native allocator itself was not mocked.

This reduction depends on preserving the relevant capacity relationship, not on claiming that a consumer GPU behaves like a GB300 system in every respect. It isolates the local failure that appears before the distributed error-handling response.

## Manufacturing the capacity mismatch

I used the same official image version as the issue, `1.4.0rc3.dev202610300406`, on one RTX 5080. Rather than consume GPU memory with model weights and a large workspace, I explicitly limited the KV pool to 4 MiB.

| Parameter | Isolated experiment |
| --- | --- |
| Device | One RTX 5080, 16 GB class |
| KV pool limit | 4 MiB; host cache and block reuse disabled |
| KV geometry | 2 layers, 2 KV heads, head dimension 128, FP16 |
| Block size | 32 tokens |
| Configured sequence length / context token budget | 8192 / 8192 |
| Maximum batch size | 1 |

For this geometry, one token's KV payload is 2 layers × 2 for K and V × 2 heads × 128 elements × 2 bytes, or 2048 bytes. A 4 MiB pool corresponds to 2048 tokens of this payload; the manager also reported an allocatable sequence length of 2048. An 8192-token context needs four times that payload before considering extra reservations.

Initialization clamped the manager's allocatable length to 2048. The original page-index width followed that clamped value. Including the experiment's one-token generation headroom, block rounding, and four-entry alignment produced 68 entries. Requesting the full 8192-token context then exceeded that index span before the allocator could simply return an allocation refusal.

The baseline reached the official module's `resize_context` at line 5265 and raised the same reported ValueError. That is the strongest observed correspondence with the original incident: the error text and the local allocation/checking path match.

## Baseline versus the initial candidate

The first candidate sized the table from configured `max_seq_len` instead of clamped `self.max_seq_len`. In this experiment, the aligned width became 260. The physical pool remained limited to 2048 tokens.

| Observation | Original image | Initial wider-table candidate |
| --- | --- | --- |
| Allocatable sequence length | 2048 | 2048 |
| Page-index width | 68 | 260 |
| 8192-token context allocation | Page-index ValueError | Returns `False` |
| Capacity and history after refusal | Test stops at exception | Unchanged |
| Suspend, prepare again, manual 32-token retry | Not reached | Succeeds |

The standalone GPU regression passed once against that initial candidate. The smaller retry was explicitly performed by the test. It was not evidence that the scheduler would automatically shrink a failed chunk.

The archived [reproducer](reproducer.py) is that historical regression test. It intentionally fails with the original ValueError on the unmodified image; it is not an assertion of the current PR's final contract. In a Linux shell with GPU-capable Docker, run:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/note:ro" -w /tmp \
  nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202610300406 \
  bash -lc 'python -m pip install pytest && python -m pytest -c /dev/null --noconftest -q /note/reproducer.py'
```

Run this from the note's directory. The command packages the archived pytest invocation for a fresh container; this exact one-shot wrapper has not been rerun. The recorded experiment installed pytest in a persistent container and used the same test invocation. [Results and provenance](results.md) record the image digest, source revision, observed output, and exact test-file hash.

## What review changed

[PR #21720](https://github.com/NVIDIA/TensorRT-LLM/pull/21720) received a substantive review: making the immediate exception disappear did not establish correct scheduling behavior. A permanently impossible request could repeatedly receive `False`, preempt other work, and eventually encounter the scheduler's deadlock protection. Expanding tables with configured context length also raised memory, copy-cost, and 32-bit indexing concerns.

The current proposal consequently keeps tables bounded and adds admission checks. At [revision 1726699](https://github.com/NVIDIA/TensorRT-LLM/pull/21720/commits/1726699312807e945875b7669692182a6cc05d65), it separates a fixed page-index budget from physical-demand estimation for supported dense configurations. Infeasible requests can receive request-local errors, while temporary allocation pressure remains retryable. Specialized cost models retain explicit limits.

The PR description reports 39 targeted passing cases using the issue-version native allocator and an isolated compatibility harness. That evidence is separate from the historical one-test result above and is not a full matching-build or official CI pass.

## What this experiment does not prove

I did not reproduce 16-rank termination, the internal MoE deployment, or GB300 performance. The initial experiment did not exercise the scheduler or the full request lifecycle. The 68-to-260 comparison does not validate the current admission-based proposal. Full relevant modules on a matching build, official CI, and the original deployment remain outstanding in the PR's published validation status.

The reusable method is to preserve the failing resource invariant, execute the real component at that boundary, and state exactly which surrounding behavior has been removed. A small reproducer can make a large failure accessible. A passing local test still needs a system-level argument about what happens to the request next.
