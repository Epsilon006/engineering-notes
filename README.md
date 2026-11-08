# Engineering notes

Reproducible investigations, debugging notes, and the limits of their evidence.

- [A 16-GPU KV cache failure, reduced to 4 MiB—and why a larger buffer was not enough](tensorrt-llm/issue-21470/README.md) — a 4 MiB native-allocator reproducer, the initial candidate, and what review changed.

Each note distinguishes observed results, proposed explanations, and unverified behavior. Results are tied to specific software revisions; an open PR is not a completed fix.

Code is available under [Apache-2.0](LICENSE). Upstream-derived files retain their original copyright and license headers; see [NOTICE](NOTICE).
