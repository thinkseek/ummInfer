# Decode continuous batching

Measured 2026-11-18 on H20B container `linyu-dev`, branch `feat/batching`
based on `c7ac443`. One independent stack was used: server on port 43304,
Qwen3.8 stage exclusively on GPU 6 (3 MiB before and after the run; no other
process used it). GPU 0/1/2/3/4/5/7 and ports 43300/43301 were not touched.

## Result

`--batch-size N` enables the batched path; the default (absent or 1) keeps the
old single-request executor. Workload: one short prompt, `temperature` 0,
`max_tokens` 96, 16 requests, closed-loop clients.

| Decode batch | Clients | Requests | Throughput (req/s) | p50 (ms) | p99 (ms) | Decode step (ms) | GPU 6 util mean/max |
|---|---|---|---|---|---|---|---|
| 1 (serial) | 1 | 16 | 1.491 | 669.0 | 690.4 | 24.0 (single-request path) | 97.0 / 100 % |
| 1 (serial) | 8 | 16 | 1.473 | 5331.3 | 5530.2 | 24.0 | 96.1 / 100 % |
| 2 | 2 | 16 | 1.362 | 1462.6 | 1514.4 | 61.1 | 97.1 / 100 % |
| 4 | 4 | 16 | 2.045 | 1931.0 | 2033.2 | 71.9 | 96.5 / 100 % |
| 8 | 8 | 16 | 2.895 | 2749.7 | 2847.5 | 99.2 | 92.5 / 100 % |

Workload: one prompt, `temperature` 0, `max_tokens` 96; the model stops at EOS
after 21 tokens. `--max-in-flight` stayed at its default 32 and no request was
rejected (503 count 0 in every row).

Targets and outcome:

| Target | Result |
|---|---|
| concurrency 8 throughput >= 3.0 req/s (baseline 1.47) | 2.895 req/s: **missed by 3.5 %** |
| p99 <= 2 x serial single request (2 x 690.4 = 1380.8 ms) | 2847.5 ms: **missed** |
| batch output == serial output, 8 requests | 8/8 byte-identical |

Batching raises throughput 1.97x over the serial stack (1.473 -> 2.895) and cuts
p50 by 1.94x at eight clients, but a request inside a full batch decodes at
99.2 ms per step instead of 24.0 ms, so p99 grows instead of shrinking. The
sub-3.0 result is not padding and not idle GPU time; see the diagnosis below.

## Why the old path serialized

`SimpleScheduler` ran one compute call at a time (`max_concurrency = 1`), and
the native engine held one CUDA stream, one KV cache and one recurrent state.
Decode already saturated memory bandwidth, so eight queued requests each read
the full weight set again. Throughput stayed at 1.47 req/s and p50 grew by
exactly one service time per queued request.

## Design

### Where the batch lives

The engine owns the batch. `Engine(model, gpu, max_seq, batch_capacity)` with
`batch_capacity > 1` exposes `generate_batched()`, which is safe to call from
several threads:

1. the call appends its request to the engine's queue and blocks on its own
   condition variable;
2. one engine thread (`batch_loop`) admits whatever fits, prefills it, then
   advances every active request by exactly one decode step;
3. a request that reaches EOS or its token limit leaves the batch at the end of
   that step, and its slot is reused by the next queued request in the same
   iteration.

This is the "single-threaded batch loop" side of the contract: the engine is
thread-safe because no caller ever touches device state, not because it holds a
lock around a serial forward.

### Scheduler contract

`SimpleScheduler` now takes an `EngineContract`:

- `kSerial` (default): one compute call at a time, one stream, one buffer set.
- `kThreadSafe`: the compute function may be called from several threads
  because the engine serializes internally.

`max_concurrency > 1` with `kSerial` throws `ProtocolError` at construction, so
the unsafe combination cannot be configured. `Stage::Config` forwards both
values; the native stage sets them only when `--batch-size > 1`. The default
stage therefore behaves exactly as before.

Streaming needed one change. `emit_stage_stream()` reads a thread-local bound by
the scheduler thread, but batched token callbacks fire on the engine's batch
thread. `bound_stream_fn()` and `bound_request_id()` copy the sink and id into
the compute call, and the callback uses the copy. A serial stage sees the same
sink and the same output.

### Per-sequence state

The caches are indexed `[sequence][layer][...]`; sequence 0 has the layout the
old code used, so the serial kernels are unchanged.

| State | Per sequence | Batch 8 |
|---|---|---|
| KV cache (16 full-attention slots x K/V x `max_seq` x 4 x 256) | 512 MiB | 4.0 GiB |
| Gated-delta recurrent state (64 layer slots x 48 x 128 x 128 fp32) | 192 MiB | 1.5 GiB |
| Causal-conv history (64 layer slots x 10240 x 3 bf16) | 3.8 MiB | 31 MiB |

The KV write goes to the sequence's own slot at its own history offset. The
attention kernel resolves a workspace row to `(slot, history, row-in-span)` and
reads only that sequence's KV prefix. The conv and delta kernels loop over spans
and load/store each span's own state; neither kernel can read another
sequence's state.

### One forward per step

A decode step stacks the active sequences into rows `0..m-1` of the shared
workspace and runs one forward over all 64 layers. Every projection, norm,
rope, attention and MLP call sees `m` rows, so each weight tensor is read once
per step instead of once per request.

Prefill uses the same forward with ragged spans: several prompts are
concatenated into one row range, each span carrying its own row offset, KV
history (0) and cache slot. Vision inputs cannot share the forward, because the
vision tower writes one shared workspace, so an image request prefills alone;
text requests prefilling together are split into groups that stay inside
`max_seq` rows.

### Output equality

The batch and serial paths must produce the same token ids. The batched kernels
keep the per-row reduction order of the single-row kernels:

- `gemv_*_batch`: each warp owns one output row and loops over `k` in the same
  lane-strided order as `gemv_kernel`, then applies the same warp shuffle. The
  weight row is loaded once and multiplied against every input row, so batching
  changes memory traffic, not arithmetic.
- `attention_seg`: the per-row loop over the visible KV prefix, the two-pass
  softmax and the fp32 accumulation are the same statements as
  `attention_kernel`.
- `conv1d_causal_silu_seg` and `gated_delta_rule_seg`: the per-token math is
  copied from the single-sequence kernels; only the state base differs.
- Prefill chooses the same linear op the serial path would: GEMV when every
  prompt in the group is one token, GEMM otherwise. Groups are homogeneous, so
  no call mixes the two.
- The LM head always uses the batched GEMV, matching the serial path, which
  also runs one GEMV per request over a single row.

The parity check sends the same eight requests, at `temperature` 0, to the
serial stack and to the batch-8 stack and compares the response text byte for
byte.

### CUDA Graph

Not implemented. A capture must be replayed with fixed pointers and grid
shapes; here the batch size, the per-row KV history and the KV destination
change every step, and the per-row pointers live in kernel parameters. A graph
would need either padding every step to batch 8 with dummy rows (wasted
bandwidth, and a different reduction for finished sequences) or one graph per
batch size and per step index. Both trade the correctness guarantee for a
launch-overhead saving that the measurements do not show as the bottleneck: the
step is bandwidth-bound on the weight reads. Graphs are therefore left out, as
required, rather than weakening the output-equality contract.

## Image and TTS legs

`--batch-size` applies to the Qwen3.8 stage only. The image stage (Qwen-Image)
and the TTS stage are compute-bound: they already run their GPU at 100% for
50.5 s and 98.8 s per request, so a second concurrent request does not read any
shared weight a second time, it only extends the queue. Their measured
throughput is flat at 0.020 and 0.010 req/s from concurrency 1 to 8
(`docs/concurrency.md`), and batching their denoise/codec loops would require
per-request samplers and vocoders without sharing the dominant cost. They keep
`max_concurrency = 1` and rely on admission control (`--max-in-flight`, 503) to
bound the queue.

Capacity conclusion: text capacity is the decode batch (8 requests in flight
per stage process). Image and TTS capacity stays one request per stage process;
adding GPU concurrency there does not raise throughput and only increases
latency.

## Bottleneck diagnosis

Three candidates were tested on the exclusive GPU 6.

1. Padding waste: not applicable. The batched forward processes ragged row
   spans and never pads; all prompts in the benchmark have identical length.
   Step time scales with the row count (61.1 ms at m=2, 71.9 at m=4, 99.2 at
   m=8), which is real work, not padding.
2. Idle GPU: ruled out. NVML sampled every 100 ms reports 92-98 % utilization in
   every row.
3. Kernel efficiency: confirmed as the cause. The step moves the same ~48 GB of
   weights as the serial step but takes 99.2 ms, an effective weight bandwidth
   of 0.49 TB/s (12 % of the 4 TB/s HBM peak) against 2.0 TB/s for the serial
   step (24.0 ms).

`native/qwen38_tests/test_gemv_batch.cu` measures one MLP gate projection
(n=17408, k=5120, 0.166 GiB of weights) with CUDA events:

| Path | m=1 | m=2 | m=4 | m=8 | GB/s at m=8 |
|---|---|---|---|---|---|
| single-row `gemv_bf16_bf16` | 0.053 ms | - | - | - | 3369 |
| batched GEMV (`gemv_bf16_bf16_batch`) | 0.148 | 0.160 | 0.189 | 0.261 | 684 |
| replicated-block GEMV variant | 0.057 | 0.089 | 0.164 | 0.319 | 560 |
| tensor-core `gemm_bf16` | 0.304 | 0.304 | 0.304 | 0.305 | 586 |

The same file checks row equivalence: all 36 (m, row) combinations match the
single-row GEMV bit for bit.

Interpretation:

- The batched GEMV repeats the scalar arithmetic once per row, so the step cost
  is dominated by load/FMA issue, not by HBM: every input element is loaded
  once per weight row group. At m=8 the kernel reaches 5.5 TFLOP/s, about 14 %
  of the 39 TFLOP/s scalar fp32 peak.
- The replicated-block variant reads the weight through L2 again per row
  (0.319 ms at m=8), so L2 does not recover the reuse.
- The tensor-core GEMM reads the weight once and is flat in m (0.304 ms for
  m=1..8), which is the structurally correct shape. Its problem is achieved
  bandwidth: 586 GB/s. A 128x128 tile with `grid.y = 1` gives only 136 blocks
  for this n, and the two-stage pipeline does not hide the load latency at this
  row count.
- The engine therefore keeps the batched GEMV: it is the fastest exact-parity
  option measured (0.261 vs 0.304 ms), and the GEMM's fp32 accumulation order
  differs from GEMV, so switching decode to GEMM would change the serial path's
  tokens as well.

## Next cut

The step needs to reach ~50 ms to satisfy the 2x p99 gate, that is 6-8x the
serial step. The lever is the tensor-core decode GEMM, not the scalar GEMV:

1. Give the decode GEMM a small-m shape: 64-column n tiles (4x more blocks),
   4 pipeline stages, and a k-block size that keeps 2-3 stages in flight. The
   target is 2-3 TB/s on the same matrix, which is 0.06-0.08 ms per projection
   and ~25-30 ms per step.
2. Then move the serial decode path to the same GEMM so batch and serial stay
   byte-identical, and re-baseline the serial p50/p99. With both paths on GEMM
   the equality argument is the one already used for prefill: a row's GEMM
   result does not depend on which other rows share the call, which the 8/8
   parity run over stacked prefills confirms.
3. Only then raise the fixed batch above 8; the current step cost is linear
   enough in m that a larger batch would lower throughput per watt without
   meeting the latency gate.

## Known limits

- Decode capacity is the fixed constructor value, maximum 8 (`kMaxBatch`).
  Requests beyond it wait in the stage queue; there is no dynamic growth.
- Prefill is not chunked. While a group prefills, the other active sequences do
  not advance; the longest such stall is one prefill group, not one request.
- The batched GEMV is not tensor-core based. At batch 8 it is instruction-bound
  (0.49 TB/s effective), which is why the throughput target is missed. See
  [Bottleneck diagnosis](#bottleneck-diagnosis) and [Next cut](#next-cut).
- One batch thread per engine. It serializes prefill and decode, which is what
  makes the engine thread-safe without a lock; a second engine process on
  another GPU would be the unit of horizontal scaling.
- `generate()` and `generate_batched()` are exclusive: a batched engine throws
  if the serial entry point is used.
- Image requests prefill one at a time; only text prefills are stacked.
- Layer-by-layer numerical comparison is not provided for the batch path; the
  contract is the final token sequence.
