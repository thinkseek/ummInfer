# Concurrency and capacity

Measured 2026-11-17 on H20B container `linyu-dev`, commit `455f671` plus the
change described below. One independent stack was used: server
`172.17.0.2:43302`, ASR and Qwen3.8 on GPU 2, image on GPU 6, TTS on GPU 7.
The shared backends on GPU 0/1/3/4/5 and the resident service on 43300/43301
were not touched.

## Method

- Closed-loop driver `tests/concurrency_load.py`: each worker holds one
  request in flight, so offered load equals concurrency when the request count
  is at least the client count. The long image and TTS rows offer fewer
  requests; the table states the count.
- Latency is client wall time from request write to terminal event. Throughput
  is completed requests divided by wall time.
- `/metrics` was scraped once per second during each case. Queue depth, stage
  in-flight, stage durations and NVML GPU gauges come from that scrape; no
  counter is hand-computed.
- Workloads are fixed so cases are comparable:
  - text: one short prompt, `temperature` 0, `max_tokens` 96; the model stops
    at EOS after about 21 tokens.
  - image: 512x512, 8 denoise steps (stage flag `--steps 8`).
  - TTS: `--max-frames 200` (16 s of audio), voice `vivian`.
- Correctness: the same prompt, temperature 0 and image seed 0 are used at
  every concurrency level. The response text and the artifact `sha256` carried
  by the final SSE chunk must be identical to the serial run.

## Baseline

| Workload | Concurrency | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---|---|---|---|---|---|---|---|
| text | 1 | 12 | 1.486 | 671.3 | 673.0 | 692.0 | 0 |
| text | 2 | 12 | 1.485 | 1337.5 | 1338.7 | 1391.5 | 0 |
| text | 4 | 12 | 1.476 | 2674.8 | 2675.5 | 2775.7 | 0 |
| text | 8 | 16 | 1.472 | 5350.9 | 5351.8 | 5516.5 | 0 |
| image | 1 | 3x1 | 0.020 | 50578 | 50578 | 50578 | 0 |
| image | 2 | 2 | 0.021 | 50129 | 97455 | 97455 | 0 |
| image | 4 | 2x2 | 0.020 | 52332 | 101861 | 101861 | 0 |
| image | 8 | 2x2 | 0.020 | 50435 | 97744 | 97744 | 0 |
| tts | 1 | 2x1 | 0.010 | 98775 | 98775 | 98775 | 0 |
| tts | 2 | 2x2 | 0.010 | 98792 | 195497 | 195497 | 0 |
| tts | 4 | 2x3 | 0.010 | 195434 | 292182 | 292182 | 0 |
| tts | 8 | 1x3 | 0.010 | 195421 | 292137 | 292137 | 0 |

Repeats are aggregated by taking the worst run's percentiles. `3x1` means three
runs of one request; `2x3` means two runs of three requests. Image and TTS rows
offer fewer requests than the client count: one long leg already shows the
serialization, and a full burst would only multiply the wait by the number of
queued requests. The image and TTS GPU (6 and 7) reached 100% utilization in
every run above; errors stayed at zero.

## Bottleneck attribution

The measurement puts the bottleneck at the stage, not at the HTTP or engine
layer:

- Throughput does not change (1.47-1.49 req/s) while concurrency rises eight
  times. A front-end or admission bottleneck would also cap throughput, but
  concurrency would not also add exactly one service time per queued request.
- p50 grows linearly: 671 ms, 1338 ms, 2675 ms, 5351 ms for 1/2/4/8. The
  increments are one full request time.
- `/metrics` during the c=8 case: `umm_stage_in_flight{stage="qwen38"}` reached
  8, `umm_queue_depth{queue="unified_admission"}` reached 8, and
  `umm_gpu_utilization_percent{index="2"}` reached 100%. The engine accepted
  all eight; the stage serialized them.
- The stage runs `SimpleScheduler` with its default `max_concurrency=1`, and
  each native engine owns one CUDA stream and one set of buffers. Batching is
  not implemented. Two requests cannot share the GPU leg.

Concurrency inside one stage would therefore add a lock, not throughput. The
useful lever is admission control: bound how many requests may wait behind the
one in-flight slot, and answer overload explicitly.

## Change

- `umm-infer serve --max-in-flight N` sets the unified engine admission cap
  (default 32, unchanged behavior).
- When the cap is reached the HTTP boundary answers `503` with
  `Retry-After: 1` and `{"error":{"type":"overloaded"}}` before any SSE frame
  is written. Previously the queue-full error was reported as a generic
  `pipeline_error`.
- New tests: a burst of eight concurrent submits against a cap of two (exactly
  two admitted, six rejected, both admitted requests complete with the serial
  result), and an HTTP test that holds the only slot and asserts 503,
  `overloaded` and `Retry-After` on the second request.

## After the change

Fast-fail comparison. The driver does not retry, so a rejected client stops.
The cap-32 row used 16 requests (two per client); the cap 2 and 4 rows used 8,
one per client, because those caps reject at submit and the remaining work
would only measure the rejection path.

| Admission cap | Clients | Requests | Completed | 503 | Completed rps | p50 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
| 32 (before) | 8 | 16 | 16 | 0 | 1.470 | 5354.2 | 5526.2 |
| 2 | 8 | 8 | 2 | 6 | 1.340 | 819.3 | 1490.8 |
| 4 | 8 | 8 | 4 | 4 | 1.415 | 2159.3 | 2823.4 |

A cap of 2 cuts the admitted p99 from 5.53 s to 1.49 s (3.7 times) and a cap of
4 cuts it to 2.82 s. The excess requests are answered with 503 instead of
waiting behind the queue. Completed requests per wall second stay at the
capacity of the GPU leg (1.34 and 1.41 against 1.47 in the baseline).

Client retry policy matters. With all 16 requests retried once a second, the
successful calls still complete, but every client wakes at the same instant and
the rejected ones queue behind each other:

| Admission cap | Retry policy | Completed | 503 | p50 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| 2 | fixed `Retry-After: 1` | 16 | 48 | 1333.0 | 10845.1 |
| 4 | fixed `Retry-After: 1` | 16 | 29 | 2661.3 | 10820.2 |
| 1 | fixed `Retry-After: 1` | 16 | 70 | 828.3 | 13843.3 |
| 2 | jitter 100-200 ms, 40 tries | 11 | 240 sent, 5 gave up | 1338.0 | 7567.9 |

Jitter reduces the client tail from 10.8 s to 7.6 s, but it still does not beat
the unthrottled baseline. With eight clients and two slots, six clients race for
every freed slot. Shedding bounds the server queue; it cannot make a client that
insists on completing all 16 requests wait less than the service time times the
number of queued positions. The honest comparison for tail latency is the
fast-fail table: same admitted work, 3.7 times lower p99.

The server-side histogram for admitted requests confirms the bound:
`umm_request_duration_seconds` p99 bucket is `<=10 s` at cap 32, `<=2.5 s` at
cap 2 and `<=1.0 s` at cap 1.

## Correctness

The same prompt, temperature 0 and image seed 0 are replayed at every level.

- Text: `distinct_texts=1` in every case, so all concurrent responses are byte
  identical to the serial response.
- Image: the serial request and both concurrent requests returned the same
  artifact `sha256` (`95b852bab5e6bd37ed3305376390912e014f7b1a3e8b0e7125c6f8b46f02bac5`).
  Image output is byte identical under concurrency.
- TTS: the two concurrent requests returned different WAV hashes, although the
  stage log reports the same 384000 samples, the same peak and an RMS that
  differs in the sixth significant digit (0.0469319 against 0.0469320). A
  sequential control with concurrency 1 also returned two different hashes
  (`76a3e7d3...` and `a88c86b9...`) with identical log statistics. The TTS
  engine is therefore not bit-reproducible from run to run, and concurrency
  does not add any divergence. This is a pre-existing engine property, not a
  serving regression; the samples and the duration are identical, only
  low-order bits differ.

## Capacity

Measured single-GPU capacity and the tail bound:

| Leg | Service time | Capacity | Admitted p99 at cap 2 |
|---|---|---|---|
| text (Qwen3.8, 21 tokens) | 0.67 s | 1.47 req/s | 1.33 s |
| image (512x512, 8 steps) | 50.5 s | 0.020 req/s | 101.1 s |
| tts (200 frames) | 98.8 s | 0.010 req/s | 195.5 s |

- p99 does not degrade versus concurrency 1 only while exactly one request is
  in flight. One queued slot doubles the admitted p99; four slots multiply it
  by four. This is the stage serialization, not a tuning artifact.
- Recommended admission cap: 2 for text, image and TTS. That gives each GPU one
  running and one queued request, and bounds the admitted p99 at twice the
  service time.
- Above the cap the server answers 503 with `Retry-After: 1`. A client that
  cannot retry should set the cap to its real burst size and accept the linear
  wait instead.
- A client that retries must jitter the retry. A fixed one-second wakeup
  synchronizes all rejected clients and raises the client-observed p99 above the
  unthrottled baseline, even though the server-side admitted tail stays bounded.

## Reproduction

```sh
# container, GPU 2/6/7 only
IMAGE_STEPS=8 TTS_FRAMES=200 MAX_IN_FLIGHT=2 \
  sh service/concurrency-stack.sh start

# closed-loop load: one in flight per worker, latency to the terminal event
python3 -I tests/concurrency_load.py --kind text --concurrency 8 --requests 16
# scrape /metrics once per second while a case runs; the gauge maxima and the
# counter deltas are what the tables above report
curl -s http://172.17.0.2:43302/metrics
```

`GET /metrics` exposes the admission queue, the per-stage in-flight gauge, the
stage duration histogram and the per-GPU gauges used above.
