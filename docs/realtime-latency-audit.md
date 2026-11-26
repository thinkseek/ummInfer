# Independent Realtime WebSocket timing/latency audit (Task #41 R3)

## Scope and verdict

This is a **CPU-only reanalysis of retained evidence**, not a new realtime run.
The input is the accepted Task #41 six-session, three-round R3 public Realtime
WebSocket capture. The auditor does not import the live driver or the earlier
verifier, and it does not execute a server, model, GPU, audio device, or network
peer. It independently decodes the retained WebSocket and MessagePack bytes,
recomputes timings and history, and writes output only after all gates pass.

The result is **PASS for evidence integrity and the checked protocol contracts**.
No protocol/timing correctness defect requiring a runtime change was exposed.
One historical **reporting defect** was exposed and is corrected below: the old
`41 responses` label is not a count of real server responses. No runtime file is
changed by this audit.

All timings are observed **digital-loopback** timings from the recorded run.
They are not WAN latency, physical speaker/microphone latency, device jitter,
acoustic echo-cancellation quality, listening quality, or generalized serving
performance. No new physical-audio claim is made. The six-session workload and
its slow cases must not be replaced with the sub-second numbers from other,
different voice benchmarks.

The complete machine report is
[`evidence/realtime-latency-audit-r3.json`](evidence/realtime-latency-audit-r3.json).
It contains every global timing sample with original integer `monotonic_ns`
anchors, all global and grouped outliers, per-session/per-scenario statistics,
PCM chunk and artifact hashes, exact history turns, prompt bindings, reconnect
boundaries, resource summaries, and input file hashes.

## Input identity and exact reproduction

The sealed R3 archive SHA-256 is:

```text
5cffcf16f77f81abf672c6179f827a0da66eb0fd9adc0d551a982b1c2c5d8161
```

After extraction, all 44 files listed by `SHA256SUMS` verify. The manifest's own
SHA-256 is
`a06aaef64e7726d67a8652fa6522b013872d39e0c4bf97d72fcf8f22a2d9b355`.
The retained `run.json` and `independent-verification.json` SHA-256 values are
respectively
`5d01adb537008856302f455a6b3667fad723b1d30623de7827d3688a9c8ab427` and
`30cf8077ab7fa7bafcdc53d030213dc8abca603adc6b14c706968957465a9029`.
The audit does not trust those summaries; it reads them only after verifying
their hashes and then reconciles them with independently derived facts.

Set `UMM_TASK41_EVIDENCE` to the extracted directory named
`umm-realtime-network-41-evidence-r3`, then run exactly:

```sh
(cd "$UMM_TASK41_EVIDENCE" && sha256sum -c SHA256SUMS)
python3 -I python/umminfer_realtime_latency_audit.py \
  "$UMM_TASK41_EVIDENCE" \
  --output docs/evidence/realtime-latency-audit-r3.json
python3 -I tests/realtime_latency_audit_verify.py \
  "$UMM_TASK41_EVIDENCE" docs/evidence/realtime-latency-audit-r3.json
python3 -I tests/realtime_latency_audit_test.py
python3 -I tests/realtime_network_test.py
umm-test-local realtime-stress-test
umm-test-local
```

The audit and verifier commands perform the real retained-evidence analysis and
a fresh-process byte-identical recomputation. The fixture suite uses
`--profile generic`; it needs no retained artifacts or third-party packages.
The realtime stress and full local commands build/run native CPU contract tests,
not a model or serving gate.

Auditor SHA-256:

```text
5be00b638ec34bd7f83cdbd1de8ecb8972ae5392cae95316c8d19c9c42f53c4d
```

Committed report SHA-256 and fresh-process verification output:

```text
7cda1b7f08db2047430630cf59b64ba55e19dc2ccab7fcce0eca54bab058fddc
REALTIME_LATENCY_INDEPENDENT PASS connections=15 connection_response_ids=41 server_responses=30 prompts=30 terminals=27 sha256=7cda1b7f08db2047430630cf59b64ba55e19dc2ccab7fcce0eca54bab058fddc
```

## Independent count reconciliation

| Independently derived quantity | Count |
| --- | ---: |
| Sessions / rounds / WebSocket connections | 6 / 3 / 15 |
| Connection-scoped response-ID entries seen in either direction | **41** |
| Globally unique response IDs | 38 |
| Actual server responses with native VLM submissions | **30** |
| Additional ACK-only, non-server connection entries inside the old count of 41 | 11 |
| ├─ foreign client ACK-only entries (8 globally unique foreign IDs) | 8 |
| └─ stale old-server-response ACK entries after reconnect | 3 |
| Native VLM prompts independently decoded and matched | 30 |
| ASR finals | 30 |
| Terminal callbacks | 27 = 18 completed + 9 cancelled |
| Exactly acknowledged-prefix history turns | 21 |
| PCM `audio.delta` frames / responses with observed audio | 1,024 / 25 |
| Playback ACK requests / accepted echoes / ignored probes | 147 / 66 / 81 |
| Reconnect boundaries / barge-in cancellations | 9 / 9 |

The earlier verifier constructed a response-state entry for **both RX and TX**
`response_id` fields. Consequently, its `responses=41` includes 11 connection
entries that exist only because the client sent a foreign or stale ACK probe.
It also counts repeated connection-local IDs separately. The correct historical
wording is therefore **41 connection-scoped response-ID entries, comprising 30
actual server responses and 11 ACK-only non-server entries**. The `docs/full-duplex.md`
and `docs/feature-map.md` labels now say this explicitly. This is a reporting
correction, not a protocol behavior change.

All 30 native prompt raw submissions were independently MessagePack-decoded.
Their normalized role/content arrays equal a fresh history model evaluated at
the observer timestamp. Every accepted ACK cursor was previously issued on the
same response's `audio.delta` or `audio.done`; every cancellation prefix is the
exact UTF-8 prefix selected by the last accepted cursor. Final ACKs equal the
full UTF-8 text length. The 21 committed turns contain no duplicate or unplayed
suffix.

## Timing method and coverage

Durations are computed in integer nanoseconds before conversion to milliseconds.
Original clocks are never reset, smoothed, or replaced. The WebSocket capture
and VLM observer used the same host monotonic clock, but their records came from
separate threads. A forwarded VLM submit can therefore be logged just before the
client records an already-emitted ASR final; the signed offset is retained in
each response record instead of inventing a cross-thread order.

For each metric, the report gives count, minimum, Q1, median, p95, Q3, maximum,
IQR (`Q3-Q1`), and MAD (median absolute deviation from the median). Quantiles use
linear interpolation at `(n-1)*p`. Tukey outliers lie outside
`[Q1-1.5*IQR, Q3+1.5*IQR]`. Outliers remain in every statistic. Global metrics
retain every sample; grouped sections retain every grouped outlier and all
summary statistics, while grouped non-outlier samples can be selected from the
global series by session, scenario, connection, response, and original clocks.
The report has 28 global distributions, 155 session-metric summaries, 232
scenario-metric summaries, **111 global outliers**, and **375 grouped outliers**.
The tables below round milliseconds for readability; the JSON preserves exact
nanosecond values.

Five responses have no first-audio sample and are never converted to zero:

| Response | Scenario | Reason |
| --- | --- | --- |
| `req-11` | immediate barge-in | cancelled before audio |
| `req-2` | immediate barge-in | cancelled before audio |
| `req-17` | immediate barge-in | cancelled before audio |
| `req-18` | reconnect during generation | connection closed before audio |
| `req-12` | reconnect during generation | connection closed before audio |

### Global ASR/VLM/TTS/TTFA results

Values are milliseconds except the explicitly marked cursor-byte row.

| Metric | n | Median | p95 | IQR | MAD | Min / max | Outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASR speech-stop → final | 30 | 52.040 | 212.947 | 41.324 | 6.850 | 33.822 / 271.063 | 3 |
| ASR last append → final | 30 | 51.362 | 91.369 | 18.539 | 6.846 | 3.267 / 96.703 | 2 |
| ASR speech-start → first partial | 30 | 75.222 | 225.736 | 104.552 | 52.856 | 19.943 / 255.916 | 0 |
| VLM observed prompt → first text | 30 | 820.389 | 6,724.914 | 2,733.221 | 519.049 | 301.267 / 9,312.296 | 2 |
| VLM first → last text delta | 30 | 1,407.346 | 3,015.030 | 537.657 | 210.560 | 0 / 3,508.167 | 4 |
| **TTFA: ASR final → first audio** | **25** | **17,290.723** | **24,890.773** | **21,446.061** | **7,416.189** | **502.371 / 28,170.845** | **0** |
| TTFA: observed VLM prompt → first audio | 25 | 17,290.782 | 24,890.766 | 21,446.257 | 7,416.292 | 502.475 / 28,170.959 | 0 |
| TTS first → last audio delta | 25 | 5,503.434 | 10,290.166 | 4,371.273 | 2,236.053 | 0 / 14,005.047 | 0 |
| TTS audio-delta interarrival | 999 | 101.342 | 281.468 | 110.324 | 21.164 | 0.116 / 523.090 | 42 |
| TTS first audio → `audio.done` | 18 | 5,771.009 | 11,009.393 | 3,080.952 | 1,959.057 | 1,872.573 / 14,005.434 | 1 |
| Last text → `audio.done` | 18 | 17,114.021 | 26,438.801 | 13,244.367 | 5,780.077 | 1,430.529 / 27,960.030 | 0 |
| Observed prompt → `response.completed` | 18 | 22,441.181 | 32,714.674 | 13,755.793 | 5,581.349 | 2,384.910 / 33,883.976 | 0 |
| Observed prompt → `response.cancelled` | 9 | 11,533.374 | 27,137.765 | 9,651.219 | 7,544.320 | 3,379.322 / 28,633.653 | 0 |

The 17.291 s median TTFA is the honest result for this six-session capture; it
does not satisfy a general sub-second TTFA aspiration. The distribution is
strongly scenario-dependent, so a zero Tukey-outlier count does not mean every
case is fast. All 25 exact samples are retained.

### Playback ACK progression and digital jitter

`text_cursor` is a UTF-8 byte cursor, not an audio duration. ACK interarrival is
therefore digital event cadence, not physical playback jitter.

| Metric | n | Median | p95 | IQR | MAD | Min / max | Outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TX playback ACK → accepted RX echo (ms) | 66 | 0.042 | 32.074 | 0.022 | 0.011 | 0.024 / 35.810 | 9 |
| Accepted ACK interarrival (ms) | 48 | 5.098 | 48.435 | 20.535 | 5.003 | 0.081 / 56.525 | 2 |
| Accepted cursor delta (UTF-8 bytes) | 48 | 9 | 199.7 | 31.5 | 9 | 0 / 228 | 12 |

The 81 ignored requests were independently classified as 36 unissued, 18
foreign-ACK probes, 12 decreasing, 12 duplicates after release, and 3 stale
old-response ACKs after reconnect. Accepted decreasing cursors, accepted
unissued cursors, unsolicited accepted echoes, stale events after cancellation,
cross-session/connection response leaks, and duplicate terminal callback kinds
are all zero.

### Barge-in, final old audio, and terminal ordering

| Metric | n | Median | p95 | IQR | MAD | Min / max | Outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First new-utterance append → `response.cancelled` (ms) | 9 | 0.437 | 43.232 | 2.856 | 0.174 | 0.201 / 43.251 | 2 |
| Final old audio → cancellation callback (ms) | 6 | 307.782 | 1,514.108 | 594.805 | 232.763 | 60.165 / 1,771.793 | 1 |
| Cancellation callback → server `speech_started` event (ms) | 9 | 0.066 | 0.121 | 0.031 | 0.015 | 0.034 / 0.137 | 1 |

The old response's final observed `audio.delta` precedes its cancellation event
in all six cancelled responses that had audio; three immediate cancellations had
no audio. Thus there is no positive “cancel → final old audio” interval to
report: post-cancellation old audio would be a stale-event failure. The report
instead preserves the actual `final old audio → cancel` and `append → final old
audio` orderings. Nine of nine cancellations also precede the captured
`speech_started` event by 34–137 microseconds; this is the server's emitted event
order, not a physical speech-onset measurement.

There are six legal `response.completed` then `response.cancelled` sequences for
generation-complete responses awaiting playback, three cancellation-only
responses, 12 completion-only responses, and nine responses with no terminal due
to connection close. `audio.done` precedes `response.completed`; no terminal kind
repeats. Cancellation precedes any subsequent response events, and no late old
text/audio/ACK/completion survives it.

The echo-disabled and two simultaneous double-talk cancellations took 0.411,
0.402, and 0.576 ms respectively from first new append to cancellation. The
protected echo interval contains zero cut-ins. Both double-talk cases retain
exact acknowledged prefixes and cancel in both echo modes. These are digital PCM
results only.

### Reconnect timing and stale suppression

| Metric | n | Median | p95 | IQR | MAD | Min / max | Outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old close TX → new `transcription_session.updated` (ms) | 9 | 1.545 | 250.650 | 248.843 | 0.115 | 1.329 / 250.664 | 0 |
| Old close TX → first new ASR final (ms) | 9 | 2,400.963 | 2,612.857 | 244.465 | 38.892 | 2,357.485 / 2,613.165 | 0 |
| Old close TX → first new audio, when observed (ms) | 7 | 19,961.434 | 29,762.535 | 8,734.774 | 4,350.527 | 15,610.907 / 30,784.010 | 0 |
| Final old audio → old close TX, when observed (ms) | 7 | 0.412 | 173.373 | 165.239 | 0.168 | 0.244 / 176.125 | 0 |

No old response event reappears on a new connection. Late old ACKs remain
unacknowledged and cannot extend history. Reconnect close timing is socket
lifecycle timing; it is not playback drain, DAC latency, or a physical-device
measurement.

## Per-scenario TTFA and control timing

The following table exposes the workload mixture behind the global TTFA number.
`immediate_barge` has zero TTFA samples because all three responses were
cancelled before audio. `post_round_final` is the final fully ACKed response
after each session's three scheduled scenarios. One reconnect-generation sample
is not a distribution estimate; its p95/IQR/MAD are mechanically reported.

| Scenario | TTFA n | TTFA median / p95 / IQR / MAD (ms) | Barge append→cancel n / median (ms) | Reconnect close→updated n / median (ms) |
| --- | ---: | --- | --- | --- |
| `final_ack` | 3 | 24,768.343 / 27,830.595 / 4,576.376 / 3,402.502 | 0 / — | 0 / — |
| `immediate_barge` | 0 | — | 3 / 0.263 | 0 / — |
| `delayed_barge` | 3 | 18,994.295 / 19,038.665 / 4,584.530 / 49.300 | 3 / 43.202 | 0 / — |
| `reconnect_playback` | 3 | 24,845.100 / 24,896.483 / 888.196 / 57.092 | 0 / — | 3 / 1.545 |
| `reconnect_generation` | 1 | 532.243 / 532.243 / 0 / 0 | 0 / — | 3 / 1.430 |
| `teardown_callbacks` | 3 | 13,248.836 / 13,284.789 / 1,782.410 / 39.947 | 0 / — | 3 / 250.630 |
| `post_round_final` | 6 | 19,679.910 / 22,583.254 / 4,583.427 / 2,349.138 | 0 / — | 0 / — |
| Echo/double-talk and follow-ups | 6 | 502.371–542.938 individual samples | 3 / 0.402–0.576 range | 0 / — |

The JSON additionally reports every ASR/VLM/TTS/ACK/terminal/reconnect metric
for each scenario, including sample counts, p95, IQR, MAD, and all outliers.

## Per-session TTFA

| Session | n | Median | p95 | IQR | MAD | Max | Group outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 19,018.092 | 21,521.224 | 1,402.527 | 23.798 | 21,799.349 | 0 |
| 1 | 3 | 19,043.594 | 22,717.580 | 2,917.539 | 1,752.871 | 23,125.801 | 0 |
| 2 | 3 | 17,560.471 | 24,116.637 | 7,485.283 | 7,284.629 | 24,845.100 | 0 |
| 3 | 3 | 22,781.338 | 24,690.106 | 5,826.678 | 2,120.853 | 24,902.192 | 0 |
| 4 | 10 | 532.442 | 23,517.638 | 9,570.034 | 15.286 | 24,768.343 | 1 |
| 5 | 3 | 11,674.063 | 26,521.167 | 9,223.441 | 1,950.100 | 28,170.845 | 0 |

The maximum values in this table are taken from the exact samples in the JSON.
No slow session is omitted from the global or grouped calculations.

## Resource and queue evidence

The resource JSONL contains 3,088 monotonically timed samples per retained
process. Recomputed server RSS values are:

| Quantity | Value |
| --- | ---: |
| Samples | 3,088 |
| Minimum / maximum RSS | 30,220,288 / 51,568,640 bytes |
| Maximum RSS minus first RSS | **21,348,352 bytes** |
| Recorded growth bound | 268,435,456 bytes |
| Result | PASS, independently recomputed |

The recorded production queue capacities remain 1,024 outbound frames, 4,096
partial events, 4,096 unified events, and 64 MiB audio per session. The retained
real run has **zero queue-depth samples**, so this checks the declared capacity
contract only. Queue saturation, an unmeasured real-server saturation point, and
WAN behavior remain `NOT_COVERED`; the deterministic stress suite's synthetic
saturation is not presented as a live saturation measurement.

## Deterministic fail-closed fixtures

`tests/realtime_latency_audit_test.py` provides ten CPU-only tests covering:

- arbitrary byte-split and coalesced WebSocket transport, including masked and
  extended-length frames, plus fragmented realtime JSON messages;
- accepted duplicate ACKs and ignored decreasing/unissued ACKs;
- delayed post-completion barge-in with exact acknowledged-prefix history;
- reconnect with a late old-response ACK and no replay;
- non-monotonic capture clocks, altered frame bytes, forged decreasing accepted
  ACKs, stale audio after cancellation, cross-connection old audio, observer
  prompt tampering, and manifest tampering;
- known-answer median/p95/IQR/MAD and retention of a slow outlier.

Each negative CLI case exits 2 and leaves no output report. The real report is
written atomically only after all input hashes, protocol checks, histories,
counts, timings, and resource gates pass. The fresh-process verifier performs no
write and requires byte-identical output.
