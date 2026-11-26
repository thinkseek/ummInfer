// SPDX-License-Identifier: Apache-2.0

import CryptoKit
import XCTest
@testable import ummmlx

private func shaHex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func sseFrame(_ frame: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: frame, options: [.sortedKeys])
    return "data: " + String(decoding: data, as: UTF8.self) + "\n\n"
}

private final class TelemetryScriptedSSE: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var bodies: [String: String] = [:]

    static func reset() {
        lock.lock()
        bodies = [:]
        lock.unlock()
    }

    static func set(_ host: String, body: String) {
        lock.lock()
        bodies[host] = body
        lock.unlock()
    }

    static func body(for host: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return bodies[host]
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.9.10.") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let host = url.host,
              let body = Self.body(for: host) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@MainActor
final class ShowcaseTelemetryTests: XCTestCase {
    override func setUp() {
        super.setUp()
        TelemetryScriptedSSE.reset()
    }

    private func makeModel() -> ChatModel {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TelemetryScriptedSSE.self]
        let client = ChatClient(session: URLSession(configuration: config))
        let model = ChatModel(client: client, audioStream: NoOpAudioStreamPlayer())
        model.model = "qwen3-native"
        return model
    }

    func testShowcasePresetDefinitions() {
        XCTAssertEqual(
            ShowcasePreset.allCases,
            [.documentary, .storyboardI2V, .smartTextOnly, .temporalVideoQA]
        )

        XCTAssertEqual(ShowcasePreset.documentary.rawValue, "documentary")
        XCTAssertEqual(ShowcasePreset.documentary.title, "Live Documentary")
        XCTAssertEqual(ShowcasePreset.documentary.subtitle, "Interleaved prose, diagram & voice")
        XCTAssertEqual(ShowcasePreset.documentary.narrativeVisualMode, "auto")
        XCTAssertTrue(ShowcasePreset.documentary.allowTextOnly)
        XCTAssertEqual(ShowcasePreset.documentary.continuityMode, .off)
        XCTAssertEqual(ShowcasePreset.documentary.styleAnchor, "")
        XCTAssertTrue(ShowcasePreset.documentary.fastVideo)
        XCTAssertTrue(ShowcasePreset.documentary.enableAudio)
        XCTAssertEqual(ShowcasePreset.documentary.videoFramePolicy, .uniform)
        XCTAssertFalse(ShowcasePreset.documentary.requiresVideoAttachment)
        XCTAssertNil(ShowcasePreset.documentary.missingVideoHint)

        XCTAssertEqual(ShowcasePreset.storyboardI2V.rawValue, "storyboard_i2v")
        XCTAssertEqual(ShowcasePreset.storyboardI2V.title, "Storyboard I2V Chain")
        XCTAssertEqual(ShowcasePreset.storyboardI2V.subtitle, "Slot 0 keyframe -> Slot 1 anchored I2V")
        XCTAssertEqual(ShowcasePreset.storyboardI2V.narrativeVisualMode, "auto")
        XCTAssertFalse(ShowcasePreset.storyboardI2V.allowTextOnly)
        XCTAssertEqual(ShowcasePreset.storyboardI2V.continuityMode, .i2vChain)
        XCTAssertEqual(ShowcasePreset.storyboardI2V.styleAnchor,
                       "cinematic 35mm film grain, warm lantern twilight, smooth camera motion")
        XCTAssertLessThanOrEqual(ShowcasePreset.storyboardI2V.styleAnchor.utf8.count, 256)
        XCTAssertTrue(ShowcasePreset.storyboardI2V.fastVideo)
        XCTAssertTrue(ShowcasePreset.storyboardI2V.enableAudio)
        XCTAssertEqual(ShowcasePreset.storyboardI2V.videoFramePolicy, .uniform)
        XCTAssertTrue(ShowcasePreset.storyboardI2V.prompt.contains("source=slot-0"))
        XCTAssertFalse(ShowcasePreset.storyboardI2V.requiresVideoAttachment)
        XCTAssertNil(ShowcasePreset.storyboardI2V.missingVideoHint)

        XCTAssertEqual(ShowcasePreset.smartTextOnly.rawValue, "smart_text_only")
        XCTAssertEqual(ShowcasePreset.smartTextOnly.title, "Smart Text-Only")
        XCTAssertEqual(ShowcasePreset.smartTextOnly.subtitle, "Autonomous zero-visual abstention")
        XCTAssertEqual(ShowcasePreset.smartTextOnly.narrativeVisualMode, "auto")
        XCTAssertTrue(ShowcasePreset.smartTextOnly.allowTextOnly)
        XCTAssertEqual(ShowcasePreset.smartTextOnly.continuityMode, .off)
        XCTAssertEqual(ShowcasePreset.smartTextOnly.styleAnchor, "")
        XCTAssertFalse(ShowcasePreset.smartTextOnly.fastVideo)
        XCTAssertFalse(ShowcasePreset.smartTextOnly.enableAudio)
        XCTAssertEqual(ShowcasePreset.smartTextOnly.videoFramePolicy, .uniform)
        XCTAssertFalse(ShowcasePreset.smartTextOnly.requiresVideoAttachment)
        XCTAssertNil(ShowcasePreset.smartTextOnly.missingVideoHint)

        XCTAssertEqual(ShowcasePreset.temporalVideoQA.rawValue, "temporal_video_qa")
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.title, "Temporal Video QA")
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.subtitle, "Motion & scene-cut multi-turn video Q&A")
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.iconName, "sparkles.tv")
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.narrativeVisualMode, "auto")
        XCTAssertTrue(ShowcasePreset.temporalVideoQA.allowTextOnly)
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.continuityMode, .off)
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.styleAnchor, "")
        XCTAssertFalse(ShowcasePreset.temporalVideoQA.fastVideo)
        XCTAssertFalse(ShowcasePreset.temporalVideoQA.enableAudio)
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.videoFramePolicy, .temporal)
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.videoMotionThresholdText, "0.02")
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.videoSceneThresholdText, "0.25")
        XCTAssertTrue(ShowcasePreset.temporalVideoQA.requiresVideoAttachment)
        XCTAssertEqual(ShowcasePreset.temporalVideoQA.missingVideoHint,
                       "Attach a video clip, then press Send")
    }

    func testDefaultControlsUntouchedBeforePresetApplied() {
        let model = makeModel()
        XCTAssertTrue(model.narrativeMode)
        XCTAssertEqual(model.narrativeVisualMode, "auto")
        XCTAssertTrue(model.narrativeAllowTextOnly)
        XCTAssertEqual(model.narrativeContinuityMode, .off)
        XCTAssertEqual(model.narrativeStyleAnchor, "")
        XCTAssertFalse(model.fastVideoMode)
        XCTAssertFalse(model.audioOutput)
        XCTAssertFalse(model.imageOutput)
        XCTAssertFalse(model.videoOutput)
        XCTAssertEqual(model.videoFramePolicy, "uniform")
        XCTAssertEqual(model.videoMotionThresholdText, "0.02")
        XCTAssertEqual(model.videoSceneThresholdText, "0.25")
        XCTAssertFalse(model.nativeAllocatorTelemetry)
        XCTAssertEqual(model.status, "Ready")
        XCTAssertEqual(model.prompt, "")
    }

    func testApplyShowcasePresetsMutatesExactControlsWithoutSending() {
        let model = makeModel()
        var requestCount = 0
        model.onRequest = { _ in requestCount += 1 }

        model.applyShowcasePreset(.documentary)
        XCTAssertTrue(model.narrativeMode)
        XCTAssertEqual(model.narrativeVisualMode, "auto")
        XCTAssertTrue(model.imageOutput)
        XCTAssertTrue(model.videoOutput)
        XCTAssertTrue(model.narrativeAllowTextOnly)
        XCTAssertEqual(model.narrativeContinuityMode, .off)
        XCTAssertEqual(model.narrativeStyleAnchor, "")
        XCTAssertTrue(model.fastVideoMode)
        XCTAssertTrue(model.audioOutput)
        XCTAssertEqual(model.videoFramePolicy, "uniform")
        XCTAssertEqual(model.status, "Ready")
        XCTAssertNil(model.error)
        XCTAssertEqual(
            model.prompt,
            "Explain how a black hole accretion disk produces relativistic jets and gravitational lensing. Speak clearly like a live documentary narrator, include one crisp diagram of the spacetime curvature, and one short motion clip of plasma spiraling into the event horizon."
        )

        model.applyShowcasePreset(.storyboardI2V)
        XCTAssertTrue(model.narrativeMode)
        XCTAssertEqual(model.narrativeVisualMode, "auto")
        XCTAssertTrue(model.imageOutput)
        XCTAssertTrue(model.videoOutput)
        XCTAssertFalse(model.narrativeAllowTextOnly)
        XCTAssertEqual(model.narrativeContinuityMode, .i2vChain)
        XCTAssertEqual(
            model.narrativeStyleAnchor,
            "cinematic 35mm film grain, warm lantern twilight, smooth camera motion"
        )
        XCTAssertTrue(model.fastVideoMode)
        XCTAssertTrue(model.audioOutput)
        XCTAssertEqual(model.videoFramePolicy, "uniform")
        XCTAssertEqual(model.status, "Ready")
        XCTAssertNil(model.error)
        XCTAssertTrue(model.prompt.contains("source=slot-0"))
        XCTAssertEqual(
            model.prompt,
            "Create a two-beat cinematic storyboard of a glowing paper lantern drifting over a misty mountain river at dusk: first illustrate the establishing keyframe as slot-0, then animate the river breeze and water reflections in a motion clip explicitly anchored to source=slot-0."
        )

        model.applyShowcasePreset(.smartTextOnly)
        XCTAssertTrue(model.narrativeMode)
        XCTAssertEqual(model.narrativeVisualMode, "auto")
        XCTAssertTrue(model.imageOutput)
        XCTAssertTrue(model.videoOutput)
        XCTAssertTrue(model.narrativeAllowTextOnly)
        XCTAssertEqual(model.narrativeContinuityMode, .off)
        XCTAssertEqual(model.narrativeStyleAnchor, "")
        XCTAssertFalse(model.fastVideoMode)
        XCTAssertFalse(model.audioOutput)
        XCTAssertEqual(model.videoFramePolicy, "uniform")
        XCTAssertEqual(model.status, "Ready")
        XCTAssertNil(model.error)
        XCTAssertEqual(
            model.prompt,
            "Prove concisely why the square root of 2 is irrational and explain the key step in two sentences."
        )

        // Temporal Video QA with no video attached enables temporal frame selection + defaults
        // and sets a subtle non-blocking status hint without throwing an error or sending.
        model.videoMotionThresholdText = "0.99"
        model.videoSceneThresholdText = "0.88"
        model.applyShowcasePreset(.temporalVideoQA)
        XCTAssertTrue(model.narrativeMode)
        XCTAssertEqual(model.narrativeVisualMode, "auto")
        XCTAssertTrue(model.imageOutput)
        XCTAssertTrue(model.videoOutput)
        XCTAssertTrue(model.narrativeAllowTextOnly)
        XCTAssertEqual(model.narrativeContinuityMode, .off)
        XCTAssertEqual(model.narrativeStyleAnchor, "")
        XCTAssertFalse(model.fastVideoMode)
        XCTAssertFalse(model.audioOutput)
        XCTAssertEqual(model.videoFramePolicy, "temporal")
        XCTAssertEqual(model.videoMotionThresholdText, "0.02")
        XCTAssertEqual(model.videoSceneThresholdText, "0.25")
        XCTAssertNil(model.error)
        XCTAssertEqual(model.status, "Attach a video clip, then press Send")
        XCTAssertEqual(
            model.prompt,
            "Watch this attached video clip and summarize the key visual events, motion changes, and scene cuts in chronological order."
        )

        // Attaching a video clears the subtle hint back to Ready.
        model.attachedVideoURL = URL(fileURLWithPath: "/tmp/sample-clip.mp4")
        XCTAssertEqual(model.status, "Ready")

        // Applying Temporal Video QA while a video is already attached keeps status Ready.
        model.applyShowcasePreset(.temporalVideoQA)
        XCTAssertEqual(model.videoFramePolicy, "temporal")
        XCTAssertEqual(model.status, "Ready")
        model.attachedVideoURL = nil

        // Switching away from Temporal Video QA to any other preset resets policy back to uniform.
        for other in [ShowcasePreset.documentary, .storyboardI2V, .smartTextOnly] {
            model.applyShowcasePreset(.temporalVideoQA)
            model.videoMotionThresholdText = "0.15"
            model.videoSceneThresholdText = "0.45"
            XCTAssertEqual(model.videoFramePolicy, "temporal")

            model.applyShowcasePreset(other)
            XCTAssertEqual(model.videoFramePolicy, "uniform",
                           "switching from temporalVideoQA to \(other.rawValue) must reset to uniform")
            XCTAssertEqual(model.videoMotionThresholdText, "0.02")
            XCTAssertEqual(model.videoSceneThresholdText, "0.25")
            XCTAssertEqual(model.status, "Ready")
            XCTAssertNil(model.error)
        }

        XCTAssertEqual(requestCount, 0)
        XCTAssertTrue(model.turns.isEmpty)
        XCTAssertFalse(model.isRunning)
        XCTAssertFalse(model.nativeAllocatorTelemetry,
                       "Showcase presets must not auto-enable nativeAllocatorTelemetry")
    }

    func testTurnTelemetryFormattingAndBounds() {
        XCTAssertTrue(TurnTelemetry().formattedPills.isEmpty)

        // Negative and non-finite timings are rejected and omitted.
        let invalid = TurnTelemetry(
            ttftMs: -1,
            ttfaMs: .nan,
            visualSettledCount: 0,
            firstVisualMs: .infinity,
            hasI2VChain: false,
            totalMs: -.infinity
        )
        XCTAssertTrue(invalid.formattedPills.isEmpty)
        XCTAssertNil(TurnTelemetry.formatDuration(-0.1))
        XCTAssertNil(TurnTelemetry.formatDuration(.nan))
        XCTAssertNil(TurnTelemetry.formatDuration(.infinity))

        // Boundary formatting: ms below 1000, s at and above 1000 (including 999.6 rounding).
        XCTAssertEqual(TurnTelemetry.formatDuration(0), "0 ms")
        XCTAssertEqual(TurnTelemetry.formatDuration(265), "265 ms")
        XCTAssertEqual(TurnTelemetry.formatDuration(999.4), "999 ms")
        XCTAssertEqual(TurnTelemetry.formatDuration(999.6), "1.00 s")
        XCTAssertEqual(TurnTelemetry.formatDuration(1420), "1.42 s")

        // Full multimodal turn with I2V chain.
        let full = TurnTelemetry(
            ttftMs: 265,
            ttfaMs: 380,
            visualSettledCount: 2,
            firstVisualMs: 4120,
            hasI2VChain: true,
            totalMs: 8790
        )
        XCTAssertEqual(
            full.formattedPills.map(\.label),
            ["TTFT 265 ms", "TTFA 380 ms", "2 visuals (4.12 s)", "I2V", "Total 8.79 s"]
        )

        // Text-only turn omits TTFA, visuals, and I2V.
        let textOnly = TurnTelemetry(ttftMs: 184, totalMs: 1120)
        XCTAssertEqual(textOnly.formattedPills.map(\.label), ["TTFT 184 ms", "Total 1.12 s"])

        // Singular visual while streaming (no total yet).
        let singleVisual = TurnTelemetry(
            ttftMs: 92,
            visualSettledCount: 1,
            firstVisualMs: 940
        )
        XCTAssertEqual(
            singleVisual.formattedPills.map(\.label),
            ["TTFT 92 ms", "1 visual (940 ms)"]
        )

        // Native allocator telemetry formatting: COVERED, NOT_COVERED (null -> N/C), and absent.
        let coveredTelemetry = TurnTelemetry(
            ttftMs: 140,
            totalMs: 1200,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm", nativeStage: "qwen38",
                        coverage: "COVERED",
                        allocatedBytesPeak: Int64((128.4 * 1024.0 * 1024.0).rounded()),
                        epochID: 17)
                ]
            )
        )
        XCTAssertEqual(
            coveredTelemetry.formattedPills.map(\.label),
            ["TTFT 140 ms", "Total 1.20 s", "VRAM: 128.4 MiB peak"]
        )
        let coveredVram = coveredTelemetry.formattedPills.first { $0.kind == .vram }
        XCTAssertTrue(coveredVram?.help?.contains("vlm/qwen38") == true)
        XCTAssertTrue(coveredVram?.help?.contains("epoch #17") == true)

        let notCoveredTelemetry = TurnTelemetry(
            ttftMs: 120,
            totalMs: 950,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "NOT_COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm", nativeStage: "qwen38",
                        coverage: "NOT_COVERED",
                        allocatedBytesPeak: nil,
                        reason: "batched_request_attribution_unavailable")
                ]
            )
        )
        XCTAssertEqual(
            notCoveredTelemetry.formattedPills.map(\.label),
            ["TTFT 120 ms", "Total 950 ms", "VRAM: N/C"]
        )
        let ncVram = notCoveredTelemetry.formattedPills.first { $0.kind == .vram }
        XCTAssertTrue(ncVram?.help?.contains("batched_request_attribution_unavailable") == true)
    }

    func testStreamRecordsObservedTelemetryForMultimodalI2VTurn() async {
        let host = "10.9.10.1"
        let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        let mp4Bytes = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70])
        let pcmBytes = Data(repeating: 0, count: 480 * 2)

        let body = [
            sseFrame([
                "choices": [["delta": ["content": "Opening beat. "], "finish_reason": NSNull()]],
                "unified": ["kind": "vlm_delta", "seq": 0, "session_id": "srv-t1",
                            "stage": "vlm", "is_final": false, "text": "Opening beat. "]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": NSNull()]],
                "unified": [
                    "kind": "artifact", "seq": 1, "session_id": "srv-t1",
                    "stage": "tts", "is_final": false, "text": "",
                    "artifact": [
                        "artifact_id": "aud-0", "media_type": "audio",
                        "mime_type": "audio/pcm", "encoding": "base64",
                        "index": 0, "total": 1,
                        "data": pcmBytes.base64EncodedString(),
                        "sha256": shaHex(pcmBytes), "bytes": pcmBytes.count,
                        "format": "pcm16", "sample_rate": 24_000, "channels": 1
                    ]
                ]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": NSNull()]],
                "unified": [
                    "kind": "visual_slot", "seq": 2, "session_id": "srv-t1",
                    "stage": "image", "slot_id": "slot-0", "slot_index": 0,
                    "media_type": "image", "prompt": "lantern keyframe"
                ]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": NSNull()]],
                "unified": [
                    "kind": "artifact", "seq": 3, "session_id": "srv-t1",
                    "stage": "image", "is_final": false, "text": "",
                    "artifact": [
                        "artifact_id": "img-0", "media_type": "image",
                        "mime_type": "image/png", "encoding": "base64",
                        "index": 0, "total": 1,
                        "data": pngBytes.base64EncodedString(),
                        "sha256": shaHex(pngBytes), "bytes": pngBytes.count,
                        "slot_id": "slot-0", "slot_index": 0
                    ]
                ]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": NSNull()]],
                "unified": [
                    "kind": "visual_slot", "seq": 4, "session_id": "srv-t1",
                    "stage": "video", "slot_id": "slot-1", "slot_index": 1,
                    "media_type": "video", "prompt": "river breeze motion",
                    "continuity_mode": "i2v_chain", "anchor_slot_index": 0
                ]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": NSNull()]],
                "unified": [
                    "kind": "artifact", "seq": 5, "session_id": "srv-t1",
                    "stage": "video", "is_final": false, "text": "",
                    "artifact": [
                        "artifact_id": "vid-1", "media_type": "video",
                        "mime_type": "video/mp4", "encoding": "base64",
                        "index": 0, "total": 1,
                        "data": mp4Bytes.base64EncodedString(),
                        "sha256": shaHex(mp4Bytes), "bytes": mp4Bytes.count,
                        "slot_id": "slot-1", "slot_index": 1
                    ]
                ]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": "stop"]],
                "unified": ["kind": "complete", "seq": 6, "session_id": "srv-t1",
                            "stage": "vlm", "is_final": true, "text": "Opening beat. "]
            ]),
            "data: [DONE]\n\n"
        ].joined()
        TelemetryScriptedSSE.set(host, body: body)

        let model = makeModel()
        model.endpoint = "http://\(host):30000/v1"
        model.applyShowcasePreset(.storyboardI2V)
        model.send()

        let deadline = Date().addingTimeInterval(5)
        while model.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.turns.count, 2)
        let telemetry = model.turns[1].telemetry
        XCTAssertNotNil(telemetry.ttftMs)
        XCTAssertNotNil(telemetry.ttfaMs)
        XCTAssertEqual(telemetry.visualSettledCount, 2)
        XCTAssertNotNil(telemetry.firstVisualMs)
        XCTAssertTrue(telemetry.hasI2VChain)
        XCTAssertNotNil(telemetry.totalMs)
        XCTAssertGreaterThanOrEqual(telemetry.ttftMs ?? -1, 0)
        XCTAssertGreaterThanOrEqual(telemetry.ttfaMs ?? -1, 0)
        XCTAssertGreaterThanOrEqual(telemetry.firstVisualMs ?? -1, 0)
        XCTAssertGreaterThanOrEqual(telemetry.totalMs ?? -1, 0)
        XCTAssertEqual(telemetry.formattedPills.map(\.kind), [.ttft, .ttfa, .visual, .i2v, .total])
    }

    func testStreamTextOnlyOmitsAudioAndVisualTelemetry() async {
        let host = "10.9.10.2"
        let body = [
            sseFrame([
                "choices": [["delta": ["content": "Assume sqrt(2) = p/q."], "finish_reason": NSNull()]],
                "unified": ["kind": "vlm_delta", "seq": 0, "session_id": "srv-t2",
                            "stage": "vlm", "is_final": false, "text": "Assume sqrt(2) = p/q."]
            ]),
            sseFrame([
                "choices": [["delta": [:], "finish_reason": "stop"]],
                "unified": ["kind": "complete", "seq": 1, "session_id": "srv-t2",
                            "stage": "vlm", "is_final": true, "text": "Assume sqrt(2) = p/q."]
            ]),
            "data: [DONE]\n\n"
        ].joined()
        TelemetryScriptedSSE.set(host, body: body)

        let model = makeModel()
        model.endpoint = "http://\(host):30000/v1"
        model.applyShowcasePreset(.smartTextOnly)
        model.send()

        let deadline = Date().addingTimeInterval(5)
        while model.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.turns.count, 2)
        let telemetry = model.turns[1].telemetry
        XCTAssertNotNil(telemetry.ttftMs)
        XCTAssertNil(telemetry.ttfaMs)
        XCTAssertEqual(telemetry.visualSettledCount, 0)
        XCTAssertNil(telemetry.firstVisualMs)
        XCTAssertFalse(telemetry.hasI2VChain)
        XCTAssertNotNil(telemetry.totalMs)
        XCTAssertEqual(telemetry.formattedPills.map(\.kind), [.ttft, .total])
    }
}
