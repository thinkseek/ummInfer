// SPDX-License-Identifier: Apache-2.0
// Offline unit tests for Full-Duplex Realtime Voice (Stage B playback_ack + barge-in).

import Foundation
import XCTest
@testable import ummmlx

// MARK: - Test Doubles

private final class MockRealtimeTransport: RealtimeTransport, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var connectedURL: URL?
    private(set) var sentMessages: [String] = []
    private(set) var closedCount = 0
    private var incomingContinuation: AsyncThrowingStream<String, Error>.Continuation?
    private var pendingIncoming: [String] = []
    var connectError: Error?
    var sendError: Error?

    func connect(url: URL) async throws {
        if let connectError { throw connectError }
        lock.withLock { connectedURL = url }
    }

    func send(text: String) async throws {
        if let sendError { throw sendError }
        lock.withLock { sentMessages.append(text) }
    }

    func receive() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let buffered: [String] = self.lock.withLock {
                self.incomingContinuation = continuation
                let copy = self.pendingIncoming
                self.pendingIncoming.removeAll()
                return copy
            }
            for item in buffered {
                continuation.yield(item)
            }
        }
    }

    func close() async {
        let cont: AsyncThrowingStream<String, Error>.Continuation? = lock.withLock {
            closedCount += 1
            let c = incomingContinuation
            incomingContinuation = nil
            return c
        }
        cont?.finish()
    }

    func pushServerJSON(_ object: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        pushServerText(String(decoding: data, as: UTF8.self))
    }

    func pushServerText(_ text: String) {
        let cont: AsyncThrowingStream<String, Error>.Continuation? = lock.withLock {
            if let incomingContinuation {
                return incomingContinuation
            }
            pendingIncoming.append(text)
            return nil
        }
        cont?.yield(text)
    }

    func finishServerStream(throwing error: Error? = nil) {
        let cont: AsyncThrowingStream<String, Error>.Continuation? = lock.withLock {
            let c = incomingContinuation
            incomingContinuation = nil
            return c
        }
        if let error {
            cont?.finish(throwing: error)
        } else {
            cont?.finish()
        }
    }

    func sentJSONObjects() -> [[String: Any]] {
        let messages = lock.withLock { sentMessages }
        return messages.compactMap {
            try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
        }
    }

    func playbackAcks() -> [(responseID: String, textCursor: Int)] {
        sentJSONObjects().compactMap { obj in
            guard (obj["type"] as? String) == "output_audio_buffer.playback_ack",
                  let rid = obj["response_id"] as? String,
                  let cursor = obj["text_cursor"] as? Int else {
                return nil
            }
            return (rid, cursor)
        }
    }
}

private final class MockAudioCapturer: RealtimeAudioCapturing, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private var continuation: AsyncThrowingStream<Data, Error>.Continuation?
    var queuedFrames: [Data] = []
    var failure: Error?

    func startCapture() -> AsyncThrowingStream<Data, Error> {
        let (frames, err): ([Data], Error?) = lock.withLock {
            startCount += 1
            return (queuedFrames, failure)
        }
        return AsyncThrowingStream { continuation in
            self.lock.withLock { self.continuation = continuation }
            for frame in frames {
                continuation.yield(frame)
            }
            if let err {
                continuation.finish(throwing: err)
            }
        }
    }

    func stopCapture() {
        let cont: AsyncThrowingStream<Data, Error>.Continuation? = lock.withLock {
            stopCount += 1
            let c = continuation
            continuation = nil
            return c
        }
        cont?.finish()
    }

    func emitFrame(_ pcm16: Data) {
        let cont: AsyncThrowingStream<Data, Error>.Continuation? = lock.withLock { continuation }
        cont?.yield(pcm16)
    }
}

/// Deterministic audio stream player where the test explicitly controls when each
/// enqueued PCM buffer finishes playing (simulating physical speaker consumption).
@MainActor
private final class ControllableAudioStreamPlayer: AudioStreamPlaying {
    var onStateChange: ((Bool) -> Void)?
    private(set) var isPlaying = false {
        didSet {
            if oldValue != isPlaying { onStateChange?(isPlaying) }
        }
    }
    private(set) var beginCount = 0
    private(set) var finishCount = 0
    private(set) var stopCount = 0
    private(set) var enqueuedChunks: [AudioStreamChunk] = []
    private var pendingCallbacks: [(generation: Int, callback: (@MainActor @Sendable () -> Void)?)] = []
    private var generation = 0

    func beginStream() {
        beginCount += 1
        pendingCallbacks.removeAll()
    }

    func enqueue(_ chunk: AudioStreamChunk) {
        enqueue(chunk, onPlayed: nil)
    }

    func enqueue(_ chunk: AudioStreamChunk, onPlayed: (@MainActor @Sendable () -> Void)?) {
        enqueuedChunks.append(chunk)
        pendingCallbacks.append((generation, onPlayed))
        isPlaying = true
    }

    func finishStream() {
        finishCount += 1
        if pendingCallbacks.isEmpty {
            isPlaying = false
        }
    }

    func stop() {
        generation += 1
        stopCount += 1
        pendingCallbacks.removeAll()
        isPlaying = false
    }

    /// Simulates the hardware audio callback completing the oldest scheduled buffer.
    func playNextChunk() {
        guard !pendingCallbacks.isEmpty else { return }
        let item = pendingCallbacks.removeFirst()
        if item.generation == generation {
            item.callback?()
        }
        if pendingCallbacks.isEmpty {
            isPlaying = false
        }
    }

    /// Simulates all currently scheduled chunks finishing playback in FIFO order.
    func drainAllChunks() {
        while !pendingCallbacks.isEmpty {
            playNextChunk()
        }
    }
}

// MARK: - URL Conversion & Security Policy Tests

final class RealtimeURLTests: XCTestCase {
    @MainActor
    func testConvertsHTTPAndHTTPSBaseURLsToRealtimeWebSocket() throws {
        let cases: [(String, String, String)] = [
            (
                "http://127.0.0.1:43301/v1",
                "qwen3-asr",
                "ws://127.0.0.1:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://127.0.0.1:43301/v1/",
                "qwen3-asr",
                "ws://127.0.0.1:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://localhost:43301",
                "qwen3-asr",
                "ws://localhost:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://[::1]:43301/v1",
                "qwen3-asr",
                "ws://[::1]:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://10.0.0.8:43301/v1",
                "qwen3-asr",
                "ws://10.0.0.8:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://172.16.4.2:43301/v1",
                "qwen3-asr",
                "ws://172.16.4.2:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://192.168.1.9:43301/v1",
                "qwen3-asr",
                "ws://192.168.1.9:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "http://[fd00::1]:43301/v1",
                "qwen3-asr",
                "ws://[fd00::1]:43301/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
            (
                "https://voice.example.com/v1",
                "Qwen/ASR 1.7B",
                "wss://voice.example.com/v1/realtime?intent=transcription&model=Qwen/ASR%201.7B"
            ),
            (
                "wss://voice.example.com/v1",
                "qwen3-asr",
                "wss://voice.example.com/v1/realtime?intent=transcription&model=qwen3-asr"
            ),
        ]
        for (base, model, expected) in cases {
            let url = try RealtimeClient.realtimeURL(base: base, model: model, allowedHTTPHosts: [])
            XCTAssertEqual(url.absoluteString, expected, "base=\(base)")
        }
    }

    @MainActor
    func testAllowlistPermitsNamedPlainHost() throws {
        let url = try RealtimeClient.realtimeURL(
            base: "http://gpu-box.lan:43301/v1",
            model: "asr-model",
            allowedHTTPHosts: ["gpu-box.lan"]
        )
        XCTAssertEqual(
            url.absoluteString,
            "ws://gpu-box.lan:43301/v1/realtime?intent=transcription&model=asr-model"
        )
    }

    @MainActor
    func testRejectsUnsafeHostsCredentialsQueryFragmentAndEmptyModel() {
        let unsafeBases = [
            "http://example.com/v1",
            "ws://example.com/v1",
            "http://fc.example.com/v1",
            "http://fd.example.com/v1",
            "http://172.32.0.1/v1",
            "http://127.example.com/v1",
            "http://user@127.0.0.1/v1",
            "http://user:pass@127.0.0.1/v1",
            "http://127.0.0.1/v1?x=1",
            "http://127.0.0.1/v1#frag",
            "ftp://127.0.0.1/v1",
            "",
        ]
        for base in unsafeBases {
            XCTAssertThrowsError(
                try RealtimeClient.realtimeURL(base: base, model: "m", allowedHTTPHosts: []),
                "Expected rejection for base=\(base)"
            )
        }
        XCTAssertThrowsError(
            try RealtimeClient.realtimeURL(base: "http://127.0.0.1/v1", model: "   ", allowedHTTPHosts: [])
        ) { error in
            XCTAssertEqual(error as? ChatClientError, .modelRequired)
        }
    }
}

// MARK: - Wire Protocol Encoding & Decoding Tests

final class RealtimeWireTests: XCTestCase {
    func testEncodesSessionUpdateWithServerVADAndPlaybackAck() throws {
        let raw = try RealtimeWire.encodeSessionUpdate(RealtimeSessionConfig(
            voice: "vivian",
            silenceMs: 400,
            playbackAck: true,
            maxNewTokens: 256
        ))
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
        )
        XCTAssertEqual(root["type"] as? String, "session.update")
        let session = try XCTUnwrap(root["session"] as? [String: Any])
        XCTAssertEqual(session["input_audio_format"] as? String, "pcm16")
        XCTAssertEqual(session["max_new_tokens"] as? Int, 256)

        let turn = try XCTUnwrap(session["turn_detection"] as? [String: Any])
        XCTAssertEqual(turn["type"] as? String, "server_vad")
        XCTAssertEqual(turn["silence_ms"] as? Int, 400)

        let audio = try XCTUnwrap(session["audio_output"] as? [String: Any])
        XCTAssertEqual(audio["voice"] as? String, "vivian")
        XCTAssertEqual(audio["playback_ack"] as? Bool, true)
    }

    func testEncodesAudioAppendAndRejectsOddByteCount() throws {
        let pcm = Data([0x01, 0x02, 0x03, 0x04])
        let raw = try RealtimeWire.encodeAudioAppend(pcm)
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
        )
        XCTAssertEqual(root["type"] as? String, "input_audio_buffer.append")
        XCTAssertEqual(root["audio"] as? String, pcm.base64EncodedString())

        XCTAssertThrowsError(try RealtimeWire.encodeAudioAppend(Data()))
        XCTAssertThrowsError(try RealtimeWire.encodeAudioAppend(Data([0x01, 0x02, 0x03])))
    }

    func testEncodesPlaybackAck() throws {
        let raw = try RealtimeWire.encodePlaybackAck(responseID: "req-1", textCursor: 12)
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
        )
        XCTAssertEqual(root["type"] as? String, "output_audio_buffer.playback_ack")
        XCTAssertEqual(root["response_id"] as? String, "req-1")
        XCTAssertEqual(root["text_cursor"] as? Int, 12)

        XCTAssertThrowsError(try RealtimeWire.encodePlaybackAck(responseID: "", textCursor: 0))
        XCTAssertThrowsError(try RealtimeWire.encodePlaybackAck(responseID: "req-1", textCursor: -1))
    }

    func testDecodesEveryDocumentedServerEvent() throws {
        let pcm = Data([0x10, 0x00, 0x20, 0x00])
        let b64 = pcm.base64EncodedString()

        let updated = try RealtimeWire.decodeServerEvent(#"""
        {"type":"transcription_session.updated","session":{"input_audio_format":"pcm16","turn_detection":{"type":"server_vad","silence_ms":400},"max_new_tokens":128,"audio_output":{"voice":"vivian","playback_ack":true}}}
        """#)
        XCTAssertEqual(updated, .sessionUpdated(RealtimeSessionSnapshot(
            inputAudioFormat: "pcm16",
            serverVADSilenceMs: 400,
            voice: "vivian",
            playbackAck: true,
            maxNewTokens: 128
        )))

        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"input_audio_buffer.speech_started","segment_id":2,"interrupted_response_id":"req-1"}"#),
            .speechStarted(segmentID: 2, interruptedResponseID: "req-1")
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"input_audio_buffer.speech_stopped","segment_id":2,"audio_ms":920}"#),
            .speechStopped(segmentID: 2, audioMs: 920)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"transcription.segment","event_index":0,"segment_id":2,"text":"Hel","is_final":false}"#),
            .transcriptionSegment(segmentID: 2, eventIndex: 0, text: "Hel", isFinal: false)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"transcription.completed","segment_id":2,"text":"Hello there."}"#),
            .transcriptionCompleted(segmentID: 2, text: "Hello there.")
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"response.text.delta","response_id":"req-1","segment_id":2,"text":"Sounds like ","text_cursor":12}"#),
            .responseTextDelta(responseID: "req-1", segmentID: 2, text: "Sounds like ", textCursor: 12)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent("""
            {"type":"audio.delta","response_id":"req-1","segment_id":2,"format":"pcm16","sample_rate":24000,"channels":1,"audio":"\(b64)","text_cursor":0,"text_end_cursor":12}
            """),
            .audioDelta(RealtimeAudioDelta(
                responseID: "req-1", segmentID: 2, pcm16: pcm,
                sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 12
            ))
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"audio.done","response_id":"req-1","segment_id":2,"text_cursor":24}"#),
            .audioDone(responseID: "req-1", segmentID: 2, textCursor: 24)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"response.completed","response_id":"req-1","segment_id":2,"text_cursor":24}"#),
            .responseCompleted(responseID: "req-1", segmentID: 2, textCursor: 24)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"response.playback_ack","response_id":"req-1","segment_id":2,"text_cursor":24,"final":true}"#),
            .responsePlaybackAck(responseID: "req-1", segmentID: 2, textCursor: 24, isFinal: true)
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"response.cancelled","response_id":"req-1","segment_id":2,"reason":"barge_in","text_cursor":12,"acknowledged_text":"Sounds like ","history_recorded":true}"#),
            .responseCancelled(RealtimeCancellation(
                responseID: "req-1",
                segmentID: 2,
                reason: "barge_in",
                textCursor: 12,
                acknowledgedText: "Sounds like ",
                historyRecorded: true
            ))
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"response.failed","response_id":"req-1","segment_id":2,"error":"tts failed"}"#),
            .responseFailed(responseID: "req-1", segmentID: 2, error: "tts failed")
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"error","error":{"code":"request_error","message":"bad audio"}}"#),
            .error(code: "request_error", message: "bad audio")
        )
        XCTAssertEqual(
            try RealtimeWire.decodeServerEvent(#"{"type":"future.event","x":1}"#),
            .unknown(type: "future.event")
        )
    }

    func testRejectsMalformedAudioDelta() {
        XCTAssertThrowsError(try RealtimeWire.decodeServerEvent(
            #"{"type":"audio.delta","response_id":"r1","format":"pcm16","sample_rate":24000,"channels":1,"audio":"!!!notbase64!!!"}"#
        ))
        // Odd byte count is not sample-aligned for mono PCM16.
        let oddB64 = Data([0x01, 0x02, 0x03]).base64EncodedString()
        XCTAssertThrowsError(try RealtimeWire.decodeServerEvent(
            #"{"type":"audio.delta","response_id":"r1","format":"pcm16","sample_rate":24000,"channels":1,"audio":"\#(oddB64)"}"#
        ))
        // Unsupported format.
        let validB64 = Data([0x01, 0x02]).base64EncodedString()
        XCTAssertThrowsError(try RealtimeWire.decodeServerEvent(
            #"{"type":"audio.delta","response_id":"r1","format":"opus","sample_rate":24000,"channels":1,"audio":"\#(validB64)"}"#
        ))
    }
}

// MARK: - Monotonic Playback Ack & Barge-in State Machine Tests

final class RealtimeStateMachineTests: XCTestCase {
    private let pcm = Data(repeating: 0x11, count: 4_800)

    private func makeAckEnabledStateMachine() -> RealtimeStateMachine {
        var sm = RealtimeStateMachine()
        _ = sm.receive(.sessionUpdated(RealtimeSessionSnapshot(
            inputAudioFormat: "pcm16",
            serverVADSilenceMs: 400,
            voice: "vivian",
            playbackAck: true,
            maxNewTokens: 128
        )))
        return sm
    }

    func testTextDeltaNeverAuthorizesPlaybackAckAndCursorsAdvanceMonotonicallyAfterConsumption() {
        var sm = makeAckEnabledStateMachine()

        // Text emission carries an informational cursor of 12, which must NOT authorize acks.
        let textActions = sm.receive(.responseTextDelta(
            responseID: "req-1", segmentID: 0, text: "Sounds like ", textCursor: 12
        ))
        XCTAssertEqual(textActions, [.forward(.responseTextDelta(
            responseID: "req-1", segmentID: 0, text: "Sounds like ", textCursor: 12
        ))])
        XCTAssertFalse(sm.issuedCursors.contains(12))
        XCTAssertTrue(sm.audioChunkConsumed(responseID: "req-1", textCursor: 12).isEmpty)

        // First sentence emits two chunks both carrying text_cursor=0, text_end_cursor=12.
        let d0 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 12
        )
        let d1 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 12
        )
        // Second sentence carries text_cursor=12, text_end_cursor=24.
        let d2 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 12, textEndCursor: 24
        )
        XCTAssertEqual(sm.receive(.audioDelta(d0)), [
            .beginAudioStream(responseID: "req-1"),
            .enqueueAudio(d0),
            .forward(.audioDelta(d0)),
        ])
        XCTAssertEqual(sm.receive(.audioDelta(d1)), [
            .enqueueAudio(d1),
            .forward(.audioDelta(d1)),
        ])
        XCTAssertEqual(sm.receive(.audioDelta(d2)), [
            .enqueueAudio(d2),
            .forward(.audioDelta(d2)),
        ])

        // Generation completes before playback drains: must NOT ack cursor 24 yet!
        let doneActions = sm.receive(.audioDone(responseID: "req-1", segmentID: 0, textCursor: 24))
        XCTAssertEqual(doneActions, [
            .finishAudioStream(responseID: "req-1"),
            .forward(.audioDone(responseID: "req-1", segmentID: 0, textCursor: 24)),
        ])
        XCTAssertFalse(doneActions.contains {
            if case .sendPlaybackAck = $0 { return true }
            return false
        })

        // Chunk 0 consumed -> emits cursor 0.
        XCTAssertEqual(
            sm.audioChunkConsumed(responseID: "req-1", textCursor: 0),
            [.sendPlaybackAck(RealtimePlaybackAck(responseID: "req-1", textCursor: 0))]
        )
        // Chunk 1 also has cursor 0 -> duplicate suppressed (monotonic).
        XCTAssertEqual(sm.audioChunkConsumed(responseID: "req-1", textCursor: 0), [])
        // Chunk 2 consumed (last pending chunk!) -> emits cursor 12 AND final cursor 24.
        XCTAssertEqual(
            sm.audioChunkConsumed(responseID: "req-1", textCursor: 12),
            [
                .sendPlaybackAck(RealtimePlaybackAck(responseID: "req-1", textCursor: 12)),
                .sendPlaybackAck(RealtimePlaybackAck(responseID: "req-1", textCursor: 24)),
            ]
        )

        // Final server ack marks response completed.
        _ = sm.receive(.responsePlaybackAck(
            responseID: "req-1", segmentID: 0, textCursor: 24, isFinal: true
        ))
        XCTAssertTrue(sm.completedResponseIDs.contains("req-1"))
        XCTAssertNil(sm.activeResponseID)
    }

    func testBargeInCancellationStopsPlaybackImmediatelyAndDropsAllStaleFrames() {
        var sm = makeAckEnabledStateMachine()
        let d0 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 12
        )
        let d1 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 12, textEndCursor: 24
        )
        _ = sm.receive(.audioDelta(d0))
        _ = sm.receive(.audioDelta(d1))
        _ = sm.receive(.audioDone(responseID: "req-1", segmentID: 0, textCursor: 24))
        _ = sm.audioChunkConsumed(responseID: "req-1", textCursor: 0)

        // Barge-in arrives while chunk 1 is still playing (even after audio.done!).
        let cancel = RealtimeCancellation(
            responseID: "req-1",
            segmentID: 0,
            reason: "barge_in",
            textCursor: 0,
            acknowledgedText: "",
            historyRecorded: true
        )
        let cancelActions = sm.receive(.responseCancelled(cancel))
        XCTAssertEqual(cancelActions, [
            .stopAudioImmediately(responseID: "req-1"),
            .forward(.responseCancelled(cancel)),
        ])

        // Late playback callback for cancelled response must emit ZERO acks.
        XCTAssertEqual(sm.audioChunkConsumed(responseID: "req-1", textCursor: 12), [])

        // Any stale server events for req-1 must be dropped.
        XCTAssertEqual(
            sm.receive(.responseTextDelta(responseID: "req-1", segmentID: 0, text: "late", textCursor: 30)),
            [.dropStale(responseID: "req-1")]
        )
        XCTAssertEqual(
            sm.receive(.audioDelta(d1)),
            [.dropStale(responseID: "req-1")]
        )
        XCTAssertEqual(
            sm.receive(.audioDone(responseID: "req-1", segmentID: 0, textCursor: 24)),
            [.dropStale(responseID: "req-1")]
        )
        XCTAssertEqual(
            sm.receive(.responseCompleted(responseID: "req-1", segmentID: 0, textCursor: 24)),
            [.dropStale(responseID: "req-1")]
        )
        XCTAssertEqual(
            sm.receive(.responsePlaybackAck(responseID: "req-1", segmentID: 0, textCursor: 24, isFinal: true)),
            [.dropStale(responseID: "req-1")]
        )

        // A new response req-2 starts cleanly after barge-in.
        let dNew = RealtimeAudioDelta(
            responseID: "req-2", segmentID: 1, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 5
        )
        XCTAssertEqual(sm.receive(.audioDelta(dNew)), [
            .beginAudioStream(responseID: "req-2"),
            .enqueueAudio(dNew),
            .forward(.audioDelta(dNew)),
        ])
    }

    func testNoPlaybackAckSentWhenServerDisabledPlaybackAck() {
        var sm = RealtimeStateMachine()
        _ = sm.receive(.sessionUpdated(RealtimeSessionSnapshot(
            inputAudioFormat: "pcm16",
            serverVADSilenceMs: 400,
            voice: "vivian",
            playbackAck: false,
            maxNewTokens: 128
        )))
        let d0 = RealtimeAudioDelta(
            responseID: "req-1", segmentID: 0, pcm16: pcm,
            sampleRate: 24_000, channels: 1, textCursor: 0, textEndCursor: 5
        )
        _ = sm.receive(.audioDelta(d0))
        _ = sm.receive(.audioDone(responseID: "req-1", segmentID: 0, textCursor: 5))
        XCTAssertEqual(sm.audioChunkConsumed(responseID: "req-1", textCursor: 0), [])
    }
}

// MARK: - ChatModel & RealtimeClient End-to-End Integration Tests (Mock Transport)

@MainActor
final class RealtimeChatModelIntegrationTests: XCTestCase {
    private let pcm = Data(repeating: 0x22, count: 4_800)

    private func waitUntil(
        ticks: Int = 400,
        _ condition: () -> Bool
    ) async {
        for _ in 0..<ticks where !condition() {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    func testFullDuplexTurnEmitsSessionUpdateMicAudioPlaybackAcksAndCompletes() async throws {
        let transport = MockRealtimeTransport()
        let capturer = MockAudioCapturer()
        let player = ControllableAudioStreamPlayer()
        let realtime = RealtimeClient(
            transportFactory: { transport },
            audioCapturer: capturer
        )
        let model = ChatModel(audioStream: player, realtime: realtime)
        model.endpoint = "http://127.0.0.1:43301/v1"
        model.model = "qwen3-asr"
        model.audioVoice = "vivian"

        model.startLiveVoice()
        XCTAssertTrue(model.isLiveVoiceActive)

        await waitUntil { !transport.sentMessages.isEmpty }
        XCTAssertEqual(
            transport.connectedURL?.absoluteString,
            "ws://127.0.0.1:43301/v1/realtime?intent=transcription&model=qwen3-asr"
        )
        // Mic capture must NOT start before transcription_session.updated.
        XCTAssertEqual(capturer.startCount, 0)

        let firstSent = try XCTUnwrap(transport.sentJSONObjects().first)
        XCTAssertEqual(firstSent["type"] as? String, "session.update")

        // Server accepts session with playback_ack=true.
        transport.pushServerJSON([
            "type": "transcription_session.updated",
            "session": [
                "input_audio_format": "pcm16",
                "turn_detection": ["type": "server_vad", "silence_ms": 400],
                "audio_output": ["voice": "vivian", "playback_ack": true],
            ],
        ])
        await waitUntil { model.liveVoiceState == .listening && capturer.startCount == 1 }
        XCTAssertEqual(model.liveVoiceState, .listening)

        // Emit one 100 ms 16 kHz PCM16 frame from mock mic.
        let micChunk = Data(repeating: 0x05, count: 3_200)
        capturer.emitFrame(micChunk)
        await waitUntil {
            transport.sentJSONObjects().contains { ($0["type"] as? String) == "input_audio_buffer.append" }
        }
        let appendMsg = try XCTUnwrap(
            transport.sentJSONObjects().first { ($0["type"] as? String) == "input_audio_buffer.append" }
        )
        XCTAssertEqual(appendMsg["audio"] as? String, micChunk.base64EncodedString())

        // Server VAD + transcription events.
        transport.pushServerJSON(["type": "input_audio_buffer.speech_started", "segment_id": 0])
        transport.pushServerJSON([
            "type": "transcription.segment", "event_index": 0, "segment_id": 0,
            "text": "Hi there", "is_final": false,
        ])
        await waitUntil { model.liveVoicePartialTranscript == "Hi there" }
        XCTAssertEqual(model.liveVoiceState, .userSpeaking)

        transport.pushServerJSON(["type": "input_audio_buffer.speech_stopped", "segment_id": 0, "audio_ms": 600])
        transport.pushServerJSON(["type": "transcription.completed", "segment_id": 0, "text": "Hi there."])

        // Assistant streams two sentences + audio.done.
        transport.pushServerJSON([
            "type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
            "text": "Sounds like ", "text_cursor": 12,
        ])
        transport.pushServerJSON([
            "type": "audio.delta", "response_id": "req-1", "segment_id": 0,
            "format": "pcm16", "sample_rate": 24_000, "channels": 1,
            "audio": pcm.base64EncodedString(), "text_cursor": 0, "text_end_cursor": 12,
        ])
        transport.pushServerJSON([
            "type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
            "text": "a good plan.", "text_cursor": 24,
        ])
        transport.pushServerJSON([
            "type": "audio.delta", "response_id": "req-1", "segment_id": 0,
            "format": "pcm16", "sample_rate": 24_000, "channels": 1,
            "audio": pcm.base64EncodedString(), "text_cursor": 12, "text_end_cursor": 24,
        ])
        transport.pushServerJSON([
            "type": "audio.done", "response_id": "req-1", "segment_id": 0, "text_cursor": 24,
        ])
        transport.pushServerJSON([
            "type": "response.completed", "response_id": "req-1", "segment_id": 0, "text_cursor": 24,
        ])

        await waitUntil { player.enqueuedChunks.count == 2 && player.finishCount == 1 }
        // Before playback consumes any chunk, ZERO playback_ack messages may be sent!
        XCTAssertTrue(transport.playbackAcks().isEmpty)

        // Chunk 0 plays -> ack cursor 0.
        player.playNextChunk()
        await waitUntil { transport.playbackAcks().count == 1 }
        XCTAssertEqual(transport.playbackAcks().map(\.textCursor), [0])

        // Chunk 1 plays (stream drains) -> ack cursors 12 and 24 monotonically!
        player.playNextChunk()
        await waitUntil { transport.playbackAcks().count == 3 }
        XCTAssertEqual(transport.playbackAcks().map(\.responseID), ["req-1", "req-1", "req-1"])
        XCTAssertEqual(transport.playbackAcks().map(\.textCursor), [0, 12, 24])

        // Server confirms final ack.
        transport.pushServerJSON([
            "type": "response.playback_ack", "response_id": "req-1",
            "segment_id": 0, "text_cursor": 24, "final": true,
        ])
        await waitUntil { model.liveVoiceLastAckCursor == 24 && model.liveVoiceState == .listening }
        XCTAssertEqual(model.turns.count, 2)
        XCTAssertEqual(model.turns[0].role, .user)
        XCTAssertEqual(model.turns[0].text, "Hi there.")
        XCTAssertEqual(model.turns[1].role, .assistant)
        XCTAssertEqual(model.turns[1].text, "Sounds like a good plan.")
        XCTAssertEqual(model.turns[1].state, .complete)

        model.stopLiveVoice()
        XCTAssertFalse(model.isLiveVoiceActive)
        XCTAssertEqual(capturer.stopCount, 1)
    }

    func testBargeInStopsAudioTrimsTurnToAcknowledgedPrefixAndDropsStaleFrames() async throws {
        let transport = MockRealtimeTransport()
        let capturer = MockAudioCapturer()
        let player = ControllableAudioStreamPlayer()
        let realtime = RealtimeClient(
            transportFactory: { transport },
            audioCapturer: capturer
        )
        let model = ChatModel(audioStream: player, realtime: realtime)
        model.endpoint = "http://127.0.0.1:43301/v1"
        model.model = "qwen3-asr"

        model.startLiveVoice()
        transport.pushServerJSON([
            "type": "transcription_session.updated",
            "session": [
                "input_audio_format": "pcm16",
                "turn_detection": ["type": "server_vad", "silence_ms": 400],
                "audio_output": ["voice": "vivian", "playback_ack": true],
            ],
        ])
        transport.pushServerJSON(["type": "transcription.completed", "segment_id": 0, "text": "Tell me a story."])
        transport.pushServerJSON([
            "type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
            "text": "Sounds like a long unplayed tail.", "text_cursor": 33,
        ])
        transport.pushServerJSON([
            "type": "audio.delta", "response_id": "req-1", "segment_id": 0,
            "format": "pcm16", "sample_rate": 24_000, "channels": 1,
            "audio": pcm.base64EncodedString(), "text_cursor": 0, "text_end_cursor": 12,
        ])
        transport.pushServerJSON([
            "type": "audio.delta", "response_id": "req-1", "segment_id": 0,
            "format": "pcm16", "sample_rate": 24_000, "channels": 1,
            "audio": pcm.base64EncodedString(), "text_cursor": 12, "text_end_cursor": 33,
        ])

        await waitUntil { player.enqueuedChunks.count == 2 }
        // Play only sentence 1 (chunk 0 -> ack 0) and start sentence 2 (chunk 1 -> ack 12).
        player.playNextChunk()
        await waitUntil { transport.playbackAcks().count == 1 }

        let stopsBeforeCancel = player.stopCount

        // User barges in! Server cancels req-1 with acknowledged prefix "Sounds like ".
        transport.pushServerJSON([
            "type": "response.cancelled",
            "response_id": "req-1",
            "segment_id": 0,
            "reason": "barge_in",
            "text_cursor": 12,
            "acknowledged_text": "Sounds like ",
            "history_recorded": true,
        ])
        // Stale frames for req-1 arrive right after cancellation.
        transport.pushServerJSON([
            "type": "audio.delta", "response_id": "req-1", "segment_id": 0,
            "format": "pcm16", "sample_rate": 24_000, "channels": 1,
            "audio": pcm.base64EncodedString(), "text_cursor": 20, "text_end_cursor": 33,
        ])
        transport.pushServerJSON([
            "type": "response.text.delta", "response_id": "req-1", "segment_id": 0,
            "text": " STALE", "text_cursor": 39,
        ])

        await waitUntil { realtime.droppedStaleEventCount == 2 }
        XCTAssertGreaterThan(player.stopCount, stopsBeforeCancel, "audioStream.stop() must fire on response.cancelled")
        XCTAssertFalse(player.isPlaying)
        XCTAssertEqual(player.enqueuedChunks.count, 2, "stale audio.delta must never be enqueued")
        XCTAssertEqual(model.turns.last?.role, .assistant)
        XCTAssertEqual(model.turns.last?.text, "Sounds like ")
        XCTAssertEqual(model.turns.last?.state, .stopped)

        // Follow-up turn req-2 works cleanly after barge-in.
        transport.pushServerJSON(["type": "transcription.completed", "segment_id": 1, "text": "Stop and summarize."])
        transport.pushServerJSON([
            "type": "response.text.delta", "response_id": "req-2", "segment_id": 1,
            "text": "Sure.", "text_cursor": 5,
        ])
        await waitUntil { model.turns.last?.text == "Sure." }
        XCTAssertEqual(model.turns.count, 4)

        model.stopLiveVoice()
    }

    func testUnsafeEndpointRejectsStartLiveVoiceWithoutActivatingSession() {
        let transport = MockRealtimeTransport()
        let capturer = MockAudioCapturer()
        let player = ControllableAudioStreamPlayer()
        let realtime = RealtimeClient(
            transportFactory: { transport },
            audioCapturer: capturer
        )
        let model = ChatModel(audioStream: player, realtime: realtime)
        model.endpoint = "http://example.com/v1"
        model.model = "qwen3-asr"

        model.startLiveVoice()
        XCTAssertFalse(model.isLiveVoiceActive)
        XCTAssertEqual(model.status, "Configuration error")
        XCTAssertNotNil(model.error)
        XCTAssertNil(transport.connectedURL)
    }
}
