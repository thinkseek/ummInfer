// SPDX-License-Identifier: Apache-2.0
// Failure and cancellation paths of ChatModel against scripted transports.
// Each test here failed before the corresponding fix.

import CryptoKit
import XCTest
@testable import ummmlx

private func shaHex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func line(_ frame: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: frame, options: [.sortedKeys])
    return "data: " + String(decoding: data, as: UTF8.self) + "\n\n"
}

private func deltaLine(seq: Int, _ text: String) -> String {
    line([
        "choices": [["delta": ["content": text], "finish_reason": NSNull()]],
        "unified": ["kind": "vlm_delta", "seq": seq, "session_id": "srv-1",
                    "stage": "vlm", "is_final": false, "text": text]
    ])
}

private func completeLine(seq: Int, _ text: String) -> String {
    line([
        "choices": [["delta": [:], "finish_reason": "stop"]],
        "unified": ["kind": "complete", "seq": seq, "session_id": "srv-1",
                    "stage": "vlm", "is_final": true, "text": text]
    ])
}

private func pcmLine(seq: Int, index: Int, _ data: Data) -> String {
    line([
        "choices": [["delta": [:], "finish_reason": NSNull()]],
        "unified": [
            "kind": "artifact", "seq": seq, "session_id": "srv-1",
            "stage": "tts", "is_final": false, "text": "",
            "artifact": [
                "artifact_id": "art-\(index)", "media_type": "audio",
                "mime_type": "audio/pcm", "encoding": "base64",
                "index": 0, "total": 1,
                "data": data.base64EncodedString(),
                "sha256": shaHex(data), "bytes": data.count,
                "format": "pcm16", "sample_rate": 24_000, "channels": 1
            ]
        ]
    ])
}

private let doneLine = "data: [DONE]\n\n"

/// Serves one scripted behavior per host on the 10.9.9.0/24 test range.
private final class ScriptedSSE: URLProtocol {
    private struct WeakBox: @unchecked Sendable {
        weak var value: ScriptedSSE?
    }
    enum Behavior {
        /// Send headers and bytes, then leave the connection open.
        case streamAndHold(String)
        /// Send headers and bytes, then break the connection.
        case streamThenFail(String)
        /// Answer with a non-2xx JSON body.
        case status(Int, String)
        /// Send the full SSE body and finish.
        case complete(String)
        /// Never answer: the request is still waiting for response headers.
        case hang
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var behaviors: [String: Behavior] = [:]
    nonisolated(unsafe) private static var stoppedHosts: [String] = []

    static func reset() {
        lock.lock()
        behaviors = [:]
        stoppedHosts = []
        lock.unlock()
    }

    static func set(_ host: String, _ behavior: Behavior) {
        lock.lock()
        behaviors[host] = behavior
        lock.unlock()
    }

    static func behavior(_ host: String) -> Behavior? {
        lock.lock()
        defer { lock.unlock() }
        return behaviors[host]
    }

    static func stopped() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return stoppedHosts
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.9.9.") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let host = url.host,
              let behavior = Self.behavior(host) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        func send(status: Int, contentType: String, body: String) {
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": contentType])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
        }
        switch behavior {
        case .streamAndHold(let body), .streamThenFail(let body):
            send(status: 200, contentType: "text/event-stream", body: body)
            if case .streamThenFail = behavior {
                let box = WeakBox(value: self)
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
                    guard let proto = box.value else { return }
                    proto.client?.urlProtocol(
                        proto, didFailWithError: URLError(.networkConnectionLost))
                }
            }
        case .status(let status, let body):
            send(status: status, contentType: "application/json", body: body)
            client?.urlProtocolDidFinishLoading(self)
        case .complete(let body):
            send(status: 200, contentType: "text/event-stream", body: body)
            client?.urlProtocolDidFinishLoading(self)
        case .hang:
            break
        }
    }

    override func stopLoading() {
        guard let host = request.url?.host else { return }
        Self.lock.lock()
        Self.stoppedHosts.append(host)
        Self.lock.unlock()
    }
}

@MainActor
private final class RecordingPlayer: AudioStreamPlaying {
    var onStateChange: ((Bool) -> Void)?
    private(set) var isPlaying = false {
        didSet {
            if oldValue != isPlaying { onStateChange?(isPlaying) }
        }
    }
    var stoppedCount = 0
    var chunks: [AudioStreamChunk] = []
    var finished = false

    func beginStream() { finished = false }
    func enqueue(_ chunk: AudioStreamChunk) {
        chunks.append(chunk)
        isPlaying = true
    }
    func finishStream() { finished = true }
    func stop() {
        stoppedCount += 1
        isPlaying = false
    }
}

/// A recognizer that yields the updates set before each run, then ends.
private final class StubTranscriber: SpeechTranscribing, @unchecked Sendable {
    var updates: [SpeechUpdate]
    var failure: Error?
    init(updates: [SpeechUpdate] = [], failure: Error? = nil) {
        self.updates = updates
        self.failure = failure
    }
    func transcribeMicrophone() -> AsyncThrowingStream<SpeechUpdate, Error> { stream() }
    func transcribeFile(at url: URL) -> AsyncThrowingStream<SpeechUpdate, Error> { stream() }
    func cancel() {}
    private func stream() -> AsyncThrowingStream<SpeechUpdate, Error> {
        let updates = updates
        let failure = failure
        return AsyncThrowingStream { continuation in
            for update in updates { continuation.yield(update) }
            if let failure {
                continuation.finish(throwing: failure)
            } else {
                continuation.finish()
            }
        }
    }
}

@MainActor
final class ChatModelRecoveryTests: XCTestCase {
    override func setUp() {
        super.setUp()
        ScriptedSSE.reset()
    }

    private func makeModel(player: RecordingPlayer = RecordingPlayer(),
                           speech: SpeechController? = nil) -> ChatModel {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ScriptedSSE.self]
        return ChatModel(
            speech: speech ?? SpeechController(transcriber: StubTranscriber()),
            client: ChatClient(session: URLSession(configuration: configuration)),
            audioStream: player)
    }

    private func waitIdle(_ model: ChatModel, ticks: Int = 600) async {
        for _ in 0..<ticks where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    // 1. Cancel while streamed PCM audio is playing.
    func testStopDuringAudioStreamHaltsPlaybackAndKeepsTheNextRequestUsable() async throws {
        let chunk = Data(repeating: 0x11, count: 4_800)
        ScriptedSSE.set("10.9.9.1", .streamAndHold(
            deltaLine(seq: 0, "Hi.") + pcmLine(seq: 1, index: 0, chunk)))
        ScriptedSSE.set("10.9.9.2", .complete(
            deltaLine(seq: 0, "o") + deltaLine(seq: 1, "k")
                + completeLine(seq: 2, "ok") + doneLine))

        let player = RecordingPlayer()
        let model = makeModel(player: player)
        model.endpoint = "http://10.9.9.1/v1"
        model.model = "m"
        model.audioOutput = true
        model.prompt = "say hi"
        model.send()

        for _ in 0..<600 where player.chunks.isEmpty {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(player.chunks.count, 1)
        XCTAssertTrue(model.isStreamingAudioPlaying)

        model.stop()

        XCTAssertGreaterThanOrEqual(player.stoppedCount, 1,
                                    "Stop must silence the live player synchronously")
        XCTAssertFalse(model.isStreamingAudioPlaying)
        XCTAssertNil(model.streamingAudioTurnID)

        await waitIdle(model)
        XCTAssertFalse(model.isRunning, "Stop must unblock the stalled stream")
        XCTAssertEqual(model.status, "Stopped")
        XCTAssertEqual(model.turns.last?.state, .stopped)
        XCTAssertEqual(model.turns.last?.text, "Hi.")

        // The next request in the same chat is unaffected.
        model.endpoint = "http://10.9.9.2/v1"
        model.prompt = "again"
        model.send()
        await waitIdle(model)
        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertEqual(model.turns.last?.text, "ok")
        XCTAssertEqual(model.turns.last?.state, .complete)
    }

    // 2. The connection breaks after a delta.
    func testMidStreamNetworkFailureMarksTheTurnFailedAndAllowsRetry() async {
        ScriptedSSE.set("10.9.9.3", .streamThenFail(deltaLine(seq: 0, "par")))
        ScriptedSSE.set("10.9.9.4", .complete(
            deltaLine(seq: 0, "ok") + completeLine(seq: 1, "ok") + doneLine))

        var sent: [[ChatMessage]] = []
        let model = makeModel()
        model.onRequest = { sent.append($0) }
        model.endpoint = "http://10.9.9.3/v1"
        model.model = "m"
        model.prompt = "hello"
        model.send()
        await waitIdle(model)

        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.status, "Error")
        XCTAssertNotNil(model.error)
        XCTAssertEqual(model.turns.last?.state, .failed)
        XCTAssertEqual(model.turns.last?.text, "par")

        model.endpoint = "http://10.9.9.4/v1"
        model.prompt = "retry"
        model.send()
        await waitIdle(model)

        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertNil(model.error)
        XCTAssertEqual(model.turns.last?.state, .complete)
        XCTAssertEqual(sent.last?.map(\.role), ["user"],
                       "a failed exchange must not be replayed as history")
    }

    // 2. A 5xx before the stream starts.
    func testServerErrorIsVisibleAndTheNextRequestSucceeds() async {
        ScriptedSSE.set("10.9.9.5", .status(503, #"{"error":"upstream unavailable"}"#))
        ScriptedSSE.set("10.9.9.6", .complete(
            deltaLine(seq: 0, "ok") + completeLine(seq: 1, "ok") + doneLine))

        let model = makeModel()
        model.endpoint = "http://10.9.9.5/v1"
        model.model = "m"
        model.prompt = "hello"
        model.send()
        await waitIdle(model)

        XCTAssertEqual(model.status, "Error")
        XCTAssertTrue(model.error?.contains("503") == true, model.error ?? "nil")
        XCTAssertEqual(model.turns.last?.role, .user,
                       "an empty failed assistant turn is removed")
        XCTAssertTrue(model.turns.allSatisfy { $0.state == .complete })

        model.endpoint = "http://10.9.9.6/v1"
        model.prompt = "retry"
        model.send()
        await waitIdle(model)
        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertEqual(model.turns.last?.text, "ok")
    }

    // 1. Stop while the request is still waiting for response headers. The
    // 600 s request timeout must not hold the composer for that long.
    func testStopWhileWaitingForHeadersReturnsToIdle() async {
        ScriptedSSE.set("10.9.9.7", .hang)
        let model = makeModel()
        model.endpoint = "http://10.9.9.7/v1"
        model.model = "m"
        model.prompt = "hello"
        model.send()
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(model.isRunning,
                      "status=\(model.status) error=\(model.error ?? "nil") "
                      + "turns=\(model.turns.count)")

        model.stop()
        await waitIdle(model, ticks: 400)
        XCTAssertFalse(model.isRunning,
                       "Stop must cancel a request that has no response yet")
        XCTAssertEqual(model.status, "Stopped")
        XCTAssertEqual(ScriptedSSE.stopped(), ["10.9.9.7"])
        XCTAssertEqual(model.turns.map(\.role), [.user],
                       "the empty assistant turn is removed, the prompt stays")
    }

    // 4. Permission denial during microphone input.
    func testMicrophonePermissionDenialLeavesListeningAndShowsTheError() async {
        let speech = SpeechController(
            transcriber: StubTranscriber(failure: SpeechError.authorizationDenied))
        let model = makeModel(speech: speech)
        model.startListening()

        for _ in 0..<600 where model.isListening {
            try? await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertFalse(model.isListening)
        XCTAssertEqual(model.status, "Speech error")
        XCTAssertTrue(model.error?.contains("not authorized") == true,
                      model.error ?? "nil")
        XCTAssertEqual(speech.phase, .failed)
    }

    // 4. A retry after a speech failure must not keep showing the old error.
    func testSuccessfulListeningClearsThePreviousSpeechError() async {
        let stub = StubTranscriber(failure: SpeechError.microphoneDenied)
        let model = makeModel(speech: SpeechController(transcriber: stub))
        model.startListening()
        for _ in 0..<600 where model.isListening {
            try? await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertNotNil(model.error)

        stub.failure = nil
        stub.updates = [.partial("hel"), .final("hello")]
        model.startListening()
        for _ in 0..<600 where model.isListening {
            try? await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertNil(model.error, "a new attempt must clear the previous failure")
        XCTAssertEqual(model.prompt, "hello")
        XCTAssertEqual(model.status, "Ready",
                       "a successful transcript clears the speech error status")
    }

    // 4. The recognizer stream ends without a final result (silence, or the
    // system ending the task). The microphone must not stay latched on.
    func testMicrophoneStreamEndingWithoutAFinalReleasesTheComposer() async {
        let speech = SpeechController(
            transcriber: StubTranscriber(updates: [.partial("hmm")]))
        let model = makeModel(speech: speech)
        model.startListening()
        XCTAssertTrue(model.isListening)

        for _ in 0..<600 where model.isListening {
            try? await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertFalse(model.isListening,
                       "a finished recognition stream must release the mic button")
        XCTAssertEqual(model.partialTranscript, "")
        XCTAssertNil(model.error)
    }

}
