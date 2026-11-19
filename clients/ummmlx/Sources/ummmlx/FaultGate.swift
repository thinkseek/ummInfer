// SPDX-License-Identifier: Apache-2.0
// Server-free fault regression for the macOS client.
//
// Run: ummmlx --automation --faults --output=/tmp/ummmlx-faults.json
//
// It drives the real ChatModel, ChatClient, URLSession and
// LiveAudioStreamPlayer through URLProtocol and scripted recognizers. It needs
// no ummInfer server, no socket, no microphone permission and no network.

import CryptoKit
import Foundation

@MainActor
enum FaultGate {
    static func run() async -> [String: Any] {
        FaultSSE.reset()
        var report = Report()

        await audioCancel(&report)
        await networkRecovery(&report)
        await serverErrorRecovery(&report)
        await speechFailure(&report)
        await speechEndsWithoutFinal(&report)
        await jitter(&report)

        return [
            "mode": "faults",
            "passed": report.passed,
            "transport": "URLProtocol; no server, no socket, no microphone",
            "checks": report.checks
        ]
    }

    private struct Report {
        var checks: [[String: Any]] = []
        var passed: Bool { checks.allSatisfy { $0["passed"] as? Bool == true } }
        mutating func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            checks.append(["name": name, "passed": ok, "detail": detail])
        }
    }

    // MARK: - Scenarios

    /// Stop while streamed PCM is playing, then a clean request in the same
    /// chat.
    private static func audioCancel(_ report: inout Report) async {
        let chunk = Data(repeating: 0x11, count: 4_800)
        FaultSSE.set("10.9.8.1", .hold(delta(0, "Hi.") + pcm(1, 0, chunk)))
        FaultSSE.set("10.9.8.2", .complete(
            delta(0, "o") + delta(1, "k") + complete(2, "ok") + done))

        let player = FaultPlayer()
        let model = makeModel(player: player)
        model.endpoint = "http://10.9.8.1/v1"
        model.model = "m"
        model.audioOutput = true
        model.prompt = "say hi"
        model.send()

        let arrived = await waitUntil { !player.chunks.isEmpty }
        model.stop()
        let silent = player.stoppedCount >= 1 && !model.isStreamingAudioPlaying
        let idle = await waitUntil { !model.isRunning }
        report.check(
            "audio_cancel",
            arrived && silent && idle && model.status == "Stopped"
                && model.turns.last?.state == .stopped,
            "chunk=\(arrived) stopped=\(player.stoppedCount) "
                + "playing=\(model.isStreamingAudioPlaying) idle=\(idle) "
                + "status=\(model.status)")

        model.endpoint = "http://10.9.8.2/v1"
        model.prompt = "again"
        model.send()
        let completed = await waitUntil { !model.isRunning }
        report.check(
            "audio_cancel_next_request",
            completed && model.status == "Completed · stop"
                && model.turns.last?.state == .complete,
            "status=\(model.status) text=\(model.turns.last?.text ?? "nil")")
    }

    /// The connection breaks after a delta; a later request succeeds and the
    /// failed exchange is not replayed.
    private static func networkRecovery(_ report: inout Report) async {
        FaultSSE.set("10.9.8.3", .drop(delta(0, "par")))
        FaultSSE.set("10.9.8.4", .complete(
            delta(0, "o") + delta(1, "k") + complete(2, "ok") + done))

        var sent: [[ChatMessage]] = []
        let model = makeModel()
        model.onRequest = { sent.append($0) }
        model.endpoint = "http://10.9.8.3/v1"
        model.model = "m"
        model.prompt = "hello"
        model.send()
        _ = await waitUntil { !model.isRunning }
        report.check(
            "network_drop_visible",
            model.status == "Error" && model.error != nil
                && model.turns.last?.state == .failed,
            "status=\(model.status) error=\(model.error ?? "nil") "
                + "state=\(String(describing: model.turns.last?.state))")

        model.endpoint = "http://10.9.8.4/v1"
        model.prompt = "retry"
        model.send()
        _ = await waitUntil { !model.isRunning }
        report.check(
            "network_drop_recovery",
            model.status == "Completed · stop" && model.error == nil
                && sent.last?.map(\.role) == ["user"],
            "status=\(model.status) history=\(sent.last?.map(\.role) ?? [])")
    }

    /// A 5xx before the stream starts is visible; the next request succeeds.
    private static func serverErrorRecovery(_ report: inout Report) async {
        FaultSSE.set("10.9.8.5",
                     .status(503, #"{"error":"upstream unavailable"}"#))
        FaultSSE.set("10.9.8.6", .complete(
            delta(0, "o") + delta(1, "k") + complete(2, "ok") + done))

        let model = makeModel()
        model.endpoint = "http://10.9.8.5/v1"
        model.model = "m"
        model.prompt = "hello"
        model.send()
        _ = await waitUntil { !model.isRunning }
        report.check(
            "server_error_visible",
            model.status == "Error"
                && (model.error?.contains("503") == true)
                && model.turns.allSatisfy { $0.state == .complete },
            "status=\(model.status) error=\(model.error ?? "nil")")

        model.endpoint = "http://10.9.8.6/v1"
        model.prompt = "retry"
        model.send()
        _ = await waitUntil { !model.isRunning }
        report.check(
            "server_error_recovery",
            model.status == "Completed · stop" && model.error == nil,
            "status=\(model.status)")
    }

    /// Permission denial on the microphone path.
    private static func speechFailure(_ report: inout Report) async {
        let speech = SpeechController(
            transcriber: FaultTranscriber(failure: SpeechError.authorizationDenied))
        let model = makeModel(speech: speech)
        model.startListening()
        _ = await waitUntil { !model.isListening }
        report.check(
            "asr_failure_visible",
            !model.isListening && model.status == "Speech error"
                && (model.error?.contains("not authorized") == true)
                && speech.phase == .failed,
            "listening=\(model.isListening) status=\(model.status) "
                + "error=\(model.error ?? "nil") phase=\(speech.phase)")
    }

    /// Silence or a system-ended task: the composer must be released.
    private static func speechEndsWithoutFinal(_ report: inout Report) async {
        let speech = SpeechController(
            transcriber: FaultTranscriber(updates: [.partial("hmm")]))
        let model = makeModel(speech: speech)
        model.startListening()
        let released = await waitUntil { !model.isListening }
        report.check(
            "asr_ends_without_final",
            released && model.partialTranscript.isEmpty && model.error == nil,
            "listening=\(model.isListening) partial=\(model.partialTranscript) "
                + "error=\(model.error ?? "nil")")
    }

    /// Burst delivery, a 2.5 s link, and a stream that follows a slow one.
    private static func jitter(_ report: inout Report) async {
        let player = LiveAudioStreamPlayer()
        defer { player.stop() }
        let chunk = AudioStreamChunk(
            pcm16: Data(repeating: 0, count: 4_800),
            sampleRate: 24_000, channels: 1)
        player.beginStream()
        player.enqueue(chunk)
        if player.outputDeviceUnavailable {
            report.check("jitter_burst", false, "no audio output device")
            report.check("jitter_slow", false, "no audio output device")
            report.check("jitter_reset", false, "no audio output device")
            return
        }
        for _ in 0..<4 { player.enqueue(chunk) }
        let burst = player.jitterTargetSeconds
        report.check("jitter_burst", abs(burst - 0.3) < 1e-9,
                     "target_s=\(burst)")

        try? await Task.sleep(nanoseconds: 2_500_000_000)
        player.enqueue(chunk)
        let slow = player.jitterTargetSeconds
        report.check("jitter_slow", abs(slow - 3.0) < 1e-9,
                     "target_s=\(slow)")

        player.beginStream()
        let reset = player.jitterTargetSeconds
        report.check("jitter_reset", abs(reset - 0.3) < 1e-9,
                     "target_s=\(reset)")
    }

    // MARK: - Fixtures

    private static func makeModel(
        player: FaultPlayer = FaultPlayer(),
        speech: SpeechController? = nil
    ) -> ChatModel {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FaultSSE.self]
        return ChatModel(
            speech: speech ?? SpeechController(transcriber: FaultTranscriber()),
            client: ChatClient(session: URLSession(configuration: configuration)),
            audioStream: player)
    }

    private static func waitUntil(timeout: Double = 5,
                                  _ condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() >= deadline { return false }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return true
    }
}

// MARK: - Scripted transport

private final class FaultSSE: URLProtocol, @unchecked Sendable {
    enum Behavior {
        case hold(String)
        case drop(String)
        case status(Int, String)
        case complete(String)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var behaviors: [String: Behavior] = [:]

    static func reset() {
        lock.lock()
        behaviors = [:]
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

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.9.8.") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

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
            client?.urlProtocol(self, didReceive: response,
                                cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
        }
        switch behavior {
        case .hold(let body), .drop(let body):
            send(status: 200, contentType: "text/event-stream", body: body)
            if case .drop = behavior {
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
                    [weak self] in
                    guard let self else { return }
                    self.client?.urlProtocol(
                        self, didFailWithError: URLError(.networkConnectionLost))
                }
            }
        case .status(let status, let body):
            send(status: status, contentType: "application/json", body: body)
            client?.urlProtocolDidFinishLoading(self)
        case .complete(let body):
            send(status: 200, contentType: "text/event-stream", body: body)
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

private func faultLine(_ frame: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: frame,
                                           options: [.sortedKeys])
    return "data: " + String(decoding: data, as: UTF8.self) + "\n\n"
}

private func delta(_ seq: Int, _ text: String) -> String {
    faultLine([
        "choices": [["delta": ["content": text], "finish_reason": NSNull()]],
        "unified": ["kind": "vlm_delta", "seq": seq, "session_id": "fault-1",
                    "stage": "vlm", "is_final": false, "text": text]
    ])
}

private func complete(_ seq: Int, _ text: String) -> String {
    faultLine([
        "choices": [["delta": [:], "finish_reason": "stop"]],
        "unified": ["kind": "complete", "seq": seq, "session_id": "fault-1",
                    "stage": "vlm", "is_final": true, "text": text]
    ])
}

private func pcm(_ seq: Int, _ index: Int, _ data: Data) -> String {
    let sha = SHA256.hash(data: data)
        .map { String(format: "%02x", $0) }.joined()
    return faultLine([
        "choices": [["delta": [:], "finish_reason": NSNull()]],
        "unified": [
            "kind": "artifact", "seq": seq, "session_id": "fault-1",
            "stage": "tts", "is_final": false, "text": "",
            "artifact": [
                "artifact_id": "art-\(index)", "media_type": "audio",
                "mime_type": "audio/pcm", "encoding": "base64",
                "index": 0, "total": 1,
                "data": data.base64EncodedString(),
                "sha256": sha, "bytes": data.count,
                "format": "pcm16", "sample_rate": 24_000, "channels": 1
            ]
        ]
    ])
}

private let done = "data: [DONE]\n\n"

// MARK: - Stubs

@MainActor
private final class FaultPlayer: AudioStreamPlaying {
    var onStateChange: ((Bool) -> Void)?
    private(set) var isPlaying = false {
        didSet {
            if oldValue != isPlaying { onStateChange?(isPlaying) }
        }
    }
    var stoppedCount = 0
    var chunks: [AudioStreamChunk] = []

    func beginStream() {}
    func enqueue(_ chunk: AudioStreamChunk) {
        chunks.append(chunk)
        isPlaying = true
    }
    func finishStream() {}
    func stop() {
        stoppedCount += 1
        isPlaying = false
    }
}

private final class FaultTranscriber: SpeechTranscribing, @unchecked Sendable {
    private let updates: [SpeechUpdate]
    private let failure: Error?
    init(updates: [SpeechUpdate] = [], failure: Error? = nil) {
        self.updates = updates
        self.failure = failure
    }
    func transcribeMicrophone() -> AsyncThrowingStream<SpeechUpdate, Error> {
        stream()
    }
    func transcribeFile(at url: URL) -> AsyncThrowingStream<SpeechUpdate, Error> {
        stream()
    }
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
