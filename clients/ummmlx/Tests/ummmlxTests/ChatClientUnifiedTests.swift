// SPDX-License-Identifier: Apache-2.0

import CryptoKit
import XCTest
@testable import ummmlx

private func sse(_ frames: [[String: Any]]) -> Data {
    var text = ""
    for frame in frames {
        let data = try! JSONSerialization.data(withJSONObject: frame)
        text += "data: " + String(decoding: data, as: UTF8.self) + "\n\n"
    }
    text += "data: [DONE]\n\n"
    return Data(text.utf8)
}

private func frame(
    kind: String, seq: Int, text: String = "", session: String = "srv-123456789",
    artifact: [String: Any]? = nil, finish: String? = nil, error: String? = nil
) -> [String: Any] {
    var delta: [String: Any] = [:]
    if kind == "vlm_delta" { delta["content"] = text }
    let reason: Any
    if let finish { reason = finish }
    else if kind == "complete" { reason = "stop" }
    else { reason = NSNull() }
    var unified: [String: Any] = [
        "kind": kind, "seq": seq, "session_id": session, "stage": "vlm",
        "is_final": kind == "complete", "text": text
    ]
    if let artifact { unified["artifact"] = artifact }
    if let error { unified["error"] = error }
    return [
        "choices": [["delta": delta, "finish_reason": reason]],
        "unified": unified
    ]
}

private let artifactBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3])
private let artifactSHA = SHA256.hash(data: artifactBytes)
    .map { String(format: "%02x", $0) }.joined()
private let pcmChunk0 = Data([0x01, 0x02, 0x03, 0x04])
private let pcmChunk0SHA = SHA256.hash(data: pcmChunk0)
    .map { String(format: "%02x", $0) }.joined()
private let pcmChunk1 = Data([0x05, 0x06, 0x07, 0x08])
private let pcmChunk1SHA = SHA256.hash(data: pcmChunk1)
    .map { String(format: "%02x", $0) }.joined()

private func pcmFrame(id: String, seq: Int, data: Data, sha: String) -> [String: Any] {
    frame(kind: "artifact", seq: seq, artifact: [
        "artifact_id": id, "media_type": "audio", "mime_type": "audio/pcm",
        "format": "pcm16", "sample_rate": 24_000, "channels": 1,
        "encoding": "base64", "index": 0, "total": 1,
        "data": data.base64EncodedString(),
        "sha256": sha, "bytes": data.count
    ])
}

private func artifactFrame(seq: Int) -> [String: Any] {
    frame(kind: "artifact", seq: seq, artifact: [
        "artifact_id": "art-0", "media_type": "image", "mime_type": "image/png",
        "encoding": "base64", "index": 0, "total": 1,
        "data": artifactBytes.base64EncodedString(),
        "sha256": artifactSHA, "bytes": artifactBytes.count
    ])
}

/// Canned unified streams, keyed by host.
private final class UnifiedSSEProtocol: URLProtocol {
    static let responses: [String: Data] = [
        "10.1.2.1": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Hel"),
            frame(kind: "vlm_delta", seq: 1, text: "lo"),
            frame(kind: "complete", seq: 2, text: "Hello")
        ]),
        "10.1.2.2": sse([
            frame(kind: "vlm_delta", seq: 0, text: "a"),
            frame(kind: "complete", seq: 2, text: "a")
        ]),
        "10.1.2.3": sse([
            artifactFrame(seq: 0),
            frame(kind: "complete", seq: 1, text: "")
        ]),
        "10.1.2.4": sse([
            frame(kind: "error", seq: 0, text: "", error: "asr failed")
        ]),
        // A scalar instead of an object must fail, not abort the process.
        "10.1.2.5": sse([["unified": "oops"]]),
        // Deltas and [DONE] but no terminal event.
        "10.1.2.6": sse([
            frame(kind: "vlm_delta", seq: 0, text: "partial")
        ]),
        // seq given as a JSON boolean decodes as 1 with a plain cast.
        "10.1.2.7": sse([
            frame(kind: "vlm_delta", seq: 0, text: "a"),
            ["unified": ["kind": "vlm_delta", "seq": true, "session_id": "s",
                         "stage": "vlm", "is_final": false, "text": "b"]]
        ]),
        // A second terminal event must fail.
        "10.1.2.8": sse([
            frame(kind: "complete", seq: 0, text: "a"),
            frame(kind: "complete", seq: 1, text: "b")
        ]),
        // Streamed TTS PCM16 chunks coalesce into one WAV artifact on complete.
        "10.1.2.10": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Hi."),
            pcmFrame(id: "art-0", seq: 1, data: pcmChunk0, sha: pcmChunk0SHA),
            pcmFrame(id: "art-1", seq: 2, data: pcmChunk1, sha: pcmChunk1SHA),
            frame(kind: "complete", seq: 3, text: "Hi.")
        ])
    ]

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.1.2.") == true
            && request.url?.host != "10.1.2.9"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let host = url.host,
              let data = Self.responses[host],
              let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "text/event-stream"]) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Rejects a multi-message replay the way the unified route does, then serves
/// one unified exchange. Records every request body so a test can assert the
/// retry shape.
private final class BodyCaptureProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var storage: [Data] = []

    static func reset() {
        lock.lock()
        storage = []
        lock.unlock()
    }

    static func bodies() -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    private static func append(_ data: Data) -> Int {
        lock.lock()
        defer { lock.unlock() }
        storage.append(data)
        return storage.count
    }

    private func body(_ request: URLRequest) -> Data {
        if let data = request.httpBody { return data }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "10.1.2.9"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let sent = body(request)
        let count = Self.append(sent)
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        if count == 1 {
            let payload = Data(#"{"error":{"message":"unified chat accepts at most one text part"}}"#.utf8)
            let response = HTTPURLResponse(
                url: url, statusCode: 400, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: payload)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        // The native-service deployment has no server-side history and rejects the flag.
        if count == 2 {
            let payload = Data(#"{"error":{"message":"session_history is not enabled on this server"}}"#.utf8)
            let response = HTTPURLResponse(
                url: url, statusCode: 400, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: payload)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let data = sse([
            frame(kind: "vlm_delta", seq: 0, text: "o"),
            frame(kind: "vlm_delta", seq: 1, text: "k"),
            frame(kind: "complete", seq: 2, text: "ok")
        ])
        let response = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class ChatClientUnifiedTests: XCTestCase {
    @MainActor
    private func makeClient() -> ChatClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UnifiedSSEProtocol.self]
        return ChatClient(session: URLSession(configuration: configuration))
    }

    @MainActor
    func testPrivatePlainHTTPHostsAreAccepted() throws {
        XCTAssertEqual(
            try ChatClient.chatURL(base: "http://172.16.0.2:43301/v1",
                                   allowedHTTPHosts: []).absoluteString,
            "http://172.16.0.2:43301/v1/chat/completions")
        XCTAssertEqual(
            try ChatClient.chatURL(base: "http://10.0.0.8/v1",
                                   allowedHTTPHosts: []).absoluteString,
            "http://10.0.0.8/v1/chat/completions")
        XCTAssertEqual(
            try ChatClient.chatURL(base: "http://192.168.1.9/v1",
                                   allowedHTTPHosts: []).absoluteString,
            "http://192.168.1.9/v1/chat/completions")
        XCTAssertThrowsError(
            try ChatClient.chatURL(base: "http://8.8.8.8/v1", allowedHTTPHosts: []))
        XCTAssertEqual(
            try ChatClient.chatURL(base: "http://8.8.8.8/v1",
                                   allowedHTTPHosts: ["8.8.8.8"]).absoluteString,
            "http://8.8.8.8/v1/chat/completions")
    }

    @MainActor
    func testTextAndSessionComeFromUnifiedEvents() async throws {
        let result = try await makeClient().stream(
            base: "http://10.1.2.1", model: "m", messages: []
        ) { _ in }
        XCTAssertEqual(result.text, "Hello")
        XCTAssertEqual(result.sessionID, "srv-123456789")
        XCTAssertEqual(result.finishReason, "stop")
        XCTAssertTrue(result.artifacts.isEmpty)
    }

    @MainActor
    func testSeqGapFailsTheRequest() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.2", model: "m", messages: []
            ) { _ in }
            XCTFail("A seq gap must fail")
        } catch {
            guard case .invalidResponse(let detail) = error as? ChatClientError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("seq 2 follows 0"), detail)
        }
    }

    @MainActor
    func testArtifactIsReassembledAndVerified() async throws {
        var delivered: [GeneratedArtifact] = []
        let result = try await makeClient().stream(
            base: "http://10.1.2.3", model: "m", messages: []
        ) { _ in } onArtifact: { delivered.append($0) }
        XCTAssertEqual(result.artifacts.count, 1)
        XCTAssertEqual(result.artifacts.first?.data, artifactBytes)
        XCTAssertEqual(result.artifacts.first?.sha256, artifactSHA)
        XCTAssertEqual(delivered.count, 1)
    }

    @MainActor
    func testUnifiedErrorTerminalIsReported() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.4", model: "m", messages: []
            ) { _ in }
            XCTFail("A server error terminal must fail")
        } catch {
            XCTAssertEqual(error as? ChatClientError, .unifiedError("asr failed"))
        }
    }

    @MainActor
    func testScalarUnifiedFailsInsteadOfCrashing() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.5", model: "m", messages: []
            ) { _ in }
            XCTFail("A scalar unified value must fail")
        } catch {
            XCTAssertEqual(error as? ChatClientError,
                           .invalidResponse("unified is not an object"))
        }
    }

    @MainActor
    func testMissingTerminalEventFailsTheRequest() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.6", model: "m", messages: []
            ) { _ in }
            XCTFail("A stream without a terminal event must fail")
        } catch {
            guard case .invalidResponse(let detail) = error as? ChatClientError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("without a terminal event"), detail)
        }
    }

    @MainActor
    func testBooleanSeqIsRejected() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.7", model: "m", messages: []
            ) { _ in }
            XCTFail("A boolean seq must fail")
        } catch {
            guard case .invalidResponse(let detail) = error as? ChatClientError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("seq is not an integer"), detail)
        }
    }

    @MainActor
    func testSecondTerminalEventFailsTheRequest() async {
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.8", model: "m", messages: []
            ) { _ in }
            XCTFail("Two terminal events must fail")
        } catch {
            guard case .invalidResponse(let detail) = error as? ChatClientError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("follows the terminal event"), detail)
        }
    }

    /// The capability fallback must put `session_history` on the wire, or the
    /// retried single message carries no context at all.
    @MainActor
    func testUnifiedHistoryFallbackSendsSessionHistory() async throws {
        BodyCaptureProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BodyCaptureProtocol.self]
        let client = ChatClient(session: URLSession(configuration: configuration))
        let result = try await client.stream(
            base: "http://10.1.2.9", model: "m",
            messages: [ChatMessage(role: "user", content: "a"),
                       ChatMessage(role: "assistant", content: "b"),
                       ChatMessage(role: "user", content: "c")],
            options: ChatStreamOptions(sessionID: "s1")
        ) { _ in }
        XCTAssertEqual(result.text, "ok")
        let bodies = BodyCaptureProtocol.bodies()
        XCTAssertEqual(bodies.count, 3,
                       "replay rejected, then session_history rejected, then accepted")
        let first = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: bodies[0]) as? [String: Any])
        XCTAssertEqual((first["messages"] as? [Any])?.count, 3)
        XCTAssertNil(first["session_history"])
        let retry = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: bodies[1]) as? [String: Any])
        XCTAssertEqual(retry["session_history"] as? Bool, true)
        let messages = try XCTUnwrap(retry["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages.first?["content"] as? String, "c")
        XCTAssertEqual(retry["session_id"] as? String, "s1")
        // Without server-side history the flag is dropped, not retried.
        let plain = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: bodies[2]) as? [String: Any])
        XCTAssertNil(plain["session_history"])
        XCTAssertEqual((plain["messages"] as? [Any])?.count, 1)
    }

    @MainActor
    func testImageSizeDefaultMatchesTheContract() {
        XCTAssertEqual(ImageOptions().size, "512x512")
    }

    @MainActor
    func testAudioRequestBodyStreamsAudioOnlyTurnsAndKeepsMultimodalOrdered() {
        var audioOnly = ChatStreamOptions()
        audioOnly.audio = AudioOptions(voice: "vivian")
        let audioBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "hi")],
            options: audioOnly)
        let audioObject = audioBody["audio"] as? [String: Any]
        XCTAssertEqual(audioObject?["voice"] as? String, "vivian")
        XCTAssertEqual(audioObject?["format"] as? String, "wav")
        XCTAssertEqual(audioObject?["stream"] as? Bool, true)

        var multimodal = ChatStreamOptions()
        multimodal.image = ImageOptions()
        multimodal.audio = AudioOptions(voice: "vivian")
        let mmBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "hi")],
            options: multimodal)
        let mmAudio = mmBody["audio"] as? [String: Any]
        XCTAssertNil(mmAudio?["stream"])
    }

    @MainActor
    func testStreamedAudioChunksFeedLiveCallbackAndCoalesceIntoOneWAVArtifact() async throws {
        var chunks: [AudioStreamChunk] = []
        var deliveredArtifacts: [GeneratedArtifact] = []
        let result = try await makeClient().stream(
            base: "http://10.1.2.10", model: "m", messages: [],
            options: ChatStreamOptions(audio: AudioOptions(voice: "vivian"))
        ) { _ in } onArtifact: {
            deliveredArtifacts.append($0)
        } onAudioChunk: {
            chunks.append($0)
        }
        XCTAssertEqual(chunks, [
            AudioStreamChunk(pcm16: pcmChunk0, sampleRate: 24_000, channels: 1),
            AudioStreamChunk(pcm16: pcmChunk1, sampleRate: 24_000, channels: 1)
        ])
        XCTAssertEqual(deliveredArtifacts.count, 1)
        XCTAssertEqual(result.artifacts.count, 1)
        let wav = try XCTUnwrap(result.artifacts.first)
        XCTAssertEqual(wav.id, "art-0")
        XCTAssertEqual(wav.mimeType, "audio/wav")
        XCTAssertEqual(wav.data.subdata(in: 44..<wav.data.count), pcmChunk0 + pcmChunk1)
    }

    @MainActor
    func testChatModelDrainsLiveAudioAfterSSECompleteAndStopsOnDemand() async throws {
        final class FakeAudioStreamPlayer: AudioStreamPlaying {
            var onStateChange: ((Bool) -> Void)?
            private(set) var isPlaying = false {
                didSet {
                    if oldValue != isPlaying { onStateChange?(isPlaying) }
                }
            }
            var beganCount = 0
            var finishedStream = false
            var stoppedCount = 0
            var chunks: [AudioStreamChunk] = []

            func beginStream() {
                beganCount += 1
                finishedStream = false
            }

            func enqueue(_ chunk: AudioStreamChunk) {
                chunks.append(chunk)
                isPlaying = true
            }

            func finishStream() {
                finishedStream = true
                // Simulate queued speaker buffers still draining after SSE ends.
            }

            func drainPlayback() {
                isPlaying = false
            }

            func stop() {
                stoppedCount += 1
                isPlaying = false
            }
        }

        let player = FakeAudioStreamPlayer()
        let model = ChatModel(client: makeClient(), audioStream: player)
        model.endpoint = "http://10.1.2.10/v1"
        model.model = "m"
        model.audioOutput = true
        model.prompt = "say hi"
        model.send()

        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertFalse(model.isRunning)
        XCTAssertTrue(player.finishedStream)
        XCTAssertEqual(player.chunks.count, 2)
        // Live playback stays active until queued buffers finish playing.
        XCTAssertTrue(model.isStreamingAudioPlaying)
        XCTAssertEqual(model.turns.last?.artifacts.count, 1)
        XCTAssertEqual(model.turns.last?.artifacts.first?.mimeType, "audio/wav")

        player.drainPlayback()
        XCTAssertFalse(model.isStreamingAudioPlaying)
        XCTAssertNil(model.streamingAudioTurnID)

        // Starting another streamed turn and calling stopStreamingAudio halts immediately.
        player.enqueue(AudioStreamChunk(pcm16: pcmChunk0, sampleRate: 24_000, channels: 1))
        XCTAssertTrue(model.isStreamingAudioPlaying)
        model.stopStreamingAudio()
        XCTAssertFalse(model.isStreamingAudioPlaying)
    }

    @MainActor
    func testRequestBodyEncodesWave9MediaAndSamplingOptions() {
        var options = ChatStreamOptions()
        options.sessionID = "sess-w9"
        options.temperature = 0.7
        options.topP = 0.9
        options.seed = -42
        options.image = ImageOptions(
            prompt: "lantern", size: "1024x1024", steps: 4, seed: 7,
            imageDataURL: "data:image/png;base64,AA==", outputFormat: "png")
        options.audio = AudioOptions(
            voice: "clone", format: "wav", stream: true,
            refAudio: "data:audio/wav;base64,UklGRg==")
        options.video = VideoOptions(
            prompt: "waves", size: "480x320", seconds: 2, fps: 16,
            numFrames: 13, steps: 15, guidanceScale: 5.0, seed: 9)

        let body = ChatClient.requestBody(
            model: "qwen3-native",
            messages: [ChatMessage(role: "user", content: "remix")],
            options: options)
        XCTAssertEqual(body["temperature"] as? Double, 0.7)
        XCTAssertEqual(body["top_p"] as? Double, 0.9)
        XCTAssertEqual(body["seed"] as? Int64, -42)
        XCTAssertEqual(body["modalities"] as? [String], ["text", "image", "audio", "video"])

        let img = body["image"] as? [String: Any]
        XCTAssertEqual(img?["steps"] as? Int, 4)
        XCTAssertEqual(img?["image"] as? String, "data:image/png;base64,AA==")

        let aud = body["audio"] as? [String: Any]
        XCTAssertEqual(aud?["ref_audio"] as? String, "data:audio/wav;base64,UklGRg==")
        XCTAssertNil(aud?["stream"], "audio.stream must stay off when video/image is active")

        let vid = body["video"] as? [String: Any]
        XCTAssertEqual(vid?["num_frames"] as? Int, 13)
        XCTAssertEqual(vid?["steps"] as? Int, 15)
        XCTAssertEqual(vid?["guidance_scale"] as? Double, 5.0)
        XCTAssertEqual(vid?["seed"] as? Int, 9)
    }
}
