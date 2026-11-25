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
    stage: String = "vlm", artifact: [String: Any]? = nil, finish: String? = nil,
    error: String? = nil, extra: [String: Any] = [:]
) -> [String: Any] {
    var delta: [String: Any] = [:]
    if kind == "vlm_delta" { delta["content"] = text }
    let reason: Any
    if let finish { reason = finish }
    else if kind == "complete" { reason = "stop" }
    else { reason = NSNull() }
    var unified: [String: Any] = [
        "kind": kind, "seq": seq, "session_id": session, "stage": stage,
        "is_final": kind == "complete", "text": text
    ]
    if let artifact { unified["artifact"] = artifact }
    if let error { unified["error"] = error }
    unified.merge(extra) { _, new in new }
    return [
        "choices": [["delta": delta, "finish_reason": reason]],
        "unified": unified
    ]
}

private let artifactBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3])
private let artifactSHA = SHA256.hash(data: artifactBytes)
    .map { String(format: "%02x", $0) }.joined()
private let videoArtifactBytes = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70])
private let videoArtifactSHA = SHA256.hash(data: videoArtifactBytes)
    .map { String(format: "%02x", $0) }.joined()
private let explicitAnchorSHA = "3a9d5bbe" + String(repeating: "4f12c890", count: 7)
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

private let slotPrompt = "a glowing red lantern over a misty lake"

private func visualSlotFrame(seq: Int, slotID: String = "slot-0",
                             index: Int = 0, kind: String = "image",
                             offset: Int = 0) -> [String: Any] {
    frame(kind: "visual_slot", seq: seq, stage: kind, extra: [
        "slot_id": slotID, "slot_index": index, "media_type": kind,
        "prompt": slotPrompt, "text_offset": offset
    ])
}

private func slotArtifactFrame(seq: Int, slotID: String = "slot-0",
                               index: Int = 0) -> [String: Any] {
    frame(kind: "artifact", seq: seq, stage: "image", artifact: [
        "artifact_id": "art-slot-0", "media_type": "image",
        "mime_type": "image/png", "slot_id": slotID, "slot_index": index,
        "encoding": "base64", "index": 0, "total": 1,
        "data": artifactBytes.base64EncodedString(),
        "sha256": artifactSHA, "bytes": artifactBytes.count
    ])
}

/// Serves image/video bytes for `POST /v1/visual/render` and records the body.
private final class VisualRenderProtocol: URLProtocol {
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
        request.url?.path.hasSuffix("/visual/render") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let sent = body(request)
        Self.lock.lock()
        Self.storage.append(sent)
        Self.lock.unlock()
        let response = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "image/png",
                           "X-UmmInfer-Kind": "image"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: artifactBytes)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Canned unified streams, keyed by host.
private final class UnifiedSSEProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var requestBodies: [Data] = []

    static func resetBodies() {
        lock.lock()
        requestBodies = []
        lock.unlock()
    }

    static func bodies() -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        return requestBodies
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
        ]),
        // Interleaved narrative: prose, a visual beat, then prose again.
        "10.1.2.11": sse([
            frame(kind: "vlm_delta", seq: 0, text: "One "),
            visualSlotFrame(seq: 1, offset: 4),
            frame(kind: "vlm_delta", seq: 2, text: "Two."),
            slotArtifactFrame(seq: 3),
            frame(kind: "complete", seq: 4, text: "One Two.")
        ]),
        // A slot leg fails: the error belongs to the card only.
        "10.1.2.13": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Story. "),
            visualSlotFrame(seq: 1),
            frame(kind: "leg_error", seq: 2, stage: "image", error: "boom",
                  extra: ["slot_id": "slot-0", "slot_index": 0,
                          "media_type": "image"]),
            frame(kind: "complete", seq: 3, text: "Story. ")
        ]),
        // Cross-slot visual continuity: prompt-anchored image followed by I2V-chained video.
        "10.1.2.14": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Act one. "),
            frame(kind: "visual_slot", seq: 1, stage: "image", extra: [
                "slot_id": "slot-0", "slot_index": 0, "media_type": "image",
                "prompt": "lantern in rain", "text_offset": 9,
                "continuity_mode": "prompt_anchor"
            ]),
            frame(kind: "vlm_delta", seq: 2, text: "Act two."),
            frame(kind: "visual_slot", seq: 3, stage: "video", extra: [
                "slot_id": "slot-1", "slot_index": 1, "media_type": "video",
                "prompt": "lantern swaying in wind", "text_offset": 17,
                "continuity_mode": "i2v_chain", "anchor_slot_index": 0
            ]),
            frame(kind: "complete", seq: 4, text: "Act one. Act two.")
        ]),
        // Wave 29 explicit source=slot-0 anchor: digest arrives on video artifact & complete.legs.
        "10.1.2.15": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Frame one. "),
            frame(kind: "visual_slot", seq: 1, stage: "image", extra: [
                "slot_id": "slot-0", "slot_index": 0, "media_type": "image",
                "prompt": "paper lantern keyframe", "text_offset": 11,
                "continuity_mode": "i2v_chain"
            ]),
            slotArtifactFrame(seq: 2, slotID: "slot-0", index: 0),
            frame(kind: "vlm_delta", seq: 3, text: "Frame two."),
            frame(kind: "visual_slot", seq: 4, stage: "video", extra: [
                "slot_id": "slot-1", "slot_index": 1, "media_type": "video",
                "prompt": "lantern drifting over river", "text_offset": 21,
                "continuity_mode": "i2v_chain", "anchor_slot_index": 0,
                "anchor_explicit": true
            ]),
            frame(kind: "artifact", seq: 5, stage: "video", artifact: [
                "artifact_id": "art-slot-1", "media_type": "video",
                "mime_type": "video/mp4", "slot_id": "slot-1", "slot_index": 1,
                "anchor_slot_index": 0, "anchor_explicit": true,
                "anchor_sha256": explicitAnchorSHA,
                "encoding": "base64", "index": 0, "total": 1,
                "data": videoArtifactBytes.base64EncodedString(),
                "sha256": videoArtifactSHA, "bytes": videoArtifactBytes.count
            ]),
            frame(kind: "complete", seq: 6, text: "Frame one. Frame two.", extra: [
                "legs": [
                    ["stage": "image", "state": "completed", "slot_id": "slot-0", "slot_index": 0],
                    [
                        "stage": "video", "state": "completed",
                        "slot_id": "slot-1", "slot_index": 1,
                        "anchor_slot_index": 0, "anchor_explicit": true,
                        "anchor_sha256": explicitAnchorSHA
                    ]
                ]
            ])
        ]),
        // Wave 29 explicit source=slot-1 self-reference failure: leg_error lands on slot card.
        "10.1.2.16": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Bad link. "),
            frame(kind: "visual_slot", seq: 1, stage: "video", extra: [
                "slot_id": "slot-1", "slot_index": 1, "media_type": "video",
                "prompt": "broken reference", "text_offset": 10,
                "continuity_mode": "i2v_chain", "anchor_explicit": true
            ]),
            frame(kind: "leg_error", seq: 2, stage: "video",
                  error: "source slot slot-1 is not an earlier slot",
                  extra: [
                    "slot_id": "slot-1", "slot_index": 1, "media_type": "video",
                    "anchor_explicit": true
                  ]),
            frame(kind: "complete", seq: 3, text: "Bad link. ")
        ]),
        // Explicit leg_error without a prior visual_slot synthesizes a failed slot card.
        "10.1.2.17": sse([
            frame(kind: "vlm_delta", seq: 0, text: "Early error. "),
            frame(kind: "leg_error", seq: 1, stage: "video",
                  error: "source slot slot-1 is not an earlier slot",
                  extra: [
                    "slot_id": "slot-1", "slot_index": 1, "media_type": "video",
                    "anchor_explicit": true
                  ]),
            frame(kind: "complete", seq: 2, text: "Early error. ")
        ])
    ]

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.1.2.") == true
            && request.url?.host != "10.1.2.9"
            && request.url?.path.hasSuffix("/visual/render") != true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let sent = body(request)
        Self.lock.lock()
        Self.requestBodies.append(sent)
        Self.lock.unlock()
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
        configuration.protocolClasses = [UnifiedSSEProtocol.self,
                                         VisualRenderProtocol.self]
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
        XCTAssertEqual(mmAudio?["stream"] as? Bool, true,
                       "global voice streams alongside image and video legs")
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
        XCTAssertEqual(aud?["stream"] as? Bool, true,
                       "global voice streams alongside image and video legs")

        let vid = body["video"] as? [String: Any]
        XCTAssertEqual(vid?["num_frames"] as? Int, 13)
        XCTAssertEqual(vid?["steps"] as? Int, 15)
        XCTAssertEqual(vid?["guidance_scale"] as? Double, 5.0)
        XCTAssertEqual(vid?["seed"] as? Int, 9)
    }

    @MainActor
    func testRequestBodyEncodesNarrativeWithGlobalStreamingVoice() {
        var options = ChatStreamOptions()
        options.image = ImageOptions()
        options.video = VideoOptions()
        options.audio = AudioOptions(voice: "vivian", stream: true)
        options.narrative = NarrativeOptions(enabled: true, visualMode: "auto",
                                             maxSlots: 3)
        let body = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: options)
        XCTAssertEqual(body["modalities"] as? [String],
                       ["text", "image", "audio", "video"])
        let audio = body["audio"] as? [String: Any]
        XCTAssertEqual(audio?["stream"] as? Bool, true)
        let narrative = body["narrative"] as? [String: Any]
        XCTAssertEqual(narrative?["enabled"] as? Bool, true)
        XCTAssertEqual(narrative?["visual_mode"] as? String, "auto")
        XCTAssertEqual(narrative?["max_slots"] as? Int, 3)
        XCTAssertEqual(narrative?["allow_text_only"] as? Bool, true)

        // Absent unless the caller opts in: backwards compatible.
        options.narrative = nil
        let plain = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "hi")],
            options: options)
        XCTAssertNil(plain["narrative"])
    }

    @MainActor
    func testRequestBodyEncodesNarrativeAllowTextOnlyFlag() {
        let defaultNarrative = NarrativeOptions()
        XCTAssertTrue(defaultNarrative.allowTextOnly)

        var defaultOpts = ChatStreamOptions()
        defaultOpts.image = ImageOptions()
        defaultOpts.narrative = defaultNarrative
        let defaultBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "2+2?")],
            options: defaultOpts)
        let defaultNav = defaultBody["narrative"] as? [String: Any]
        XCTAssertEqual(defaultNav?["allow_text_only"] as? Bool, true)

        var optOutOpts = ChatStreamOptions()
        optOutOpts.image = ImageOptions()
        optOutOpts.narrative = NarrativeOptions(
            enabled: true, visualMode: "auto", maxSlots: 2, allowTextOnly: false)
        let optOutBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "2+2?")],
            options: optOutOpts)
        let optOutNav = optOutBody["narrative"] as? [String: Any]
        XCTAssertEqual(optOutNav?["enabled"] as? Bool, true)
        XCTAssertEqual(optOutNav?["visual_mode"] as? String, "auto")
        XCTAssertEqual(optOutNav?["max_slots"] as? Int, 2)
        XCTAssertEqual(optOutNav?["allow_text_only"] as? Bool, false)

        var optInOpts = ChatStreamOptions()
        optInOpts.image = ImageOptions()
        optInOpts.narrative = NarrativeOptions(
            enabled: true, visualMode: "auto", maxSlots: 4, allowTextOnly: true)
        let optInBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: optInOpts)
        let optInNav = optInBody["narrative"] as? [String: Any]
        XCTAssertEqual(optInNav?["max_slots"] as? Int, 4)
        XCTAssertEqual(optInNav?["allow_text_only"] as? Bool, true)

        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        XCTAssertTrue(model.narrativeAllowTextOnly)
    }

    @MainActor
    func testRequestBodyEncodesVideoURLContentPart() {
        let vidURL = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
        let body = ChatClient.requestBody(
            model: "qwen3-native",
            messages: [ChatMessage(role: "user", content: "describe motion",
                                   videoDataURL: vidURL)],
            options: ChatStreamOptions())
        let msgs = body["messages"] as? [[String: Any]]
        let parts = msgs?.first?["content"] as? [[String: Any]]
        XCTAssertEqual(parts?.count, 2)
        XCTAssertEqual(parts?[0]["type"] as? String, "text")
        XCTAssertEqual(parts?[0]["text"] as? String, "describe motion")
        XCTAssertEqual(parts?[1]["type"] as? String, "video_url")
        let videoObj = parts?[1]["video_url"] as? [String: Any]
        XCTAssertEqual(videoObj?["url"] as? String, vidURL)
    }

    @MainActor
    func testVideoFrameSelectionDefaultsAndTemporalWireSchema() throws {
        let videoURL = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
        let message = ChatMessage(role: "user", content: "describe motion",
                                  videoDataURL: videoURL)
        let defaultOptions = ChatStreamOptions()
        XCTAssertNil(defaultOptions.videoFrameSelection)
        let defaultBody = ChatClient.requestBody(
            model: "qwen3-native", messages: [message], options: defaultOptions)
        let defaultMessages = try XCTUnwrap(defaultBody["messages"] as? [[String: Any]])
        let defaultParts = try XCTUnwrap(defaultMessages.first?["content"] as? [[String: Any]])
        let defaultVideo = try XCTUnwrap(defaultParts[1]["video_url"] as? [String: Any])
        XCTAssertEqual(defaultVideo.count, 1)
        XCTAssertEqual(defaultVideo["url"] as? String, videoURL)
        XCTAssertNil(defaultVideo["frame_selection"],
                     "Uniform default must preserve the legacy video_url object")
        XCTAssertNil(try VideoFrameSelectionOptions.parse(
            policy: "uniform", motionText: "ignored", sceneText: ""))

        let temporal = try XCTUnwrap(VideoFrameSelectionOptions.parse(
            policy: "temporal", motionText: "0.02", sceneText: "0.25"))
        var options = ChatStreamOptions()
        options.videoFrameSelection = temporal
        options.video = VideoOptions(size: "704x480", numFrames: 29, steps: 25)
        options.videoTeaCacheThresh = 0.05
        let body = ChatClient.requestBody(
            model: "qwen3-native", messages: [message], options: options)
        let messages = try XCTUnwrap(body["messages"] as? [[String: Any]])
        let parts = try XCTUnwrap(messages.first?["content"] as? [[String: Any]])
        let video = try XCTUnwrap(parts[1]["video_url"] as? [String: Any])
        XCTAssertEqual(video["url"] as? String, videoURL)
        let selection = try XCTUnwrap(video["frame_selection"] as? [String: Any])
        XCTAssertEqual(selection.count, 3)
        XCTAssertEqual(selection["policy"] as? String, "temporal")
        XCTAssertEqual(try XCTUnwrap(selection["motion_threshold"] as? Double),
                       0.02, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(selection["scene_threshold"] as? Double),
                       0.25, accuracy: 1e-12)

        // Input temporal selection and generated-video Fast Video coexist in
        // their separate accepted locations; neither rewrites the other.
        let generated = try XCTUnwrap(body["video"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(generated["teacache_thresh"] as? Double),
                       0.05, accuracy: 1e-12)
        XCTAssertEqual(generated["num_frames"] as? Int, 29)
        XCTAssertEqual(generated["steps"] as? Int, 25)
        XCTAssertEqual(body["modalities"] as? [String], ["text", "video"])
    }

    @MainActor
    func testVideoFrameSelectionThresholdBoundariesAndInvalidInput() throws {
        let boundaries = try XCTUnwrap(VideoFrameSelectionOptions.parse(
            policy: "temporal", motionText: "0", sceneText: "1"))
        XCTAssertEqual(boundaries.motionThreshold, 0)
        XCTAssertEqual(boundaries.sceneThreshold, 1)
        try boundaries.validate()

        let invalid = ["", " ", "abc", "0.2x", "NaN", "inf", "-inf",
                       "-0.0001", "1.0001", "-1", "2"]
        for value in invalid {
            XCTAssertThrowsError(try VideoFrameSelectionOptions.parse(
                policy: "temporal", motionText: value, sceneText: "0.25"),
                "invalid motion threshold was accepted: \(value)")
            XCTAssertThrowsError(try VideoFrameSelectionOptions.parse(
                policy: "temporal", motionText: "0.02", sceneText: value),
                "invalid scene threshold was accepted: \(value)")
        }
        XCTAssertThrowsError(try VideoFrameSelectionOptions.parse(
            policy: "scene-only", motionText: "0.02", sceneText: "0.25"))
        XCTAssertThrowsError(try VideoFrameSelectionOptions(
            motionThreshold: Double.nan, sceneThreshold: 0.25).validate())
        XCTAssertThrowsError(try VideoFrameSelectionOptions(
            motionThreshold: 0.02, sceneThreshold: Double.infinity).validate())
    }

    @MainActor
    func testClientRejectsNonFiniteTemporalSelectionBeforeTransport() async {
        UnifiedSSEProtocol.resetBodies()
        var options = ChatStreamOptions()
        options.videoFrameSelection = VideoFrameSelectionOptions(
            motionThreshold: 0.02, sceneThreshold: Double.nan)
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.1/v1", model: "m",
                messages: [ChatMessage(role: "user", content: "video",
                                       videoDataURL: "data:video/mp4;base64,AAAA")],
                options: options) { _ in }
            XCTFail("a non-finite threshold must fail before transport")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("[0,1]"))
        }
        XCTAssertTrue(UnifiedSSEProtocol.bodies().isEmpty)
    }

    @MainActor
    func testChatModelTemporalValidationResetAndReconnect() async throws {
        UnifiedSSEProtocol.resetBodies()
        let videoURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        try Data([0, 1, 2, 3]).write(to: videoURL)
        defer { try? FileManager.default.removeItem(at: videoURL) }

        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.1/v1"
        model.model = "m"
        model.videoFramePolicy = "temporal"
        model.videoMotionThresholdText = "-0.01"
        model.videoSceneThresholdText = "0.25"
        model.attachedVideoURL = videoURL
        model.prompt = "describe motion"
        model.send()
        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.status, "Configuration error")
        XCTAssertTrue(model.error?.contains("[0,1]") == true)
        XCTAssertEqual(model.attachedVideoURL, videoURL,
                       "validation failure must not discard the attachment")
        XCTAssertTrue(model.turns.isEmpty)
        XCTAssertTrue(UnifiedSSEProtocol.bodies().isEmpty,
                      "invalid temporal values must not reach URLSession")

        // Correcting the values and reconnecting uses fresh options and accepts
        // both inclusive boundaries on the exact video_url wire object.
        model.videoMotionThresholdText = "0"
        model.videoSceneThresholdText = "1"
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertEqual(UnifiedSSEProtocol.bodies().count, 1)
        let temporalBody = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: UnifiedSSEProtocol.bodies()[0]) as? [String: Any])
        let temporalMessages = try XCTUnwrap(temporalBody["messages"] as? [[String: Any]])
        let temporalParts = try XCTUnwrap(
            temporalMessages.first?["content"] as? [[String: Any]])
        let temporalVideo = try XCTUnwrap(temporalParts[1]["video_url"] as? [String: Any])
        let temporalSelection = try XCTUnwrap(
            temporalVideo["frame_selection"] as? [String: Any])
        XCTAssertEqual(temporalSelection["policy"] as? String, "temporal")
        XCTAssertEqual(temporalSelection["motion_threshold"] as? Double, 0)
        XCTAssertEqual(temporalSelection["scene_threshold"] as? Double, 1)

        // New chat + reset to Uniform restores legacy bytes. Fast Video still
        // controls only generated video and malformed disabled fields are inert.
        model.newChat()
        model.videoFramePolicy = "uniform"
        model.videoMotionThresholdText = "not-a-number"
        model.videoSceneThresholdText = ""
        model.videoOutput = true
        model.fastVideoMode = true
        model.attachedVideoURL = videoURL
        model.prompt = "describe and animate"
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertEqual(UnifiedSSEProtocol.bodies().count, 2)
        let resetBody = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: UnifiedSSEProtocol.bodies()[1]) as? [String: Any])
        let resetMessages = try XCTUnwrap(resetBody["messages"] as? [[String: Any]])
        let resetParts = try XCTUnwrap(resetMessages.first?["content"] as? [[String: Any]])
        let resetVideo = try XCTUnwrap(resetParts[1]["video_url"] as? [String: Any])
        XCTAssertEqual(resetVideo.count, 1)
        XCTAssertNil(resetVideo["frame_selection"])
        let generated = try XCTUnwrap(resetBody["video"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(generated["teacache_thresh"] as? Double),
                       0.05, accuracy: 1e-12)
        XCTAssertEqual(generated["num_frames"] as? Int, 29)
        XCTAssertEqual(generated["steps"] as? Int, 25)
    }

    @MainActor
    func testVisualRenderURLKeepsTheChatURLSecurityPolicyAndPath() throws {
        XCTAssertEqual(
            try ChatClient.visualRenderURL(
                base: "http://127.0.0.1:43301/v1", allowedHTTPHosts: []
            ).absoluteString,
            "http://127.0.0.1:43301/v1/visual/render")
        XCTAssertEqual(
            try ChatClient.visualRenderURL(
                base: "http://10.0.0.8", allowedHTTPHosts: []
            ).absoluteString,
            "http://10.0.0.8/v1/visual/render")
        XCTAssertEqual(
            try ChatClient.visualRenderURL(
                base: "https://example.com/custom/", allowedHTTPHosts: []
            ).absoluteString,
            "https://example.com/custom/visual/render")
        XCTAssertEqual(
            try ChatClient.visualRenderURL(
                base: "http://8.8.8.8/v1", allowedHTTPHosts: ["8.8.8.8"]
            ).absoluteString,
            "http://8.8.8.8/v1/visual/render")
        XCTAssertThrowsError(try ChatClient.visualRenderURL(
            base: "http://example.com/v1", allowedHTTPHosts: []))
        XCTAssertThrowsError(try ChatClient.visualRenderURL(
            base: "http://127.0.0.1/v1?x=1", allowedHTTPHosts: []))
    }

    @MainActor
    func testChatModelInterleavesVisualSlotsAndAttributesArtifacts() async throws {
        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.11/v1"
        model.model = "m"
        model.imageOutput = true
        model.narrativeMode = true
        model.prompt = "story"
        model.send()

        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertFalse(model.isRunning)
        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
        XCTAssertEqual(turn.text, "One Two.")
        XCTAssertEqual(turn.blocks.count, 3)
        guard case .text(_, let first) = turn.blocks[0] else {
            return XCTFail("the first block must be prose")
        }
        XCTAssertEqual(first, "One ")
        guard case .visual(let slot) = turn.blocks[1] else {
            return XCTFail("the visual beat must sit between the prose blocks")
        }
        XCTAssertEqual(slot.slotID, "slot-0")
        XCTAssertEqual(slot.slotIndex, 0)
        XCTAssertEqual(slot.prompt, slotPrompt)
        XCTAssertEqual(slot.activeKind, "image")
        XCTAssertEqual(slot.imageArtifact?.data, artifactBytes)
        XCTAssertNil(slot.error)
        XCTAssertFalse(slot.isLoading)
        guard case .text(_, let last) = turn.blocks[2] else {
            return XCTFail("prose after the beat must start a new block")
        }
        XCTAssertEqual(last, "Two.")
        // Legacy fields stay populated for replay and recovery.
        XCTAssertEqual(turn.artifacts.count, 1)
        XCTAssertEqual(turn.artifacts.first?.slotID, "slot-0")
        XCTAssertEqual(turn.artifacts.first?.slotIndex, 0)
    }

    @MainActor
    func testSlotLegErrorLandsOnTheCardOnce() async throws {
        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.13/v1"
        model.model = "m"
        model.imageOutput = true
        model.prompt = "story"
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
        XCTAssertTrue(turn.legErrors.isEmpty,
                      "a slot failure must not duplicate at turn level")
        XCTAssertEqual(turn.blocks.count, 2)
        guard case .visual(let slot) = turn.blocks[1] else {
            return XCTFail("the visual beat must follow the prose block")
        }
        XCTAssertEqual(slot.error, "boom")
        XCTAssertFalse(slot.isLoading)
        XCTAssertNil(slot.imageArtifact)
    }

    @MainActor
    func testChatModelSwapsVisualSlotInPlaceAndReusesTheCachedKind() async throws {
        VisualRenderProtocol.reset()
        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.11/v1"
        model.model = "m"
        model.imageOutput = true
        model.narrativeMode = true
        model.prompt = "story"
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })

        model.swapVisualSlot(turnID: turn.id, slotID: "slot-0", toKind: "video")
        var swapped: VisualSlotState?
        for _ in 0..<200 {
            swapped = model.turns
                .first(where: { $0.id == turn.id })?.blocks
                .compactMap(\.visualSlot)
                .first(where: { $0.slotID == "slot-0" })
            if swapped?.videoArtifact != nil { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(swapped?.activeKind, "video")
        XCTAssertEqual(swapped?.videoArtifact?.data, artifactBytes)
        XCTAssertFalse(swapped?.isLoading ?? true)
        XCTAssertNil(swapped?.error)
        XCTAssertEqual(VisualRenderProtocol.bodies().count, 1)
        let body = try JSONSerialization.jsonObject(
            with: try XCTUnwrap(VisualRenderProtocol.bodies().first))
            as? [String: Any]
        XCTAssertEqual(body?["kind"] as? String, "video")
        XCTAssertEqual(body?["prompt"] as? String, slotPrompt)
        XCTAssertEqual(body?["size"] as? String, "704x480")
        XCTAssertEqual(body?["image"] as? String,
                       "data:image/png;base64," + artifactBytes.base64EncodedString(),
                       "a video swap must animate the slot's image as the first frame")

        // The image is still cached: swapping back makes no network call.
        model.swapVisualSlot(turnID: turn.id, slotID: "slot-0", toKind: "image")
        let back = model.turns
            .first(where: { $0.id == turn.id })?.blocks
            .compactMap(\.visualSlot)
            .first(where: { $0.slotID == "slot-0" })
        XCTAssertEqual(back?.activeKind, "image")
        XCTAssertEqual(back?.imageArtifact?.data, artifactBytes)
        XCTAssertFalse(back?.isLoading ?? true)
        XCTAssertEqual(VisualRenderProtocol.bodies().count, 1,
                       "a cached kind must not hit the network")
    }

    @MainActor
    func testFastVideoTeaCacheOptionOmittedByDefaultAndSentWhenOptedIn() async throws {
        // Default ChatModel.fastVideoMode must be false (lossless 0.0 stage default).
        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        XCTAssertFalse(model.fastVideoMode)

        // Chat requestBody omits teacache_thresh when videoTeaCacheThresh is nil.
        var defaultOptions = ChatStreamOptions()
        defaultOptions.video = VideoOptions(size: "704x480")
        let defaultBody = ChatClient.requestBody(
            model: "m",
            messages: [ChatMessage(role: "user", content: "waves")],
            options: defaultOptions)
        let defaultVideo = try XCTUnwrap(defaultBody["video"] as? [String: Any])
        XCTAssertNil(defaultVideo["teacache_thresh"],
                     "teacache_thresh must be omitted unless opted in")

        // Chat requestBody includes teacache_thresh: 0.05 when videoTeaCacheThresh is set.
        var fastOptions = ChatStreamOptions()
        fastOptions.video = VideoOptions(size: "704x480")
        fastOptions.videoTeaCacheThresh = 0.05
        let fastBody = ChatClient.requestBody(
            model: "m",
            messages: [ChatMessage(role: "user", content: "waves")],
            options: fastOptions)
        let fastVideo = try XCTUnwrap(fastBody["video"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(fastVideo["teacache_thresh"] as? Double),
                       0.05, accuracy: 1e-9)

        // /v1/visual/render omits teacache_thresh when fastVideoMode == false,
        // sends 0.05 when fastVideoMode == true on video swap, and never on image.
        VisualRenderProtocol.reset()
        let client = makeClient()
        _ = try await client.renderVisual(
            base: "http://10.1.2.11/v1", kind: "video", prompt: "calm sea",
            size: "704x480")
        _ = try await client.renderVisual(
            base: "http://10.1.2.11/v1", kind: "video", prompt: "fast sea",
            size: "704x480", teaCacheThresh: 0.05)
        _ = try await client.renderVisual(
            base: "http://10.1.2.11/v1", kind: "image", prompt: "still sea",
            size: "512x512", teaCacheThresh: 0.05)
        let vrBodies = VisualRenderProtocol.bodies()
        XCTAssertEqual(vrBodies.count, 3)
        let vrDefault = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: vrBodies[0]) as? [String: Any])
        XCTAssertNil(vrDefault["teacache_thresh"])
        let vrFast = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: vrBodies[1]) as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(vrFast["teacache_thresh"] as? Double),
                       0.05, accuracy: 1e-9)
        let vrImage = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: vrBodies[2]) as? [String: Any])
        XCTAssertNil(vrImage["teacache_thresh"],
                     "image render must never carry video-only teacache_thresh")

        // End-to-end ChatModel swapVisualSlot honours fastVideoMode.
        VisualRenderProtocol.reset()
        model.endpoint = "http://10.1.2.11/v1"
        model.model = "m"
        model.imageOutput = true
        model.narrativeMode = true
        model.fastVideoMode = true
        model.prompt = "story"
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
        model.swapVisualSlot(turnID: turn.id, slotID: "slot-0", toKind: "video")
        for _ in 0..<200 {
            let slot = model.turns
                .first(where: { $0.id == turn.id })?.blocks
                .compactMap(\.visualSlot)
                .first(where: { $0.slotID == "slot-0" })
            if slot?.videoArtifact != nil { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        let swapBody = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: try XCTUnwrap(VisualRenderProtocol.bodies().first))
                as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(swapBody["teacache_thresh"] as? Double),
                       0.05, accuracy: 1e-9)
    }

    @MainActor
    func testNarrativeContinuityDefaultsOmitWireKeysAndOptInEncodesExactSchema() throws {
        // Default .off + nil/empty omits continuity_mode and style_anchor completely.
        let defaultNarrative = NarrativeOptions()
        XCTAssertEqual(defaultNarrative.continuityMode, .off)
        XCTAssertNil(defaultNarrative.styleAnchor)

        var defaultOpts = ChatStreamOptions()
        defaultOpts.image = ImageOptions()
        defaultOpts.narrative = defaultNarrative
        let defaultBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: defaultOpts)
        let defaultNav = try XCTUnwrap(defaultBody["narrative"] as? [String: Any])
        XCTAssertEqual(defaultNav.count, 4)
        XCTAssertNil(defaultNav["continuity_mode"])
        XCTAssertNil(defaultNav["style_anchor"])

        // Whitespace-only styleAnchor with .off also omits both keys.
        var whitespaceOpts = ChatStreamOptions()
        whitespaceOpts.image = ImageOptions()
        whitespaceOpts.narrative = NarrativeOptions(
            continuityMode: .off, styleAnchor: "  \t\n  ")
        let whitespaceBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: whitespaceOpts)
        let whitespaceNav = try XCTUnwrap(whitespaceBody["narrative"] as? [String: Any])
        XCTAssertEqual(whitespaceNav.count, 4)
        XCTAssertNil(whitespaceNav["continuity_mode"])
        XCTAssertNil(whitespaceNav["style_anchor"])

        // .i2vChain + valid styleAnchor (trimmed, including exact 256 UTF-8 byte multi-byte cue).
        let exact256 = String(repeating: "墨", count: 85) + "a"
        XCTAssertEqual(exact256.utf8.count, 256)
        var chainOpts = ChatStreamOptions()
        chainOpts.image = ImageOptions()
        chainOpts.video = VideoOptions()
        chainOpts.narrative = NarrativeOptions(
            enabled: true, visualMode: "auto", maxSlots: 3, allowTextOnly: true,
            continuityMode: .i2vChain, styleAnchor: "  \(exact256)  ")
        let chainBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: chainOpts)
        let chainNav = try XCTUnwrap(chainBody["narrative"] as? [String: Any])
        XCTAssertEqual(chainNav["continuity_mode"] as? String, "i2v_chain")
        XCTAssertEqual(chainNav["style_anchor"] as? String, exact256)

        // .promptAnchor with nil styleAnchor omits style_anchor.
        var anchorOpts = ChatStreamOptions()
        anchorOpts.image = ImageOptions()
        anchorOpts.narrative = NarrativeOptions(
            continuityMode: .promptAnchor, styleAnchor: nil)
        let anchorBody = ChatClient.requestBody(
            model: "m", messages: [ChatMessage(role: "user", content: "story")],
            options: anchorOpts)
        let anchorNav = try XCTUnwrap(anchorBody["narrative"] as? [String: Any])
        XCTAssertEqual(anchorNav["continuity_mode"] as? String, "prompt_anchor")
        XCTAssertNil(anchorNav["style_anchor"])
    }

    @MainActor
    func testValidateStyleAnchorRejectsOversizedUTF8BeforeTransport() async throws {
        XCTAssertNil(try validateStyleAnchor(""))
        XCTAssertNil(try validateStyleAnchor("   \n\t "))
        let exact256 = String(repeating: "风", count: 85) + "a"
        XCTAssertEqual(try validateStyleAnchor(" \(exact256) "), exact256)

        let overASCII = String(repeating: "a", count: 257)
        let overCJK = String(repeating: "风", count: 86) // 86 chars, 258 UTF-8 bytes
        XCTAssertEqual(overCJK.count, 86)
        XCTAssertEqual(overCJK.utf8.count, 258)
        XCTAssertThrowsError(try validateStyleAnchor(overASCII))
        XCTAssertThrowsError(try validateStyleAnchor(overCJK))

        // ChatClient.stream must reject >256 UTF-8 bytes before URLSession.
        UnifiedSSEProtocol.resetBodies()
        var opts = ChatStreamOptions()
        opts.image = ImageOptions()
        opts.narrative = NarrativeOptions(
            continuityMode: .i2vChain, styleAnchor: overCJK)
        do {
            _ = try await makeClient().stream(
                base: "http://10.1.2.14/v1", model: "m",
                messages: [ChatMessage(role: "user", content: "story")],
                options: opts) { _ in }
            XCTFail("oversized style_anchor must fail before URLSession")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("256 UTF-8 bytes"))
        }
        XCTAssertTrue(UnifiedSSEProtocol.bodies().isEmpty)
    }

    @MainActor
    func testChatModelStyleAnchorValidationPreservesDraftAndDecodesContinuityBadges() async throws {
        UnifiedSSEProtocol.resetBodies()
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
        try artifactBytes.write(to: imageURL)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.14/v1"
        model.model = "m"
        model.imageOutput = true
        model.videoOutput = true
        model.narrativeMode = true
        model.narrativeContinuityMode = .i2vChain
        model.narrativeStyleAnchor = String(repeating: "景", count: 86) // 258 UTF-8 bytes
        model.attachedImageURL = imageURL
        model.prompt = "two-act lantern tale"
        model.send()

        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.status, "Configuration error")
        XCTAssertTrue(model.error?.contains("256 UTF-8 bytes") == true)
        XCTAssertEqual(model.prompt, "two-act lantern tale",
                       "validation failure must preserve draft prompt")
        XCTAssertEqual(model.attachedImageURL, imageURL,
                       "validation failure must preserve attachments")
        XCTAssertTrue(model.turns.isEmpty)
        XCTAssertTrue(UnifiedSSEProtocol.bodies().isEmpty,
                      "oversized style_anchor must not hit URLSession")

        // Fix style anchor and send: wire body carries both fields and slots decode badges.
        model.narrativeStyleAnchor = "  warm watercolor ink  "
        model.send()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.status, "Completed · stop")
        XCTAssertEqual(UnifiedSSEProtocol.bodies().count, 1)

        let sent = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: UnifiedSSEProtocol.bodies()[0]) as? [String: Any])
        let nav = try XCTUnwrap(sent["narrative"] as? [String: Any])
        XCTAssertEqual(nav["continuity_mode"] as? String, "i2v_chain")
        XCTAssertEqual(nav["style_anchor"] as? String, "warm watercolor ink")

        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
        let slots = turn.blocks.compactMap(\.visualSlot)
        XCTAssertEqual(slots.count, 2)
        XCTAssertEqual(slots[0].continuityMode, "prompt_anchor")
        XCTAssertNil(slots[0].anchorSlotIndex)
        XCTAssertEqual(slots[0].continuityBadgeText, "Anchor")
        XCTAssertEqual(slots[1].continuityMode, "i2v_chain")
        XCTAssertEqual(slots[1].anchorSlotIndex, 0)
        XCTAssertEqual(slots[1].continuityBadgeText, "I2V ← Slot 1")
    }

    @MainActor
    func testVisualSlotContinuityBadgeTextAndHelpForExplicitAndImplicitAnchors() {
        let explicitWithDigest = VisualSlotState(
            slotID: "slot-1", slotIndex: 1, prompt: "river breeze",
            activeKind: "video", isLoading: false,
            continuityMode: "i2v_chain", anchorSlotIndex: 0,
            anchorExplicit: true, anchorSha256: explicitAnchorSHA)
        XCTAssertEqual(explicitWithDigest.continuityBadgeText, "Explicit I2V ← Slot 1 · 3a9d5bbe")
        XCTAssertEqual(explicitWithDigest.continuityBadgeHelp,
                       "Explicit anchor slot-0 · SHA-256: \(explicitAnchorSHA)")

        let explicitNoDigest = VisualSlotState(
            slotID: "slot-1", slotIndex: 1, prompt: "river breeze",
            activeKind: "video", isLoading: true,
            continuityMode: "i2v_chain", anchorSlotIndex: 0,
            anchorExplicit: true, anchorSha256: nil)
        XCTAssertEqual(explicitNoDigest.continuityBadgeText, "Explicit I2V ← Slot 1")
        XCTAssertEqual(explicitNoDigest.continuityBadgeHelp, "Explicit anchor slot-0")

        let implicitPlain = VisualSlotState(
            slotID: "slot-1", slotIndex: 1, prompt: "river breeze",
            activeKind: "video", isLoading: false,
            continuityMode: "i2v_chain", anchorSlotIndex: 0)
        XCTAssertEqual(implicitPlain.continuityBadgeText, "I2V ← Slot 1")
        XCTAssertNil(implicitPlain.continuityBadgeHelp)

        let implicitWithDigest = VisualSlotState(
            slotID: "slot-1", slotIndex: 1, prompt: "river breeze",
            activeKind: "video", isLoading: false,
            continuityMode: "i2v_chain", anchorSlotIndex: 0,
            anchorExplicit: false, anchorSha256: explicitAnchorSHA)
        XCTAssertEqual(implicitWithDigest.continuityBadgeText, "I2V ← Slot 1")
        XCTAssertEqual(implicitWithDigest.continuityBadgeHelp,
                       "Anchor slot-0 · SHA-256: \(explicitAnchorSHA)")

        let promptAnchor = VisualSlotState(
            slotID: "slot-0", slotIndex: 0, prompt: "lantern",
            activeKind: "image", isLoading: false,
            continuityMode: "prompt_anchor")
        XCTAssertEqual(promptAnchor.continuityBadgeText, "Anchor")
        XCTAssertNil(promptAnchor.continuityBadgeHelp)

        let telemetry = TurnTelemetry(
            ttftMs: 120, visualSettledCount: 2, firstVisualMs: 2500,
            hasI2VChain: true, hasExplicitAnchor: true, totalMs: 7400)
        XCTAssertEqual(
            telemetry.formattedPills.map(\.label),
            ["TTFT 120 ms", "2 visuals (2.50 s)", "Explicit I2V", "Total 7.40 s"])
    }

    @MainActor
    func testChatModelDecodesExplicitAnchorDigestFromArtifactAndCompleteLegs() async throws {
        let model = ChatModel(client: makeClient(),
                              audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.1.2.15/v1"
        model.model = "m"
        model.imageOutput = true
        model.videoOutput = true
        model.narrativeMode = true
        model.narrativeContinuityMode = .i2vChain
        model.prompt = "two-beat explicit storyboard"
        model.send()

        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertFalse(model.isRunning)
        XCTAssertEqual(model.status, "Completed · stop")

        let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
        let slots = turn.blocks.compactMap(\.visualSlot)
        XCTAssertEqual(slots.count, 2)
        XCTAssertFalse(slots[0].anchorExplicit)
        XCTAssertNil(slots[0].anchorSha256)
        XCTAssertTrue(slots[1].anchorExplicit)
        XCTAssertEqual(slots[1].anchorSlotIndex, 0)
        XCTAssertEqual(slots[1].anchorSha256, explicitAnchorSHA)
        XCTAssertEqual(slots[1].continuityBadgeText, "Explicit I2V ← Slot 1 · 3a9d5bbe")
        XCTAssertEqual(slots[1].continuityBadgeHelp,
                       "Explicit anchor slot-0 · SHA-256: \(explicitAnchorSHA)")
        XCTAssertTrue(turn.telemetry.hasI2VChain)
        XCTAssertTrue(turn.telemetry.hasExplicitAnchor)
        XCTAssertTrue(turn.telemetry.formattedPills.contains {
            $0.kind == .i2v && $0.label == "Explicit I2V"
        })
    }

    @MainActor
    func testChatModelPropagatesExplicitAnchorSlotLegErrorToSlotCard() async throws {
        for host in ["10.1.2.16", "10.1.2.17"] {
            let model = ChatModel(client: makeClient(),
                                  audioStream: NoOpAudioStreamPlayer())
            model.endpoint = "http://\(host)/v1"
            model.model = "m"
            model.videoOutput = true
            model.narrativeMode = true
            model.prompt = "bad explicit slot reference"
            model.send()

            for _ in 0..<200 where model.isRunning {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            XCTAssertFalse(model.isRunning)
            let turn = try XCTUnwrap(model.turns.last { $0.role == .assistant })
            XCTAssertTrue(turn.legErrors.isEmpty,
                          "explicit slot leg_error must land on the slot card, not turn-level legErrors (\(host))")
            let slots = turn.blocks.compactMap(\.visualSlot)
            XCTAssertEqual(slots.count, 1, host)
            XCTAssertEqual(slots[0].slotID, "slot-1", host)
            XCTAssertTrue(slots[0].anchorExplicit, host)
            XCTAssertFalse(slots[0].isLoading, host)
            XCTAssertEqual(slots[0].error, "source slot slot-1 is not an earlier slot", host)
        }
    }
}
