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
        // The H20B deployment has no server-side history and rejects the flag.
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
            try ChatClient.chatURL(base: "http://172.17.0.2:43301/v1",
                                   allowedHTTPHosts: []).absoluteString,
            "http://172.17.0.2:43301/v1/chat/completions")
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
}
