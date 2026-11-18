// SPDX-License-Identifier: Apache-2.0

import CryptoKit
import XCTest
@testable import ummmlx

private func eventData(_ fields: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: fields)
}

private func unifiedEvent(
    kind: String, seq: Int, text: String = "", session: String = "s-1",
    stage: String = "vlm", isFinal: Bool = false, error: String? = nil,
    artifact: [String: Any]? = nil
) -> [String: Any] {
    var object: [String: Any] = [
        "kind": kind, "seq": seq, "session_id": session, "stage": stage,
        "is_final": isFinal, "text": text
    ]
    if let error { object["error"] = error }
    if let artifact { object["artifact"] = artifact }
    return object
}

private func artifactChunk(
    id: String = "art-0", media: String = "image", mime: String = "image/png",
    index: Int, total: Int, data: Data, sha: String? = nil, bytes: Int? = nil
) -> [String: Any] {
    var object: [String: Any] = [
        "artifact_id": id, "media_type": media, "mime_type": mime,
        "encoding": "base64", "index": index, "total": total,
        "data": data.base64EncodedString()
    ]
    if let sha { object["sha256"] = sha }
    if let bytes { object["bytes"] = bytes }
    return object
}

private func shaHex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

final class UnifiedDecodeTests: XCTestCase {
    func testDecodesEveryDocumentedKind() throws {
        let cases: [(String, UnifiedKind)] = [
            ("asr_partial", .asrPartial), ("asr_final", .asrFinal),
            ("vlm_delta", .vlmDelta), ("artifact", .artifact),
            ("leg_error", .legError), ("complete", .complete),
            ("error", .error), ("cancelled", .cancelled)
        ]
        for (name, expected) in cases {
            var object: [String: Any] = [
                "kind": name, "seq": 0, "session_id": "s-1",
                "stage": "vlm", "is_final": false, "text": "t"
            ]
            if name == "artifact" {
                object["artifact"] = artifactChunk(
                    index: 0, total: 1, data: Data([1]),
                    sha: shaHex(Data([1])), bytes: 1)
            }
            let event = try UnifiedDecode.decode(eventData(object))
            XCTAssertEqual(event.kind, expected, name)
            XCTAssertEqual(event.seq, 0)
            XCTAssertEqual(event.sessionID, "s-1")
        }
    }

    func testDecodesLegErrorText() throws {
        let data = try eventData(unifiedEvent(
            kind: "leg_error", seq: 3, stage: "image", error: "boom"))
        let event = try UnifiedDecode.decode(data)
        XCTAssertEqual(event.kind, .legError)
        XCTAssertEqual(event.error, "boom")
        XCTAssertEqual(event.stage, "image")
    }

    func testRejectsUnknownKindMissingSeqAndWrongTypes() throws {
        for object in [
            unifiedEvent(kind: "mystery", seq: 0),
            ["kind": "complete", "session_id": "s", "stage": "vlm", "text": ""],
            ["kind": 7, "seq": 0, "session_id": "s", "stage": "vlm", "text": ""],
            ["kind": "complete", "seq": -1, "session_id": "s", "stage": "vlm", "text": ""],
            ["kind": "complete", "seq": 0.5, "session_id": "s", "stage": "vlm", "text": ""],
            ["kind": "complete", "seq": 0, "session_id": 1, "stage": "vlm", "text": ""]
        ] {
            XCTAssertThrowsError(try UnifiedDecode.decode(eventData(object)))
        }
    }

    func testArtifactKindRequiresArtifactObject() throws {
        let data = try eventData(unifiedEvent(kind: "artifact", seq: 0))
        XCTAssertThrowsError(try UnifiedDecode.decode(data))
    }

    func testInvalidBase64IsRejected() throws {
        var chunk = artifactChunk(index: 0, total: 1, data: Data([1]))
        chunk["data"] = "not base64!!"
        let data = try eventData(unifiedEvent(kind: "artifact", seq: 0,
                                              artifact: chunk))
        XCTAssertThrowsError(try UnifiedDecode.decode(data))
    }
}

final class UnifiedSequenceTests: XCTestCase {
    func testStartsAtZeroAndAdvances() throws {
        var guardrail = UnifiedSequenceGuard()
        for seq in 0..<5 { try guardrail.accept(seq) }
        XCTAssertEqual(guardrail.last, 4)
    }

    func testFirstSeqMustBeZero() {
        var guardrail = UnifiedSequenceGuard()
        XCTAssertThrowsError(try guardrail.accept(1))
    }

    func testGapFails() throws {
        var guardrail = UnifiedSequenceGuard()
        try guardrail.accept(0)
        try guardrail.accept(1)
        XCTAssertThrowsError(try guardrail.accept(3)) { error in
            XCTAssertEqual(
                error as? ChatClientError,
                .invalidResponse("unified seq 3 follows 1; expected 2"))
        }
    }

    func testDuplicateFails() throws {
        var guardrail = UnifiedSequenceGuard()
        try guardrail.accept(0)
        try guardrail.accept(1)
        XCTAssertThrowsError(try guardrail.accept(1))
    }
}

final class ArtifactCollectorTests: XCTestCase {
    func testSingleChunkVerifiesSha() throws {
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4])
        var collector = ArtifactCollector()
        let artifact = try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: bytes,
            sha256: shaHex(bytes), bytes: bytes.count))
        XCTAssertEqual(artifact?.data, bytes)
        XCTAssertEqual(artifact?.sha256, shaHex(bytes))
        XCTAssertFalse(collector.hasIncompleteArtifacts)
    }

    func testMultiChunkReassemblyUsesFullSha() throws {
        let first = Data([1, 2, 3, 4])
        let second = Data([5, 6, 7])
        let all = first + second
        var collector = ArtifactCollector()
        XCTAssertNil(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "audio", mimeType: "audio/wav",
            encoding: "base64", index: 0, total: 2, data: first,
            sha256: nil, bytes: nil)))
        XCTAssertTrue(collector.hasIncompleteArtifacts)
        let artifact = try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "audio", mimeType: "audio/wav",
            encoding: "base64", index: 1, total: 2, data: second,
            sha256: shaHex(all), bytes: all.count))
        XCTAssertEqual(artifact?.data, all)
        XCTAssertFalse(collector.hasIncompleteArtifacts)
        XCTAssertEqual(collector.decodedBytes, all.count)
    }

    func testShaMismatchFails() {
        let bytes = Data([1, 2, 3])
        var collector = ArtifactCollector()
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: bytes,
            sha256: String(repeating: "0", count: 64), bytes: bytes.count)))
    }

    func testByteCountMismatchFails() {
        let bytes = Data([1, 2, 3])
        var collector = ArtifactCollector()
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: bytes,
            sha256: shaHex(bytes), bytes: bytes.count + 1)))
    }

    func testFinalChunkRequiresShaAndBytes() {
        let bytes = Data([1])
        var collector = ArtifactCollector()
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: bytes,
            sha256: nil, bytes: nil)))
    }

    func testOutOfOrderFails() {
        var collector = ArtifactCollector()
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 1, total: 2, data: Data([2]),
            sha256: nil, bytes: nil)))
    }

    func testChangedTotalFails() throws {
        var collector = ArtifactCollector()
        _ = try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 2, data: Data([1]),
            sha256: nil, bytes: nil))
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 1, total: 3, data: Data([2]),
            sha256: nil, bytes: nil)))
    }

    func testPerArtifactAndAggregateLimits() throws {
        var perArtifact = ArtifactCollector(perArtifactLimit: 2, aggregateLimit: 10)
        XCTAssertThrowsError(try perArtifact.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: Data([1, 2, 3]),
            sha256: shaHex(Data([1, 2, 3])), bytes: 3)))

        var aggregate = ArtifactCollector(perArtifactLimit: 10, aggregateLimit: 3)
        _ = try aggregate.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: Data([1, 2]),
            sha256: shaHex(Data([1, 2])), bytes: 2))
        XCTAssertThrowsError(try aggregate.ingest(ArtifactChunk(
            artifactID: "art-1", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: Data([3, 4]),
            sha256: shaHex(Data([3, 4])), bytes: 2)))
    }

    func testUnsupportedMimeFails() {
        var collector = ArtifactCollector()
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "image/webp",
            encoding: "base64", index: 0, total: 1, data: Data([1]),
            sha256: shaHex(Data([1])), bytes: 1)))
    }
}
