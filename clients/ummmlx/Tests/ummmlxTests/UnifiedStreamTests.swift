// SPDX-License-Identifier: Apache-2.0

import AVFoundation
import CryptoKit
import XCTest
@testable import ummmlx

private func eventData(_ fields: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: fields)
}

private func unifiedEvent(
    kind: String, seq: Int, text: String = "", session: String = "s-1",
    stage: String = "vlm", isFinal: Bool = false, error: String? = nil,
    artifact: [String: Any]? = nil, slotID: String? = nil,
    slotIndex: Int? = nil, mediaType: String? = nil, prompt: String? = nil,
    textOffset: Int? = nil, continuityMode: String? = nil,
    anchorSlotIndex: Int? = nil, anchorExplicit: Bool? = nil,
    anchorSha256: String? = nil, legs: [[String: Any]]? = nil
) -> [String: Any] {
    var object: [String: Any] = [
        "kind": kind, "seq": seq, "session_id": session, "stage": stage,
        "is_final": isFinal, "text": text
    ]
    if let error { object["error"] = error }
    if let artifact { object["artifact"] = artifact }
    if let slotID { object["slot_id"] = slotID }
    if let slotIndex { object["slot_index"] = slotIndex }
    if let mediaType { object["media_type"] = mediaType }
    if let prompt { object["prompt"] = prompt }
    if let textOffset { object["text_offset"] = textOffset }
    if let continuityMode { object["continuity_mode"] = continuityMode }
    if let anchorSlotIndex { object["anchor_slot_index"] = anchorSlotIndex }
    if let anchorExplicit { object["anchor_explicit"] = anchorExplicit }
    if let anchorSha256 { object["anchor_sha256"] = anchorSha256 }
    if let legs { object["legs"] = legs }
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
            ("vlm_delta", .vlmDelta), ("visual_slot", .visualSlot),
            ("artifact", .artifact),
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
            if name == "visual_slot" {
                object["slot_id"] = "slot-0"
                object["slot_index"] = 0
                object["media_type"] = "image"
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

    func testDecodesVisualSlotAttribution() throws {
        let data = try eventData(unifiedEvent(
            kind: "visual_slot", seq: 4, stage: "image",
            slotID: "slot-0", slotIndex: 0, mediaType: "image",
            prompt: "a glowing red lantern over a misty lake", textOffset: 128))
        let event = try UnifiedDecode.decode(data)
        XCTAssertEqual(event.kind, .visualSlot)
        XCTAssertEqual(event.stage, "image")
        XCTAssertEqual(event.slotID, "slot-0")
        XCTAssertEqual(event.slotIndex, 0)
        XCTAssertEqual(event.mediaType, "image")
        XCTAssertEqual(event.prompt, "a glowing red lantern over a misty lake")
        XCTAssertEqual(event.textOffset, 128)
        XCTAssertNil(event.continuityMode)
        XCTAssertNil(event.anchorSlotIndex)
        XCTAssertFalse(event.kind.isTerminal)
    }

    func testDecodesVisualSlotContinuityAttributionAndToleratesNulls() throws {
        let chained = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "visual_slot", seq: 1, stage: "video",
            slotID: "slot-1", slotIndex: 1, mediaType: "video",
            prompt: "camera pans past the lantern", textOffset: 200,
            continuityMode: "i2v_chain", anchorSlotIndex: 0)))
        XCTAssertEqual(chained.kind, .visualSlot)
        XCTAssertEqual(chained.continuityMode, "i2v_chain")
        XCTAssertEqual(chained.anchorSlotIndex, 0)

        let anchored = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "visual_slot", seq: 0, stage: "image",
            slotID: "slot-0", slotIndex: 0, mediaType: "image",
            prompt: "lantern over lake", textOffset: 40,
            continuityMode: "prompt_anchor")))
        XCTAssertEqual(anchored.continuityMode, "prompt_anchor")
        XCTAssertNil(anchored.anchorSlotIndex)

        var nullFields = unifiedEvent(
            kind: "visual_slot", seq: 2, stage: "image",
            slotID: "slot-2", slotIndex: 2, mediaType: "image")
        nullFields["continuity_mode"] = NSNull()
        nullFields["anchor_slot_index"] = NSNull()
        let nullable = try UnifiedDecode.decode(eventData(nullFields))
        XCTAssertNil(nullable.continuityMode)
        XCTAssertNil(nullable.anchorSlotIndex)

        var badMode = unifiedEvent(
            kind: "visual_slot", seq: 3, stage: "video",
            slotID: "slot-3", slotIndex: 3, mediaType: "video")
        badMode["continuity_mode"] = 42
        XCTAssertThrowsError(try UnifiedDecode.decode(eventData(badMode)))

        var badAnchor = unifiedEvent(
            kind: "visual_slot", seq: 4, stage: "video",
            slotID: "slot-4", slotIndex: 4, mediaType: "video")
        badAnchor["anchor_slot_index"] = true
        XCTAssertThrowsError(try UnifiedDecode.decode(eventData(badAnchor)))
    }

    func testVisualSlotRequiresSlotIdentity() throws {
        XCTAssertThrowsError(try UnifiedDecode.decode(
            eventData(unifiedEvent(kind: "visual_slot", seq: 0,
                                   mediaType: "image"))))
        XCTAssertThrowsError(try UnifiedDecode.decode(
            eventData(unifiedEvent(kind: "visual_slot", seq: 0,
                                   slotID: "slot-0"))))
    }

    func testLegErrorCarriesSlotAttribution() throws {
        let event = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "leg_error", seq: 5, stage: "video", error: "oom",
            slotID: "slot-1", slotIndex: 1, mediaType: "video")))
        XCTAssertEqual(event.kind, .legError)
        XCTAssertEqual(event.error, "oom")
        XCTAssertEqual(event.slotID, "slot-1")
        XCTAssertEqual(event.slotIndex, 1)
        XCTAssertEqual(event.mediaType, "video")
        XCTAssertFalse(event.anchorExplicit)
        XCTAssertNil(event.anchorSha256)
    }

    func testDecodesExplicitAnchorAndSha256AcrossEventsAndToleratesOmission() throws {
        let digest = "3a9d5bbe" + String(repeating: "4f12c890", count: 7)
        XCTAssertEqual(digest.count, 64)

        // 1. visual_slot with anchor_explicit=true and uppercase anchor_sha256.
        let explicitSlot = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "visual_slot", seq: 1, stage: "video",
            slotID: "slot-1", slotIndex: 1, mediaType: "video",
            prompt: "river breeze", textOffset: 80,
            continuityMode: "i2v_chain", anchorSlotIndex: 0,
            anchorExplicit: true, anchorSha256: digest.uppercased())))
        XCTAssertEqual(explicitSlot.anchorSlotIndex, 0)
        XCTAssertTrue(explicitSlot.anchorExplicit)
        XCTAssertEqual(explicitSlot.anchorSha256, digest)

        // 2. leg_error with anchor_explicit=true, anchor_slot_index, and anchor_sha256.
        let explicitErr = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "leg_error", seq: 2, stage: "video",
            error: "source slot slot-1 is not an earlier slot",
            slotID: "slot-1", slotIndex: 1, mediaType: "video",
            anchorSlotIndex: 0, anchorExplicit: true, anchorSha256: digest)))
        XCTAssertEqual(explicitErr.kind, .legError)
        XCTAssertEqual(explicitErr.error, "source slot slot-1 is not an earlier slot")
        XCTAssertEqual(explicitErr.anchorSlotIndex, 0)
        XCTAssertTrue(explicitErr.anchorExplicit)
        XCTAssertEqual(explicitErr.anchorSha256, digest)

        // 3. complete.legs carrying anchor_explicit + anchor_sha256 + omission compatibility.
        let complete = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "complete", seq: 3, text: "done", stage: "vlm", isFinal: true,
            legs: [
                ["stage": "image", "state": "completed", "slot_id": "slot-0", "slot_index": 0],
                [
                    "stage": "video", "state": "completed",
                    "slot_id": "slot-1", "slot_index": 1,
                    "anchor_slot_index": 0, "anchor_explicit": true,
                    "anchor_sha256": digest
                ]
            ])))
        XCTAssertEqual(complete.legs.count, 2)
        XCTAssertEqual(complete.legs[0].slotID, "slot-0")
        XCTAssertFalse(complete.legs[0].anchorExplicit)
        XCTAssertNil(complete.legs[0].anchorSha256)
        XCTAssertEqual(complete.legs[1].slotID, "slot-1")
        XCTAssertEqual(complete.legs[1].anchorSlotIndex, 0)
        XCTAssertTrue(complete.legs[1].anchorExplicit)
        XCTAssertEqual(complete.legs[1].anchorSha256, digest)

        // 4. artifact carrying anchor_slot_index + anchor_explicit + anchor_sha256 survives collector.
        let bytes = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70])
        var chunk = artifactChunk(
            id: "art-vid-1", media: "video", mime: "video/mp4",
            index: 0, total: 1, data: bytes, sha: shaHex(bytes), bytes: bytes.count)
        chunk["slot_id"] = "slot-1"
        chunk["slot_index"] = 1
        chunk["anchor_slot_index"] = 0
        chunk["anchor_explicit"] = true
        chunk["anchor_sha256"] = digest
        let artEvent = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "artifact", seq: 4, stage: "video", artifact: chunk)))
        XCTAssertEqual(artEvent.artifact?.anchorSlotIndex, 0)
        XCTAssertEqual(artEvent.artifact?.anchorExplicit, true)
        XCTAssertEqual(artEvent.artifact?.anchorSha256, digest)
        var collector = ArtifactCollector()
        let assembled = try XCTUnwrap(try collector.ingest(try XCTUnwrap(artEvent.artifact)))
        XCTAssertEqual(assembled.anchorSlotIndex, 0)
        XCTAssertTrue(assembled.anchorExplicit)
        XCTAssertEqual(assembled.anchorSha256, digest)

        // 5. Omission, null, empty, and non-64-hex anchor_sha256 yield defaults cleanly.
        var nullable = unifiedEvent(
            kind: "visual_slot", seq: 5, stage: "video",
            slotID: "slot-1", slotIndex: 1, mediaType: "video",
            continuityMode: "i2v_chain", anchorSlotIndex: 0)
        nullable["anchor_explicit"] = NSNull()
        nullable["anchor_sha256"] = NSNull()
        let decodedNull = try UnifiedDecode.decode(eventData(nullable))
        XCTAssertFalse(decodedNull.anchorExplicit)
        XCTAssertNil(decodedNull.anchorSha256)

        for emptyOrShort in ["", "   ", "3a9d5bbe", String(repeating: "z", count: 64)] {
            let decoded = try UnifiedDecode.decode(eventData(unifiedEvent(
                kind: "visual_slot", seq: 6, stage: "video",
                slotID: "slot-1", slotIndex: 1, mediaType: "video",
                anchorSlotIndex: 0, anchorExplicit: true, anchorSha256: emptyOrShort)))
            XCTAssertTrue(decoded.anchorExplicit)
            XCTAssertNil(decoded.anchorSha256)
        }

        // 6. Wrong JSON types for anchor_explicit / anchor_sha256 are rejected.
        var badExplicit = unifiedEvent(
            kind: "visual_slot", seq: 7, stage: "video",
            slotID: "slot-1", slotIndex: 1, mediaType: "video")
        badExplicit["anchor_explicit"] = "yes"
        XCTAssertThrowsError(try UnifiedDecode.decode(eventData(badExplicit)))

        var badSha = unifiedEvent(
            kind: "leg_error", seq: 8, stage: "video", error: "fail",
            slotID: "slot-1", slotIndex: 1)
        badSha["anchor_sha256"] = 12345
        XCTAssertThrowsError(try UnifiedDecode.decode(eventData(badSha)))
    }

    func testArtifactDecodesSlotAttribution() throws {
        let bytes = Data([1, 2, 3])
        var chunk = artifactChunk(index: 0, total: 1, data: bytes,
                                  sha: shaHex(bytes), bytes: bytes.count)
        chunk["slot_id"] = "slot-0"
        chunk["slot_index"] = 0
        let event = try UnifiedDecode.decode(eventData(unifiedEvent(
            kind: "artifact", seq: 0, artifact: chunk)))
        XCTAssertEqual(event.artifact?.slotID, "slot-0")
        XCTAssertEqual(event.artifact?.slotIndex, 0)
    }

    func testUnknownKindIsTolerated() throws {
        // A newer server may add a kind. It must not fail the stream: its seq is
        // consumed and it is never treated as a terminal.
        let event = try UnifiedDecode.decode(
            eventData(unifiedEvent(kind: "mystery", seq: 0)))
        XCTAssertEqual(event.kind, .unknown)
        XCTAssertEqual(event.seq, 0)
        XCTAssertFalse(event.kind.isTerminal)
    }

    func testRejectsMissingSeqAndWrongTypes() throws {
        for object in [
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

    func testReassemblesMultiChunkVideoArtifact() throws {
        let first = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70])
        let second = Data([0x6d, 0x70, 0x34, 0x32, 0x00, 0x00, 0x00, 0x00])
        let all = first + second
        var collector = ArtifactCollector()
        XCTAssertNil(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "video", mimeType: "video/mp4",
            encoding: "base64", index: 0, total: 2, data: first,
            sha256: nil, bytes: nil)))
        XCTAssertTrue(collector.hasIncompleteArtifacts)
        let artifact = try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "video", mimeType: "video/mp4",
            encoding: "base64", index: 1, total: 2, data: second,
            sha256: shaHex(all), bytes: all.count))
        XCTAssertEqual(artifact?.mediaType, "video")
        XCTAssertEqual(artifact?.mimeType, "video/mp4")
        XCTAssertEqual(artifact?.data, all)
        XCTAssertFalse(collector.hasIncompleteArtifacts)
    }

    func testSlotAttributionSurvivesReassembly() throws {
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4])
        var collector = ArtifactCollector()
        let artifact = try collector.ingest(ArtifactChunk(
            artifactID: "art-slot-0", mediaType: "image", mimeType: "image/png",
            encoding: "base64", index: 0, total: 1, data: bytes,
            sha256: shaHex(bytes), bytes: bytes.count,
            slotID: "slot-0", slotIndex: 0))
        XCTAssertEqual(artifact?.slotID, "slot-0")
        XCTAssertEqual(artifact?.slotIndex, 0)
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
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "image", mimeType: "audio/wav",
            encoding: "base64", index: 0, total: 1, data: Data([1, 2]),
            sha256: shaHex(Data([1, 2])), bytes: 2)))
    }

    func testAudioPCMChunkValidatesAlignmentAndFormat() throws {
        var collector = ArtifactCollector()
        let pcm = Data([0x00, 0x80, 0xFF, 0x7F])
        let artifact = try collector.ingest(ArtifactChunk(
            artifactID: "art-0", mediaType: "audio", mimeType: "audio/pcm",
            encoding: "base64", index: 0, total: 1, data: pcm,
            sha256: shaHex(pcm), bytes: pcm.count, format: "pcm16",
            sampleRate: 24_000, channels: 1))
        XCTAssertEqual(artifact?.mimeType, "audio/pcm")
        XCTAssertEqual(artifact?.sampleRate, 24_000)
        XCTAssertEqual(artifact?.channels, 1)

        let odd = Data([0x01, 0x02, 0x03])
        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-1", mediaType: "audio", mimeType: "audio/pcm",
            encoding: "base64", index: 0, total: 1, data: odd,
            sha256: shaHex(odd), bytes: odd.count, format: "pcm16",
            sampleRate: 24_000, channels: 1)))

        XCTAssertThrowsError(try collector.ingest(ArtifactChunk(
            artifactID: "art-2", mediaType: "audio", mimeType: "audio/pcm",
            encoding: "base64", index: 0, total: 1, data: pcm,
            sha256: shaHex(pcm), bytes: pcm.count, format: "mulaw",
            sampleRate: 24_000, channels: 1)))
    }

    func testWAVCodecAndStreamedAccumulatorProducePlayableWAV() throws {
        // 2400 mono samples = 0.1 s at 24 kHz.
        let c0 = Data(repeating: 0x01, count: 2400)
        let c1 = Data(repeating: 0x02, count: 2400)
        var acc = StreamedAudioAccumulator()
        let s0 = try acc.append(GeneratedArtifact(
            id: "art-0", mediaType: "audio", mimeType: "audio/pcm",
            data: c0, sha256: shaHex(c0), sampleRate: 24_000, channels: 1))
        let s1 = try acc.append(GeneratedArtifact(
            id: "art-1", mediaType: "audio", mimeType: "audio/pcm",
            data: c1, sha256: shaHex(c1), sampleRate: 24_000, channels: 1))
        XCTAssertEqual(s0.pcm16, c0)
        XCTAssertEqual(s1.pcm16, c1)

        let wav = try XCTUnwrap(try acc.finalizeWAV())
        XCTAssertEqual(wav.id, "art-0")
        XCTAssertEqual(wav.mimeType, "audio/wav")
        XCTAssertEqual(wav.data.count, 44 + c0.count + c1.count)
        XCTAssertEqual(wav.data.subdata(in: 44..<wav.data.count), c0 + c1)
        let player = try AVAudioPlayer(data: wav.data)
        XCTAssertEqual(player.duration, 0.1, accuracy: 0.01)
    }

    func testStreamedAccumulatorRejectsMidStreamFormatChange() throws {
        let pcm = Data([0, 0, 1, 0])
        var acc = StreamedAudioAccumulator()
        _ = try acc.append(GeneratedArtifact(
            id: "art-0", mediaType: "audio", mimeType: "audio/pcm",
            data: pcm, sha256: shaHex(pcm), sampleRate: 24_000, channels: 1))
        XCTAssertThrowsError(try acc.append(GeneratedArtifact(
            id: "art-1", mediaType: "audio", mimeType: "audio/pcm",
            data: pcm, sha256: shaHex(pcm), sampleRate: 16_000, channels: 1)))
    }

    @MainActor
    func testMakePCMBufferConvertsLittleEndianInt16ToFloat32() throws {
        // Stereo frame: left = -32768 (0x8000), right = 16384 (0x4000).
        let bytes = Data([0x00, 0x80, 0x00, 0x40])
        let chunk = AudioStreamChunk(pcm16: bytes, sampleRate: 24_000, channels: 2)
        let buffer = try XCTUnwrap(LiveAudioStreamPlayer.makePCMBuffer(chunk))
        XCTAssertEqual(buffer.frameLength, 1)
        XCTAssertEqual(buffer.format.channelCount, 2)
        XCTAssertEqual(buffer.format.sampleRate, 24_000)
        let channels = try XCTUnwrap(buffer.floatChannelData)
        XCTAssertEqual(channels[0][0], -1.0, accuracy: 1e-6)
        XCTAssertEqual(channels[1][0], 0.5, accuracy: 1e-6)
    }
}
