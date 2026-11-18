// SPDX-License-Identifier: Apache-2.0
// Unified event parsing for the ummInfer mixed-output contract
// (docs/mixed-output-contract.md). Pure values and checks: no network, no UI.

import CoreFoundation
import CryptoKit
import Foundation

enum UnifiedKind: String, CaseIterable {
    case asrPartial = "asr_partial"
    case asrFinal = "asr_final"
    case vlmDelta = "vlm_delta"
    case artifact
    case legError = "leg_error"
    case complete
    case error
    case cancelled

    var isTerminal: Bool { self == .complete || self == .error || self == .cancelled }
}

struct ArtifactChunk {
    let artifactID: String
    let mediaType: String
    let mimeType: String
    let encoding: String
    let index: Int
    let total: Int
    let data: Data
    let sha256: String?
    let bytes: Int?
}

struct UnifiedEvent {
    let kind: UnifiedKind
    let seq: Int
    let sessionID: String
    let stage: String
    let isFinal: Bool
    let text: String
    let error: String
    let artifact: ArtifactChunk?
}

/// One complete generated artifact after chunk reassembly and verification.
struct GeneratedArtifact: Identifiable, Equatable {
    let id: String
    let mediaType: String
    let mimeType: String
    let data: Data
    let sha256: String
}

enum UnifiedDecode {
    static func object(_ value: Any?) throws -> [String: Any] {
        guard let object = value as? [String: Any] else {
            throw ChatClientError.invalidResponse("a unified event is not an object")
        }
        return object
    }

    static func string(_ value: Any?, _ name: String) throws -> String {
        guard let text = value as? String else {
            throw ChatClientError.invalidResponse("unified \(name) is not a string")
        }
        return text
    }

    static func integer(_ value: Any?, _ name: String) throws -> Int {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number) else {
            throw ChatClientError.invalidResponse("unified \(name) is not an integer")
        }
        return number.intValue
    }

    static func boolean(_ value: Any?, _ name: String) throws -> Bool {
        guard let flag = value as? NSNumber, CFNumberIsFloatType(flag) == false,
              flag.intValue == 0 || flag.intValue == 1 else {
            throw ChatClientError.invalidResponse("unified \(name) is not a boolean")
        }
        return flag.intValue == 1
    }

    static func decode(_ data: Data) throws -> UnifiedEvent {
        let root: [String: Any]
        do {
            root = try object(JSONSerialization.jsonObject(with: data))
        } catch let error as ChatClientError {
            throw error
        } catch {
            throw ChatClientError.invalidResponse("a unified event is not JSON")
        }
        let kindText = try string(root["kind"], "kind")
        guard let kind = UnifiedKind(rawValue: kindText) else {
            throw ChatClientError.invalidResponse("unified kind '\(kindText)' is unknown")
        }
        let seq = try integer(root["seq"], "seq")
        guard seq >= 0 else {
            throw ChatClientError.invalidResponse("unified seq is negative")
        }
        let sessionID = try string(root["session_id"], "session_id")
        let stage = try string(root["stage"], "stage")
        let isFinal: Bool
        if let rawFinal = root["is_final"] {
            isFinal = try boolean(rawFinal, "is_final")
        } else {
            isFinal = false
        }
        let text = try root["text"].map { try string($0, "text") } ?? ""
        let errorText = try root["error"].map { try string($0, "error") } ?? ""

        var artifact: ArtifactChunk?
        if kind == .artifact {
            let raw = try object(root["artifact"])
            artifact = try decodeArtifact(raw)
        }
        return UnifiedEvent(
            kind: kind, seq: seq, sessionID: sessionID, stage: stage,
            isFinal: isFinal, text: text, error: errorText,
            artifact: artifact
        )
    }

    private static func decodeArtifact(_ raw: [String: Any]) throws -> ArtifactChunk {
        func require(_ name: String) throws -> Any {
            guard let value = raw[name] else {
                throw ChatClientError.invalidResponse("artifact is missing \(name)")
            }
            return value
        }
        let artifactID = try string(try require("artifact_id"), "artifact_id")
        let mediaType = try string(try require("media_type"), "media_type")
        let mimeType = try string(try require("mime_type"), "mime_type")
        let encoding = try string(try require("encoding"), "encoding")
        let index = try integer(try require("index"), "artifact.index")
        let total = try integer(try require("total"), "artifact.total")
        let encoded = try string(try require("data"), "artifact.data")
        guard let decoded = Data(base64Encoded: encoded, options: []) else {
            throw ChatClientError.invalidResponse("artifact data is not legal base64")
        }
        let sha = try raw["sha256"].map { try string($0, "artifact.sha256") }
        let bytes = try raw["bytes"].map { try integer($0, "artifact.bytes") }
        return ArtifactChunk(
            artifactID: artifactID, mediaType: mediaType, mimeType: mimeType,
            encoding: encoding, index: index, total: total, data: decoded,
            sha256: sha, bytes: bytes
        )
    }
}

/// Enforces seq 0, +1, +1, ... for one request. A gap, a repeat or a nonzero
/// first value is a protocol failure, never a silent skip.
struct UnifiedSequenceGuard {
    private(set) var last: Int?

    mutating func accept(_ seq: Int) throws {
        if let last {
            guard seq == last + 1 else {
                throw ChatClientError.invalidResponse(
                    "unified seq \(seq) follows \(last); expected \(last + 1)")
            }
        } else {
            guard seq == 0 else {
                throw ChatClientError.invalidResponse(
                    "unified seq starts at \(seq), not 0")
            }
        }
        last = seq
    }
}

/// Reassembles base64 artifact chunks by artifact_id, checks order, size, the
/// final-chunk byte count and sha256 against the decoded bytes.
struct ArtifactCollector {
    private struct Slot {
        var mediaType: String
        var mimeType: String
        var nextIndex: Int
        var total: Int
        var data: Data
    }

    static let supportedMimeTypes: Set<String> = [
        "image/png", "image/jpeg", "audio/wav"
    ]

    private var slots: [String: Slot] = [:]
    private(set) var decodedBytes = 0
    private let perArtifactLimit: Int
    private let aggregateLimit: Int

    init(perArtifactLimit: Int = 48 << 20, aggregateLimit: Int = 60 << 20) {
        self.perArtifactLimit = perArtifactLimit
        self.aggregateLimit = aggregateLimit
    }

    mutating func ingest(_ chunk: ArtifactChunk) throws -> GeneratedArtifact? {
        guard chunk.mediaType == "image" || chunk.mediaType == "audio" else {
            throw ChatClientError.invalidResponse(
                "artifact media_type must be image or audio")
        }
        guard Self.supportedMimeTypes.contains(chunk.mimeType) else {
            throw ChatClientError.invalidResponse(
                "artifact mime_type '\(chunk.mimeType)' is not supported")
        }
        guard chunk.encoding == "base64" else {
            throw ChatClientError.invalidResponse("artifact encoding must be base64")
        }
        guard chunk.total >= 1, chunk.index >= 0, chunk.index < chunk.total else {
            throw ChatClientError.invalidResponse("artifact chunk index is out of range")
        }
        guard !chunk.artifactID.isEmpty else {
            throw ChatClientError.invalidResponse("artifact_id is empty")
        }

        var slot = slots[chunk.artifactID] ?? Slot(
            mediaType: chunk.mediaType, mimeType: chunk.mimeType,
            nextIndex: 0, total: chunk.total, data: Data()
        )
        guard slot.mediaType == chunk.mediaType, slot.mimeType == chunk.mimeType else {
            throw ChatClientError.invalidResponse(
                "artifact descriptor changed between chunks")
        }
        guard slot.total == chunk.total else {
            throw ChatClientError.invalidResponse(
                "artifact total changed between chunks")
        }
        guard chunk.index == slot.nextIndex else {
            throw ChatClientError.invalidResponse(
                "artifact chunks arrived out of order: expected \(slot.nextIndex), got \(chunk.index)")
        }
        guard slot.data.count + chunk.data.count <= perArtifactLimit else {
            throw ChatClientError.invalidResponse(
                "\(chunk.mediaType) artifact exceeds its size limit")
        }
        guard decodedBytes + chunk.data.count <= aggregateLimit else {
            throw ChatClientError.invalidResponse(
                "artifacts exceed the aggregate byte limit")
        }
        decodedBytes += chunk.data.count
        slot.data.append(chunk.data)
        slot.nextIndex = chunk.index + 1

        guard chunk.index + 1 == chunk.total else {
            slots[chunk.artifactID] = slot
            return nil
        }
        guard let reportedSha = chunk.sha256, let reportedBytes = chunk.bytes else {
            throw ChatClientError.invalidResponse(
                "final artifact chunk requires sha256 and bytes")
        }
        guard reportedBytes == slot.data.count else {
            throw ChatClientError.invalidResponse(
                "artifact byte count \(reportedBytes) does not match \(slot.data.count) decoded bytes")
        }
        let digest = Self.sha256Hex(slot.data)
        guard digest == reportedSha.lowercased() else {
            throw ChatClientError.invalidResponse("artifact sha256 mismatch")
        }
        slots.removeValue(forKey: chunk.artifactID)
        return GeneratedArtifact(
            id: chunk.artifactID, mediaType: chunk.mediaType,
            mimeType: chunk.mimeType, data: slot.data, sha256: digest
        )
    }

    /// An artifact that started but never reached its final chunk.
    var hasIncompleteArtifacts: Bool { !slots.isEmpty }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
