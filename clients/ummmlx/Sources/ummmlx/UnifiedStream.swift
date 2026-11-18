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
    let format: String?
    let sampleRate: Int?
    let channels: Int?

    init(artifactID: String, mediaType: String, mimeType: String,
         encoding: String, index: Int, total: Int, data: Data,
         sha256: String?, bytes: Int?, format: String? = nil,
         sampleRate: Int? = nil, channels: Int? = nil) {
        self.artifactID = artifactID
        self.mediaType = mediaType
        self.mimeType = mimeType
        self.encoding = encoding
        self.index = index
        self.total = total
        self.data = data
        self.sha256 = sha256
        self.bytes = bytes
        self.format = format
        self.sampleRate = sampleRate
        self.channels = channels
    }
}

struct AudioStreamChunk: Equatable, Sendable {
    let pcm16: Data
    let sampleRate: Int
    let channels: Int
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
    let sampleRate: Int?
    let channels: Int?

    init(id: String, mediaType: String, mimeType: String, data: Data,
         sha256: String, sampleRate: Int? = nil, channels: Int? = nil) {
        self.id = id
        self.mediaType = mediaType
        self.mimeType = mimeType
        self.data = data
        self.sha256 = sha256
        self.sampleRate = sampleRate
        self.channels = channels
    }
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
        let format = try raw["format"].map { try string($0, "artifact.format") }
        let sampleRate = try raw["sample_rate"].map {
            try integer($0, "artifact.sample_rate")
        }
        let channels = try raw["channels"].map {
            try integer($0, "artifact.channels")
        }
        return ArtifactChunk(
            artifactID: artifactID, mediaType: mediaType, mimeType: mimeType,
            encoding: encoding, index: index, total: total, data: decoded,
            sha256: sha, bytes: bytes, format: format,
            sampleRate: sampleRate, channels: channels
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
        "image/png", "image/jpeg", "audio/wav", "audio/pcm"
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
        guard Self.supportedMimeTypes.contains(chunk.mimeType),
              chunk.mimeType.hasPrefix(chunk.mediaType + "/") else {
            throw ChatClientError.invalidResponse(
                "artifact mime_type '\(chunk.mimeType)' is not supported")
        }
        var pcmSampleRate: Int?
        var pcmChannels: Int?
        if chunk.mimeType == "audio/pcm" {
            if let format = chunk.format, format != "pcm16" {
                throw ChatClientError.invalidResponse(
                    "audio/pcm artifact format '\(format)' is not pcm16")
            }
            let rate = chunk.sampleRate ?? 24_000
            let ch = chunk.channels ?? 1
            guard rate > 0, ch >= 1, ch <= 8 else {
                throw ChatClientError.invalidResponse(
                    "audio/pcm artifact carries invalid sample_rate or channels")
            }
            pcmSampleRate = rate
            pcmChannels = ch
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
        if let ch = pcmChannels {
            let frameBytes = ch * 2
            guard !slot.data.isEmpty, slot.data.count % frameBytes == 0 else {
                throw ChatClientError.invalidResponse(
                    "audio/pcm artifact does not contain whole PCM16 samples")
            }
        }
        let digest = Self.sha256Hex(slot.data)
        guard digest == reportedSha.lowercased() else {
            throw ChatClientError.invalidResponse("artifact sha256 mismatch")
        }
        slots.removeValue(forKey: chunk.artifactID)
        return GeneratedArtifact(
            id: chunk.artifactID, mediaType: chunk.mediaType,
            mimeType: chunk.mimeType, data: slot.data, sha256: digest,
            sampleRate: pcmSampleRate, channels: pcmChannels
        )
    }

    /// An artifact that started but never reached its final chunk.
    var hasIncompleteArtifacts: Bool { !slots.isEmpty }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

enum WAVCodec {
    static let headerByteCount = 44

    static func wrapPCM16(_ pcm: Data, sampleRate: Int = 24_000,
                          channels: Int = 1) throws -> Data {
        guard sampleRate > 0, channels >= 1, channels <= 8 else {
            throw ChatClientError.invalidResponse("invalid WAV sample rate or channel count")
        }
        let blockAlign = channels * 2
        guard !pcm.isEmpty, pcm.count % blockAlign == 0 else {
            throw ChatClientError.invalidResponse("PCM16 audio must contain whole samples")
        }
        let byteRate64 = UInt64(sampleRate) * UInt64(blockAlign)
        let riffPayload64 = UInt64(pcm.count) + 36
        guard byteRate64 <= UInt64(UInt32.max),
              riffPayload64 <= UInt64(UInt32.max) else {
            throw ChatClientError.invalidResponse("PCM16 audio exceeds WAV container limits")
        }

        var wav = Data()
        wav.reserveCapacity(headerByteCount + pcm.count)
        func appendU16(_ value: UInt16) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { wav.append(contentsOf: $0) }
        }
        func appendU32(_ value: UInt32) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { wav.append(contentsOf: $0) }
        }
        wav.append(contentsOf: "RIFF".utf8)
        appendU32(UInt32(riffPayload64))
        wav.append(contentsOf: "WAVEfmt ".utf8)
        appendU32(16)
        appendU16(1)  // PCM
        appendU16(UInt16(channels))
        appendU32(UInt32(sampleRate))
        appendU32(UInt32(byteRate64))
        appendU16(UInt16(blockAlign))
        appendU16(16)
        wav.append(contentsOf: "data".utf8)
        appendU32(UInt32(pcm.count))
        wav.append(pcm)
        return wav
    }
}

/// Coalesces incremental `audio/pcm` chunks of one chat turn into a single
/// replayable `audio/wav` artifact while forwarding each chunk immediately for
/// live playback.
struct StreamedAudioAccumulator {
    private let perArtifactLimit: Int
    private var firstArtifactID: String?
    private var sampleRate: Int?
    private var channels: Int?
    private var pcm = Data()

    init(perArtifactLimit: Int = 48 << 20) {
        self.perArtifactLimit = perArtifactLimit
    }

    var isEmpty: Bool { pcm.isEmpty }

    mutating func append(_ artifact: GeneratedArtifact) throws -> AudioStreamChunk {
        let rate = artifact.sampleRate ?? 24_000
        let ch = artifact.channels ?? 1
        if let existingRate = sampleRate, let existingChannels = channels {
            guard existingRate == rate, existingChannels == ch else {
                throw ChatClientError.invalidResponse(
                    "streamed audio format changed between chunks")
            }
        } else {
            firstArtifactID = artifact.id
            sampleRate = rate
            channels = ch
        }
        guard WAVCodec.headerByteCount + pcm.count + artifact.data.count <= perArtifactLimit else {
            throw ChatClientError.invalidResponse("audio artifact exceeds its size limit")
        }
        pcm.append(artifact.data)
        return AudioStreamChunk(pcm16: artifact.data, sampleRate: rate, channels: ch)
    }

    mutating func finalizeWAV() throws -> GeneratedArtifact? {
        guard !pcm.isEmpty,
              let id = firstArtifactID,
              let rate = sampleRate,
              let ch = channels else {
            return nil
        }
        let wav = try WAVCodec.wrapPCM16(pcm, sampleRate: rate, channels: ch)
        pcm.removeAll(keepingCapacity: false)
        return GeneratedArtifact(
            id: id, mediaType: "audio", mimeType: "audio/wav",
            data: wav, sha256: ArtifactCollector.sha256Hex(wav),
            sampleRate: rate, channels: ch
        )
    }
}
