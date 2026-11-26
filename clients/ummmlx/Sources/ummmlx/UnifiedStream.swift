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
    /// A narrative visual beat. The server emits it as soon as the directive
    /// closes, before the image or video leg finishes.
    case visualSlot = "visual_slot"
    case artifact
    case legError = "leg_error"
    case complete
    case error
    case cancelled
    /// A kind this client does not know, added by a newer server. Its seq is
    /// consumed and the event is otherwise ignored; it is never terminal, so it
    /// cannot truncate a stream.
    case unknown

    init?(rawValue: String) {
        switch rawValue {
        case "asr_partial": self = .asrPartial
        case "asr_final": self = .asrFinal
        case "vlm_delta": self = .vlmDelta
        case "visual_slot": self = .visualSlot
        case "artifact": self = .artifact
        case "leg_error": self = .legError
        case "complete": self = .complete
        case "error": self = .error
        case "cancelled": self = .cancelled
        default: self = .unknown
        }
    }

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
    /// Narrative slot attribution. Present only for interleaved visual legs.
    let slotID: String?
    let slotIndex: Int?
    /// Optional cross-slot anchor metadata carried on resolved visual artifacts.
    let anchorSlotIndex: Int?
    let anchorExplicit: Bool
    let anchorSha256: String?

    init(artifactID: String, mediaType: String, mimeType: String,
         encoding: String, index: Int, total: Int, data: Data,
         sha256: String?, bytes: Int?, format: String? = nil,
         sampleRate: Int? = nil, channels: Int? = nil,
         slotID: String? = nil, slotIndex: Int? = nil,
         anchorSlotIndex: Int? = nil, anchorExplicit: Bool = false,
         anchorSha256: String? = nil) {
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
        self.slotID = slotID
        self.slotIndex = slotIndex
        self.anchorSlotIndex = anchorSlotIndex
        self.anchorExplicit = anchorExplicit
        self.anchorSha256 = anchorSha256
    }
}

struct AudioStreamChunk: Equatable, Sendable {
    let pcm16: Data
    let sampleRate: Int
    let channels: Int
}

/// One entry of the terminal `legs` summary array on `complete`/`cancelled`/`error`.
struct UnifiedLegSummary: Equatable {
    let stage: String
    let state: String
    let error: String?
    let slotID: String?
    let slotIndex: Int?
    let anchorSlotIndex: Int?
    let anchorExplicit: Bool
    let anchorSha256: String?

    init(stage: String, state: String, error: String? = nil,
         slotID: String? = nil, slotIndex: Int? = nil,
         anchorSlotIndex: Int? = nil, anchorExplicit: Bool = false,
         anchorSha256: String? = nil) {
        self.stage = stage
        self.state = state
        self.error = error
        self.slotID = slotID
        self.slotIndex = slotIndex
        self.anchorSlotIndex = anchorSlotIndex
        self.anchorExplicit = anchorExplicit
        self.anchorSha256 = anchorSha256
    }
}

/// One stage entry inside a server-returned opt-in `telemetry` envelope
/// (`telemetry.native_allocator=true`). `allocatedBytesPeak` stays `nil` when
/// the server reports `null` or `NOT_COVERED` — never fabricated as `0`.
struct NativeAllocatorStageTelemetry: Equatable {
    let stage: String
    let nativeStage: String?
    let backend: String?
    let coverage: String
    let allocatedBytesPeak: Int64?
    let reservedBytesPeak: Int64?
    let allocatedBytesFinal: Int64?
    let reservedBytesFinal: Int64?
    let epochID: Int64?
    let reason: String?
    let lifecycleStatus: String?

    init(stage: String, nativeStage: String? = nil, backend: String? = nil,
         coverage: String, allocatedBytesPeak: Int64? = nil,
         reservedBytesPeak: Int64? = nil, allocatedBytesFinal: Int64? = nil,
         reservedBytesFinal: Int64? = nil, epochID: Int64? = nil,
         reason: String? = nil, lifecycleStatus: String? = nil) {
        self.stage = stage
        self.nativeStage = nativeStage
        self.backend = backend
        self.coverage = coverage
        self.allocatedBytesPeak = allocatedBytesPeak
        self.reservedBytesPeak = reservedBytesPeak
        self.allocatedBytesFinal = allocatedBytesFinal
        self.reservedBytesFinal = reservedBytesFinal
        self.epochID = epochID
        self.reason = reason
        self.lifecycleStatus = lifecycleStatus
    }
}

/// Bounded request-native CUDA allocator telemetry envelope returned on
/// `unified.telemetry` (or top-level `telemetry` on non-unified responses).
struct NativeAllocatorTelemetryEnvelope: Equatable {
    let schemaVersion: Int?
    let coverage: String
    let scope: String?
    let requestID: String?
    let lifecycleStatus: String?
    let stages: [NativeAllocatorStageTelemetry]
    let truncated: Bool

    init(schemaVersion: Int? = 1, coverage: String, scope: String? = nil,
         requestID: String? = nil, lifecycleStatus: String? = nil,
         stages: [NativeAllocatorStageTelemetry], truncated: Bool = false) {
        self.schemaVersion = schemaVersion
        self.coverage = coverage
        self.scope = scope
        self.requestID = requestID
        self.lifecycleStatus = lifecycleStatus
        self.stages = stages
        self.truncated = truncated
    }

    var summary: NativeAllocatorTelemetrySummary {
        NativeAllocatorTelemetrySummary.from(self)
    }
}

/// Honest summary of a `NativeAllocatorTelemetryEnvelope` for display in the
/// per-turn telemetry strip (`VRAM: <peak> peak` vs `VRAM: N/C`).
struct NativeAllocatorTelemetrySummary: Equatable {
    let covered: Bool
    let coverage: String
    let coveredStageCount: Int
    let totalStageCount: Int
    let peakBytes: Int64?
    let label: String
    let help: String

    static func formatBytes(_ bytes: Int64) -> String? {
        guard bytes >= 0 else { return nil }
        let kib: Double = 1024
        let mib: Double = 1024 * 1024
        let gib: Double = 1024 * 1024 * 1024
        func fmt(_ val: Double, _ unit: String) -> String {
            let rounded = (val * 10.0).rounded() / 10.0
            if rounded.truncatingRemainder(dividingBy: 1.0) == 0 {
                return "\(Int64(rounded)) \(unit)"
            }
            return String(format: "%.1f %@", rounded, unit)
        }
        let value = Double(bytes)
        if value >= gib { return fmt(value / gib, "GiB") }
        if value >= mib { return fmt(value / mib, "MiB") }
        if value >= kib { return fmt(value / kib, "KiB") }
        return "\(bytes) B"
    }

    static func from(_ envelope: NativeAllocatorTelemetryEnvelope) -> NativeAllocatorTelemetrySummary {
        var coveredCount = 0
        var peak: Int64? = nil
        var detailLines: [String] = []
        for entry in envelope.stages {
            let stage = entry.stage
            let nativeStage = (entry.nativeStage?.isEmpty == false && entry.nativeStage != "unknown")
                ? entry.nativeStage : nil
            let stageName: String
            if !stage.isEmpty, let nativeStage, nativeStage != stage {
                stageName = "\(stage)/\(nativeStage)"
            } else if let nativeStage {
                stageName = nativeStage
            } else if !stage.isEmpty {
                stageName = stage
            } else {
                stageName = "stage"
            }
            if entry.coverage == "COVERED",
               let bytes = entry.allocatedBytesPeak,
               bytes >= 0,
               let formattedPeak = formatBytes(bytes) {
                coveredCount += 1
                peak = peak.map { max($0, bytes) } ?? bytes
                let epochSuffix = entry.epochID.flatMap { $0 > 0 ? " · epoch #\($0)" : nil } ?? ""
                detailLines.append("\(stageName): COVERED · peak \(formattedPeak)\(epochSuffix)")
            } else {
                let reasonSuffix = (entry.reason?.isEmpty == false) ? " (\(entry.reason!))" : ""
                detailLines.append("\(stageName): NOT_COVERED\(reasonSuffix)")
            }
        }
        if coveredCount > 0, let peak, let formatted = formatBytes(peak) {
            let count = envelope.stages.count
            let header = "Native CUDA allocator (\(coveredCount)/\(count) stage\(count == 1 ? "" : "s") covered)"
            return NativeAllocatorTelemetrySummary(
                covered: true,
                coverage: coveredCount == count ? "COVERED" : "PARTIAL",
                coveredStageCount: coveredCount,
                totalStageCount: count,
                peakBytes: peak,
                label: "VRAM: \(formatted) peak",
                help: ([header] + detailLines).joined(separator: "\n")
            )
        }
        let header = "Native CUDA allocator (NOT_COVERED)"
        let lines = detailLines.isEmpty ? ["no stages reported"] : detailLines
        return NativeAllocatorTelemetrySummary(
            covered: false,
            coverage: "NOT_COVERED",
            coveredStageCount: 0,
            totalStageCount: envelope.stages.count,
            peakBytes: nil,
            label: "VRAM: N/C",
            help: ([header] + lines).joined(separator: "\n")
        )
    }
}

/// Server-returned opt-in sampled behavior token/log-probability evidence from
/// the final SSE chunk (`"behavior": {...}`). Never fabricated when omitted or
/// `not_exposed`.
struct BehaviorLogprobsEvidence: Equatable {
    let format: String?
    let tokenizerSha256: String?
    let tokenIDs: [Int64]
    let logprobs: [Double]?
    let mask: [Int]

    init(format: String? = nil, tokenizerSha256: String? = nil,
         tokenIDs: [Int64], logprobs: [Double]? = nil, mask: [Int]) {
        self.format = format
        self.tokenizerSha256 = tokenizerSha256
        self.tokenIDs = tokenIDs
        self.logprobs = logprobs
        self.mask = mask
    }

    var supervisedCount: Int {
        mask.reduce(0) { $0 + ($1 == 1 ? 1 : 0) }
    }
}

/// Honest summary of opt-in behavior logprobs evidence for display in the
/// per-turn telemetry strip (`Behavior: N tok (M sup)` vs
/// `Behavior: not_exposed` / `Behavior: absent`).
struct BehaviorLogprobsSummary: Equatable {
    let status: String
    let tokenCount: Int?
    let supervisedCount: Int?
    let label: String
    let help: String

    static func normalizeHeader(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return trimmed.isEmpty ? nil : trimmed
    }

    static func from(
        requested: Bool,
        header: String?,
        evidence: BehaviorLogprobsEvidence?
    ) -> BehaviorLogprobsSummary? {
        let normHeader = normalizeHeader(header)
        let validEvidence: BehaviorLogprobsEvidence?
        if let evidence,
           evidence.tokenIDs.count == evidence.mask.count,
           evidence.mask.allSatisfy({ $0 == 0 || $0 == 1 }),
           evidence.logprobs.map({ $0.count == evidence.tokenIDs.count && $0.allSatisfy(\.isFinite) }) ?? true {
            validEvidence = evidence
        } else {
            validEvidence = nil
        }

        if !requested && normHeader == nil && validEvidence == nil {
            return nil
        }
        if normHeader == "exposed", let validEvidence {
            let total = validEvidence.tokenIDs.count
            let supervised = validEvidence.supervisedCount
            var lines = ["Behavior logprobs: exposed"]
            if let format = validEvidence.format, !format.isEmpty {
                lines.append("Format: \(format)")
            }
            lines.append("Tokens: \(total) (\(supervised) supervised)")
            if let sha = validEvidence.tokenizerSha256, !sha.isEmpty {
                lines.append("Tokenizer SHA-256: \(sha)")
            }
            return BehaviorLogprobsSummary(
                status: "exposed",
                tokenCount: total,
                supervisedCount: supervised,
                label: "Behavior: \(total) tok (\(supervised) sup)",
                help: lines.joined(separator: "\n")
            )
        }
        if normHeader == "not_exposed" {
            return BehaviorLogprobsSummary(
                status: "not_exposed",
                tokenCount: nil,
                supervisedCount: nil,
                label: "Behavior: not_exposed",
                help: "Behavior logprobs: not_exposed (no behavior payload)"
            )
        }
        guard let normHeader else {
            guard requested else { return nil }
            return BehaviorLogprobsSummary(
                status: "absent",
                tokenCount: nil,
                supervisedCount: nil,
                label: "Behavior: absent",
                help: "Behavior logprobs requested, but X-UmmInfer-Behavior-Evidence header was absent"
            )
        }
        let help = normHeader == "exposed"
            ? "Behavior logprobs: exposed header received without a valid behavior payload"
            : "Behavior logprobs: \(normHeader) (no valid exposed payload)"
        return BehaviorLogprobsSummary(
            status: "not_exposed",
            tokenCount: nil,
            supervisedCount: nil,
            label: "Behavior: not_exposed",
            help: help
        )
    }
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
    /// Narrative slot attribution, present on `visual_slot`, and on `leg_error`
    /// and `artifact` events of a visual leg.
    let slotID: String?
    let slotIndex: Int?
    /// "image" or "video" for a visual slot; empty otherwise.
    let mediaType: String
    /// ASCII visual scene description of a visual slot; empty otherwise.
    let prompt: String
    /// Prose length at which the directive appeared.
    let textOffset: Int?
    /// Optional cross-slot continuity mode on `visual_slot` events
    /// (e.g. "i2v_chain" or "prompt_anchor").
    let continuityMode: String?
    /// Optional 0-based anchor slot index on `visual_slot`/`leg_error` events when chained.
    let anchorSlotIndex: Int?
    /// True when the slot used an explicit `source=slot-N` anchor directive.
    let anchorExplicit: Bool
    /// Optional verified 64-char hex SHA-256 digest of the resolved anchor image.
    let anchorSha256: String?
    /// Optional terminal leg summaries carried on `complete`/`cancelled`/`error`.
    let legs: [UnifiedLegSummary]
    /// Optional opt-in request-native CUDA allocator telemetry envelope.
    let telemetry: NativeAllocatorTelemetryEnvelope?

    init(kind: UnifiedKind, seq: Int, sessionID: String, stage: String,
         isFinal: Bool, text: String, error: String,
         artifact: ArtifactChunk? = nil, slotID: String? = nil,
         slotIndex: Int? = nil, mediaType: String = "", prompt: String = "",
         textOffset: Int? = nil, continuityMode: String? = nil,
         anchorSlotIndex: Int? = nil, anchorExplicit: Bool = false,
         anchorSha256: String? = nil, legs: [UnifiedLegSummary] = [],
         telemetry: NativeAllocatorTelemetryEnvelope? = nil) {
        self.kind = kind
        self.seq = seq
        self.sessionID = sessionID
        self.stage = stage
        self.isFinal = isFinal
        self.text = text
        self.error = error
        self.artifact = artifact
        self.slotID = slotID
        self.slotIndex = slotIndex
        self.mediaType = mediaType
        self.prompt = prompt
        self.textOffset = textOffset
        self.continuityMode = continuityMode
        self.anchorSlotIndex = anchorSlotIndex
        self.anchorExplicit = anchorExplicit
        self.anchorSha256 = anchorSha256
        self.legs = legs
        self.telemetry = telemetry
    }
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
    /// Narrative slot attribution carried from the wire, when present.
    let slotID: String?
    let slotIndex: Int?
    /// Optional cross-slot anchor metadata carried from the wire, when present.
    let anchorSlotIndex: Int?
    let anchorExplicit: Bool
    let anchorSha256: String?

    init(id: String, mediaType: String, mimeType: String, data: Data,
         sha256: String, sampleRate: Int? = nil, channels: Int? = nil,
         slotID: String? = nil, slotIndex: Int? = nil,
         anchorSlotIndex: Int? = nil, anchorExplicit: Bool = false,
         anchorSha256: String? = nil) {
        self.id = id
        self.mediaType = mediaType
        self.mimeType = mimeType
        self.data = data
        self.sha256 = sha256
        self.sampleRate = sampleRate
        self.channels = channels
        self.slotID = slotID
        self.slotIndex = slotIndex
        self.anchorSlotIndex = anchorSlotIndex
        self.anchorExplicit = anchorExplicit
        self.anchorSha256 = anchorSha256
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

    static func normalizedAnchorSHA256(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard trimmed.count == 64,
              trimmed.unicodeScalars.allSatisfy({ scalar in
                  (scalar.value >= 0x30 && scalar.value <= 0x39)
                      || (scalar.value >= 0x61 && scalar.value <= 0x66)
              }) else {
            return nil
        }
        return trimmed
    }

    private static func optionalAnchorExplicit(_ value: Any?, _ name: String) throws -> Bool {
        guard let value, !(value is NSNull) else { return false }
        guard let flag = value as? NSNumber,
              CFGetTypeID(flag) == CFBooleanGetTypeID() else {
            throw ChatClientError.invalidResponse("unified \(name) is not a boolean")
        }
        return flag.boolValue
    }

    private static func optionalAnchorSHA256(_ value: Any?, _ name: String) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        let raw = try string(value, name)
        return normalizedAnchorSHA256(raw)
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
        let kind = UnifiedKind(rawValue: kindText) ?? .unknown
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
        let slotID = try root["slot_id"].map { try string($0, "slot_id") }
        let slotIndex = try root["slot_index"].map { try integer($0, "slot_index") }
        let mediaType = try root["media_type"].map { try string($0, "media_type") } ?? ""
        let prompt = try root["prompt"].map { try string($0, "prompt") } ?? ""
        let textOffset = try root["text_offset"].map { try integer($0, "text_offset") }
        let continuityMode: String?
        if let rawMode = root["continuity_mode"], !(rawMode is NSNull) {
            continuityMode = try string(rawMode, "continuity_mode")
        } else {
            continuityMode = nil
        }
        let anchorSlotIndex: Int?
        if let rawAnchor = root["anchor_slot_index"], !(rawAnchor is NSNull) {
            anchorSlotIndex = try integer(rawAnchor, "anchor_slot_index")
        } else {
            anchorSlotIndex = nil
        }
        let anchorExplicit = try optionalAnchorExplicit(root["anchor_explicit"], "anchor_explicit")
        let anchorSha256 = try optionalAnchorSHA256(root["anchor_sha256"], "anchor_sha256")
        if kind == .visualSlot {
            guard slotID?.isEmpty == false, slotIndex != nil else {
                throw ChatClientError.invalidResponse(
                    "a visual_slot event requires slot_id and slot_index")
            }
        }

        var artifact: ArtifactChunk?
        if kind == .artifact {
            let raw = try object(root["artifact"])
            artifact = try decodeArtifact(raw)
        }
        var legs: [UnifiedLegSummary] = []
        if let rawLegs = root["legs"], !(rawLegs is NSNull) {
            guard let array = rawLegs as? [Any] else {
                throw ChatClientError.invalidResponse("unified legs is not an array")
            }
            legs = try array.map { try decodeLeg(try object($0)) }
        }
        var telemetry: NativeAllocatorTelemetryEnvelope? = nil
        if let rawTelemetry = root["telemetry"] as? [String: Any] {
            telemetry = decodeNativeAllocatorTelemetry(rawTelemetry)
        }
        return UnifiedEvent(
            kind: kind, seq: seq, sessionID: sessionID, stage: stage,
            isFinal: isFinal, text: text, error: errorText,
            artifact: artifact, slotID: slotID, slotIndex: slotIndex,
            mediaType: mediaType, prompt: prompt, textOffset: textOffset,
            continuityMode: continuityMode, anchorSlotIndex: anchorSlotIndex,
            anchorExplicit: anchorExplicit, anchorSha256: anchorSha256,
            legs: legs, telemetry: telemetry
        )
    }

    private static func optionalNonNegativeInt64(_ value: Any?) -> Int64? {
        guard let value, !(value is NSNull),
              let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number) else {
            return nil
        }
        let intVal = number.int64Value
        return intVal >= 0 ? intVal : nil
    }

    static func decodeNativeAllocatorTelemetry(
        _ raw: [String: Any]
    ) -> NativeAllocatorTelemetryEnvelope? {
        let rawStages = raw["stages"] as? [Any]
        let coverageText = raw["coverage"] as? String
        guard rawStages != nil || coverageText != nil else { return nil }
        var stages: [NativeAllocatorStageTelemetry] = []
        if let rawStages {
            for item in rawStages {
                guard let entry = item as? [String: Any] else { continue }
                let stage = (entry["stage"] as? String) ?? ""
                let nativeStage = entry["native_stage"] as? String
                let backend = entry["backend"] as? String
                let entryCoverage = (entry["coverage"] as? String) ?? "NOT_COVERED"
                let allocatedPeak = optionalNonNegativeInt64(entry["allocated_bytes_peak"])
                let reservedPeak = optionalNonNegativeInt64(entry["reserved_bytes_peak"])
                let allocatedFinal = optionalNonNegativeInt64(entry["allocated_bytes_final"])
                let reservedFinal = optionalNonNegativeInt64(entry["reserved_bytes_final"])
                let epochID: Int64?
                if let epochObj = entry["epoch"] as? [String: Any] {
                    epochID = optionalNonNegativeInt64(epochObj["id"])
                } else {
                    epochID = nil
                }
                let reason = entry["reason"] as? String
                let lifecycle = entry["lifecycle_status"] as? String
                stages.append(NativeAllocatorStageTelemetry(
                    stage: stage, nativeStage: nativeStage, backend: backend,
                    coverage: entryCoverage,
                    allocatedBytesPeak: allocatedPeak,
                    reservedBytesPeak: reservedPeak,
                    allocatedBytesFinal: allocatedFinal,
                    reservedBytesFinal: reservedFinal,
                    epochID: epochID, reason: reason, lifecycleStatus: lifecycle
                ))
            }
        }
        let schemaVersion = optionalNonNegativeInt64(raw["schema_version"]).map(Int.init)
        let scope = raw["scope"] as? String
        let requestID = raw["request_id"] as? String
        let lifecycleStatus = raw["lifecycle_status"] as? String
        let truncated: Bool
        if let sampleCov = raw["sample_coverage"] as? [String: Any],
           let flag = sampleCov["truncated"] as? NSNumber,
           CFGetTypeID(flag) == CFBooleanGetTypeID() {
            truncated = flag.boolValue
        } else {
            truncated = false
        }
        return NativeAllocatorTelemetryEnvelope(
            schemaVersion: schemaVersion,
            coverage: coverageText ?? "NOT_COVERED",
            scope: scope,
            requestID: requestID,
            lifecycleStatus: lifecycleStatus,
            stages: stages,
            truncated: truncated
        )
    }

    static func decodeBehaviorLogprobsEvidence(
        _ raw: [String: Any]
    ) -> BehaviorLogprobsEvidence? {
        guard let rawTokenIDs = raw["token_ids"] as? [Any],
              let rawMask = raw["mask"] as? [Any],
              rawTokenIDs.count == rawMask.count else {
            return nil
        }
        var tokenIDs: [Int64] = []
        tokenIDs.reserveCapacity(rawTokenIDs.count)
        for item in rawTokenIDs {
            guard let number = item as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  !CFNumberIsFloatType(number) else {
                return nil
            }
            tokenIDs.append(number.int64Value)
        }
        var mask: [Int] = []
        mask.reserveCapacity(rawMask.count)
        for item in rawMask {
            guard let number = item as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  !CFNumberIsFloatType(number) else {
                return nil
            }
            let val = number.intValue
            guard val == 0 || val == 1 else { return nil }
            mask.append(val)
        }
        var logprobs: [Double]? = nil
        if let rawLogprobs = raw["logprobs"], !(rawLogprobs is NSNull) {
            guard let array = rawLogprobs as? [Any],
                  array.count == tokenIDs.count else {
                return nil
            }
            var parsed: [Double] = []
            parsed.reserveCapacity(array.count)
            for item in array {
                guard let number = item as? NSNumber,
                      CFGetTypeID(number) != CFBooleanGetTypeID() else {
                    return nil
                }
                let val = number.doubleValue
                guard val.isFinite else { return nil }
                parsed.append(val)
            }
            logprobs = parsed
        }
        let format = raw["format"] as? String
        let tokenizerSha256 = raw["tokenizer_sha256"] as? String
        return BehaviorLogprobsEvidence(
            format: format,
            tokenizerSha256: tokenizerSha256,
            tokenIDs: tokenIDs,
            logprobs: logprobs,
            mask: mask
        )
    }

    private static func decodeLeg(_ raw: [String: Any]) throws -> UnifiedLegSummary {
        let stage: String
        if let value = raw["stage"], !(value is NSNull) {
            stage = try string(value, "legs.stage")
        } else {
            stage = ""
        }
        let state: String
        if let value = raw["state"], !(value is NSNull) {
            state = try string(value, "legs.state")
        } else {
            state = ""
        }
        let error: String?
        if let value = raw["error"], !(value is NSNull) {
            error = try string(value, "legs.error")
        } else {
            error = nil
        }
        let slotID: String?
        if let value = raw["slot_id"], !(value is NSNull) {
            slotID = try string(value, "legs.slot_id")
        } else {
            slotID = nil
        }
        let slotIndex: Int?
        if let value = raw["slot_index"], !(value is NSNull) {
            slotIndex = try integer(value, "legs.slot_index")
        } else {
            slotIndex = nil
        }
        let anchorSlotIndex: Int?
        if let value = raw["anchor_slot_index"], !(value is NSNull) {
            anchorSlotIndex = try integer(value, "legs.anchor_slot_index")
        } else {
            anchorSlotIndex = nil
        }
        let anchorExplicit = try optionalAnchorExplicit(raw["anchor_explicit"], "legs.anchor_explicit")
        let anchorSha256 = try optionalAnchorSHA256(raw["anchor_sha256"], "legs.anchor_sha256")
        return UnifiedLegSummary(
            stage: stage, state: state, error: error,
            slotID: slotID, slotIndex: slotIndex,
            anchorSlotIndex: anchorSlotIndex, anchorExplicit: anchorExplicit,
            anchorSha256: anchorSha256
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
        let slotID = try raw["slot_id"].map { try string($0, "artifact.slot_id") }
        let slotIndex = try raw["slot_index"].map {
            try integer($0, "artifact.slot_index")
        }
        let anchorSlotIndex: Int?
        if let rawAnchor = raw["anchor_slot_index"], !(rawAnchor is NSNull) {
            anchorSlotIndex = try integer(rawAnchor, "artifact.anchor_slot_index")
        } else {
            anchorSlotIndex = nil
        }
        let anchorExplicit = try optionalAnchorExplicit(raw["anchor_explicit"], "artifact.anchor_explicit")
        let anchorSha256 = try optionalAnchorSHA256(raw["anchor_sha256"], "artifact.anchor_sha256")
        return ArtifactChunk(
            artifactID: artifactID, mediaType: mediaType, mimeType: mimeType,
            encoding: encoding, index: index, total: total, data: decoded,
            sha256: sha, bytes: bytes, format: format,
            sampleRate: sampleRate, channels: channels,
            slotID: slotID, slotIndex: slotIndex,
            anchorSlotIndex: anchorSlotIndex, anchorExplicit: anchorExplicit,
            anchorSha256: anchorSha256
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
        var slotID: String?
        var slotIndex: Int?
        var anchorSlotIndex: Int?
        var anchorExplicit: Bool
        var anchorSha256: String?
        var nextIndex: Int
        var total: Int
        var data: Data
    }

    static let supportedMimeTypes: Set<String> = [
        "image/png", "image/jpeg", "audio/wav", "audio/pcm", "video/mp4"
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
        guard chunk.mediaType == "image" || chunk.mediaType == "audio" || chunk.mediaType == "video" else {
            throw ChatClientError.invalidResponse(
                "artifact media_type must be image, audio or video")
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
            slotID: chunk.slotID, slotIndex: chunk.slotIndex,
            anchorSlotIndex: chunk.anchorSlotIndex,
            anchorExplicit: chunk.anchorExplicit,
            anchorSha256: chunk.anchorSha256,
            nextIndex: 0, total: chunk.total, data: Data()
        )
        if slot.anchorSlotIndex == nil, let anchor = chunk.anchorSlotIndex {
            slot.anchorSlotIndex = anchor
        }
        if chunk.anchorExplicit {
            slot.anchorExplicit = true
        }
        if slot.anchorSha256 == nil, let sha = chunk.anchorSha256 {
            slot.anchorSha256 = sha
        }
        guard slot.mediaType == chunk.mediaType, slot.mimeType == chunk.mimeType else {
            throw ChatClientError.invalidResponse(
                "artifact descriptor changed between chunks")
        }
        if let slotID = chunk.slotID, slot.slotID != nil, slotID != slot.slotID {
            throw ChatClientError.invalidResponse(
                "artifact slot_id changed between chunks")
        }
        if let slotIndex = chunk.slotIndex, slot.slotIndex != nil,
           slotIndex != slot.slotIndex {
            throw ChatClientError.invalidResponse(
                "artifact slot_index changed between chunks")
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
            sampleRate: pcmSampleRate, channels: pcmChannels,
            slotID: slot.slotID ?? chunk.slotID,
            slotIndex: slot.slotIndex ?? chunk.slotIndex,
            anchorSlotIndex: slot.anchorSlotIndex ?? chunk.anchorSlotIndex,
            anchorExplicit: slot.anchorExplicit || chunk.anchorExplicit,
            anchorSha256: slot.anchorSha256 ?? chunk.anchorSha256
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
