// SPDX-License-Identifier: Apache-2.0
// Pure client-side session export (JSON + Markdown) for macOS ummmlx.
// Zero network calls, zero binary/base64 media bloat, zero fabricated metrics.

import CoreFoundation
import Foundation

enum SessionExportFormat: String, CaseIterable, Equatable, Sendable {
    case json
    case markdown

    var fileExtension: String {
        switch self {
        case .json: return "json"
        case .markdown: return "md"
        }
    }
}

struct SessionExportVisualSlot: Equatable, Encodable, Sendable {
    let slotID: String
    let slotIndex: Int
    let kind: String
    let prompt: String?
    let bytes: Int?
    let sha256: String?
    let continuityMode: String?
    let anchorSlotIndex: Int?
    let anchorExplicit: Bool?
    let anchorSha256: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case slotID = "slot_id"
        case slotIndex = "slot_index"
        case kind
        case prompt
        case bytes
        case sha256
        case continuityMode = "continuity_mode"
        case anchorSlotIndex = "anchor_slot_index"
        case anchorExplicit = "anchor_explicit"
        case anchorSha256 = "anchor_sha256"
        case error
    }

    var isChainedBadge: Bool {
        guard let anchorSlotIndex, anchorSlotIndex >= 0 else { return false }
        return continuityMode == NarrativeContinuityMode.i2vChain.rawValue
            || continuityMode == nil
            || continuityMode?.isEmpty == true
            || anchorExplicit == true
    }

    var continuityBadgeText: String? {
        if isChainedBadge, let anchorSlotIndex {
            let slotLabel = "Slot \(anchorSlotIndex + 1)"
            if anchorExplicit == true {
                if let sha = UnifiedDecode.normalizedAnchorSHA256(anchorSha256) {
                    return "Explicit I2V ← \(slotLabel) · \(sha.prefix(8))"
                }
                return "Explicit I2V ← \(slotLabel)"
            }
            return "I2V ← \(slotLabel)"
        }
        if anchorExplicit == true {
            if let sha = UnifiedDecode.normalizedAnchorSHA256(anchorSha256) {
                return "Explicit I2V · \(sha.prefix(8))"
            }
            return "Explicit I2V"
        }
        if continuityMode == NarrativeContinuityMode.promptAnchor.rawValue {
            return "Anchor"
        }
        return nil
    }
}

struct SessionExportNativeAllocator: Equatable, Encodable, Sendable {
    let coverage: String
    let covered: Bool
    let coveredStageCount: Int
    let totalStageCount: Int
    let peakBytes: Int64?
    let label: String
    let details: [String]

    enum CodingKeys: String, CodingKey {
        case coverage
        case covered
        case coveredStageCount = "covered_stage_count"
        case totalStageCount = "total_stage_count"
        case peakBytes = "peak_bytes"
        case label
        case details
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(coverage, forKey: .coverage)
        try container.encode(covered, forKey: .covered)
        try container.encode(coveredStageCount, forKey: .coveredStageCount)
        try container.encode(totalStageCount, forKey: .totalStageCount)
        // Explicitly encode null when NOT_COVERED — never fabricate 0.
        try container.encode(peakBytes, forKey: .peakBytes)
        try container.encode(label, forKey: .label)
        try container.encode(details, forKey: .details)
    }

    var displayLabel: String {
        if covered, let peakBytes, peakBytes >= 0,
           let formatted = NativeAllocatorTelemetrySummary.formatBytes(peakBytes) {
            return "VRAM: \(formatted) peak"
        }
        return "VRAM: N/C"
    }

    var helpText: String {
        let header: String
        if covered {
            header = "Native CUDA allocator (\(coveredStageCount)/\(totalStageCount) stage\(totalStageCount == 1 ? "" : "s") covered)"
        } else {
            header = "Native CUDA allocator (NOT_COVERED)"
        }
        let lines = details.isEmpty ? (covered ? [] : ["no stages reported"]) : details
        return ([header] + lines).joined(separator: "\n")
    }
}

struct SessionExportBehaviorLogprobs: Equatable, Encodable, Sendable {
    let status: String
    let tokenCount: Int?
    let supervisedCount: Int?
    let label: String
    let format: String?
    let tokenizerSha256: String?

    enum CodingKeys: String, CodingKey {
        case status
        case tokenCount = "token_count"
        case supervisedCount = "supervised_count"
        case label
        case format
        case tokenizerSha256 = "tokenizer_sha256"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(status, forKey: .status)
        // Explicitly encode null when not_exposed/absent — never fabricate 0.
        try container.encode(tokenCount, forKey: .tokenCount)
        try container.encode(supervisedCount, forKey: .supervisedCount)
        try container.encode(label, forKey: .label)
        try container.encodeIfPresent(format, forKey: .format)
        try container.encodeIfPresent(tokenizerSha256, forKey: .tokenizerSha256)
    }

    var displayLabel: String {
        if status == "exposed", let tokenCount, let supervisedCount {
            return "Behavior: \(tokenCount) tok (\(supervisedCount) sup)"
        }
        return "Behavior: \(status)"
    }

    var helpText: String {
        if status == "exposed", let tokenCount, let supervisedCount {
            var lines = ["Behavior logprobs: exposed"]
            if let format, !format.isEmpty {
                lines.append("Format: \(format)")
            }
            lines.append("Tokens: \(tokenCount) (\(supervisedCount) supervised)")
            if let tokenizerSha256, !tokenizerSha256.isEmpty {
                lines.append("Tokenizer SHA-256: \(tokenizerSha256)")
            }
            return lines.joined(separator: "\n")
        }
        if status == "absent" {
            return "Behavior logprobs requested, but X-UmmInfer-Behavior-Evidence header was absent"
        }
        return "Behavior logprobs: not_exposed (no behavior payload)"
    }
}

struct SessionExportTelemetry: Equatable, Encodable, Sendable {
    let ttftMs: Double?
    let ttfaMs: Double?
    let firstVisualMs: Double?
    let totalMs: Double?
    let visualSlots: Int
    let i2vCount: Int
    let nativeAllocator: SessionExportNativeAllocator?
    let behaviorLogprobs: SessionExportBehaviorLogprobs?

    enum CodingKeys: String, CodingKey {
        case ttftMs = "ttft_ms"
        case ttfaMs = "ttfa_ms"
        case firstVisualMs = "first_visual_ms"
        case totalMs = "total_ms"
        case visualSlots = "visual_slots"
        case i2vCount = "i2v_count"
        case nativeAllocator = "native_allocator"
        case behaviorLogprobs = "behavior_logprobs"
    }
}

struct SessionExportTurn: Equatable, Encodable, Sendable {
    let index: Int
    let prompt: String
    let response: String
    let asrText: String?
    let interrupted: Bool?
    let spokenBytes: Int?
    let visualSlots: [SessionExportVisualSlot]
    let telemetry: SessionExportTelemetry?

    enum CodingKeys: String, CodingKey {
        case index
        case prompt
        case response
        case asrText = "asr_text"
        case interrupted
        case spokenBytes = "spoken_bytes"
        case visualSlots = "visual_slots"
        case telemetry
    }
}

struct SessionExportReport: Equatable, Encodable, Sendable {
    let schema: String
    let exportedAt: String
    let sessionID: String?
    let model: String?
    let turnCount: Int
    let turns: [SessionExportTurn]

    enum CodingKeys: String, CodingKey {
        case schema
        case exportedAt = "exported_at"
        case sessionID = "session_id"
        case model
        case turnCount = "turn_count"
        case turns
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schema, forKey: .schema)
        try container.encode(exportedAt, forKey: .exportedAt)
        try container.encode(sessionID, forKey: .sessionID)
        try container.encode(model, forKey: .model)
        try container.encode(turnCount, forKey: .turnCount)
        try container.encode(turns, forKey: .turns)
    }
}

enum SessionImportError: Error, LocalizedError, Equatable, Sendable {
    case emptyPayload
    case payloadTooLarge(bytes: Int, limit: Int)
    case invalidJSON(String)
    case invalidRootObject
    case unsupportedSchema(String?)
    case missingConversationArray
    case invalidField(String)
    case invalidRole(String)
    case invalidTurn(index: Int, reason: String)
    case sessionBusy(String)

    var errorDescription: String? {
        switch self {
        case .emptyPayload:
            return "Session file is empty."
        case .payloadTooLarge(let bytes, let limit):
            return "Session file (\(bytes) bytes) exceeds the \(limit / (1024 * 1024)) MiB import limit."
        case .invalidJSON(let detail):
            return "Invalid JSON: \(detail)"
        case .invalidRootObject:
            return "Session file root must be a JSON object."
        case .unsupportedSchema(let found):
            if let found, !found.isEmpty {
                return "Unsupported session export schema '\(found)'."
            }
            return "Missing or invalid session export schema."
        case .missingConversationArray:
            return "Session file must contain a 'turns' or 'messages' array."
        case .invalidField(let detail):
            return "Invalid session field: \(detail)."
        case .invalidRole(let role):
            return "Unsupported message role '\(role)'."
        case .invalidTurn(let index, let reason):
            return "Invalid entry at index \(index): \(reason)."
        case .sessionBusy(let reason):
            return reason
        }
    }
}

struct ImportedSession {
    let schema: String
    let exportedAt: String?
    let sessionID: String?
    let model: String?
    let turnCount: Int
    let turns: [ChatTurn]
    let messages: [ChatMessage]
}

enum SessionExporter {
    static let schema = "umminfer-session-export-v1"
    static let webSchema = "umminfer-web-session-export-v1"
    static let supportedImportSchemas: Set<String> = [schema, webSchema]
    static let maxImportBytes = 2 * 1024 * 1024

    /// Sanitizes an arbitrary session identifier into a safe filename token
    /// (`[A-Za-z0-9_-]`, max 64 chars, no path traversal or leading/trailing
    /// separators), falling back to `"untitled"` when empty.
    static func sanitizeSessionID(_ id: String) -> String {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        var collapsed = ""
        collapsed.reserveCapacity(min(trimmed.count, 64))
        var lastWasDash = false
        for scalar in trimmed.unicodeScalars {
            let isASCIIAlphaNum =
                (scalar.value >= 0x30 && scalar.value <= 0x39) ||
                (scalar.value >= 0x41 && scalar.value <= 0x5A) ||
                (scalar.value >= 0x61 && scalar.value <= 0x7A)
            let isUnderscore = scalar.value == 0x5F
            let isDash = scalar.value == 0x2D
            if isASCIIAlphaNum || isUnderscore {
                collapsed.unicodeScalars.append(scalar)
                lastWasDash = false
            } else if isDash || !lastWasDash {
                collapsed.append("-")
                lastWasDash = true
            }
        }
        let stripEdge: (Character) -> Bool = { $0 == "-" || $0 == "_" }
        let stripped = String(
            collapsed
                .drop(while: stripEdge)
                .reversed()
                .drop(while: stripEdge)
                .reversed()
        )
        let bounded = String(stripped.prefix(64))
        let finalToken = String(bounded.reversed().drop(while: stripEdge).reversed())
        return finalToken.isEmpty ? "untitled" : finalToken
    }

    static func defaultFilename(sessionID: String, format: SessionExportFormat) -> String {
        "umminfer-session-\(sanitizeSessionID(sessionID)).\(format.fileExtension)"
    }

    static func formatISO8601UTC(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    /// Strips any inline `data:<mime>;base64,...` blobs from free-text fields so
    /// exports never carry raw base64 media payloads.
    static func scrubInlineDataURIs(_ text: String) -> String {
        guard text.localizedCaseInsensitiveContains(";base64,") else { return text }
        let pattern = #"data:[a-z0-9.+-]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return text
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "[omitted-base64]")
    }

    static func buildData(
        format: SessionExportFormat,
        sessionID: String,
        model: String,
        turns: [ChatTurn],
        exportedAt: Date = Date()
    ) -> Data {
        switch format {
        case .json:
            return buildJSONData(sessionID: sessionID, model: model, turns: turns, exportedAt: exportedAt)
        case .markdown:
            return Data(buildMarkdown(sessionID: sessionID, model: model, turns: turns, exportedAt: exportedAt).utf8)
        }
    }

    static func buildJSONData(
        sessionID: String,
        model: String,
        turns: [ChatTurn],
        exportedAt: Date = Date()
    ) -> Data {
        let (report, _) = buildReportWithPills(
            sessionID: sessionID,
            model: model,
            turns: turns,
            exportedAt: exportedAt
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = (try? encoder.encode(report)) ?? Data("{}".utf8)
        data.append(0x0A)
        return data
    }

    static func buildMarkdown(
        sessionID: String,
        model: String,
        turns: [ChatTurn],
        exportedAt: Date = Date()
    ) -> String {
        let (report, perTurnPills) = buildReportWithPills(
            sessionID: sessionID,
            model: model,
            turns: turns,
            exportedAt: exportedAt
        )

        var lines: [String] = [
            "# ummInfer Session Export",
            "",
            "- **Session ID:** \(report.sessionID.map { "`\($0)`" } ?? "_none_")",
        ]
        if let modelName = report.model {
            lines.append("- **Model:** `\(modelName)`")
        }
        lines.append("- **Exported at:** `\(report.exportedAt)`")
        lines.append("- **Turns:** \(report.turnCount)")

        if report.turnCount == 0 {
            lines.append(contentsOf: ["", "_No turns recorded._"])
            return lines.joined(separator: "\n") + "\n"
        }

        for (offset, turn) in report.turns.enumerated() {
            lines.append(contentsOf: ["", "---", "", "## Turn \(turn.index)"])

            if !turn.prompt.isEmpty {
                lines.append(contentsOf: ["", "### User", "", turn.prompt])
            }
            if let asrText = turn.asrText, !asrText.isEmpty {
                lines.append(contentsOf: ["", "> _ASR:_ \(asrText)"])
            }

            let hasAssistantSection = !turn.response.isEmpty
                || turn.interrupted == true
                || !turn.visualSlots.isEmpty
                || turn.telemetry != nil
            if hasAssistantSection {
                var heading = "### Assistant"
                if turn.interrupted == true {
                    let badge = ChatTurn.formatInterruptedBadge(acknowledgedBytes: turn.spokenBytes)
                    heading += " · \(badge ?? "Interrupted")"
                }
                lines.append(contentsOf: ["", heading])
                if !turn.response.isEmpty {
                    lines.append(contentsOf: ["", turn.response])
                }
            }

            if !turn.visualSlots.isEmpty {
                lines.append(contentsOf: ["", "#### Visual Slots", ""])
                for slot in turn.visualSlots {
                    let kindLabel = slot.kind == "video" ? "Video" : "Image"
                    var headerParts = ["**\(kindLabel)**"]
                    if let badge = slot.continuityBadgeText {
                        headerParts.append("`\(badge)`")
                    }
                    var metaParts: [String] = []
                    if let bytes = slot.bytes,
                       let formattedBytes = NativeAllocatorTelemetrySummary.formatBytes(Int64(bytes)) {
                        metaParts.append("`\(formattedBytes)`")
                    }
                    if let sha = slot.sha256 {
                        metaParts.append("`sha256: \(sha.prefix(8))`")
                    }
                    if let error = slot.error, !error.isEmpty {
                        metaParts.append("error: \(error)")
                    }
                    var item = "- `\(slot.slotID)` (\(headerParts.joined(separator: ", ")))"
                    if let prompt = slot.prompt, !prompt.isEmpty {
                        item += " — \(prompt)"
                    }
                    if !metaParts.isEmpty {
                        item += " (\(metaParts.joined(separator: ", ")))"
                    }
                    lines.append(item)
                }
            }

            let pills = offset < perTurnPills.count ? perTurnPills[offset] : []
            if !pills.isEmpty {
                let pillText = pills.map { "`\($0.label)`" }.joined(separator: " · ")
                lines.append(contentsOf: ["", "**Telemetry:** \(pillText)"])
            }
        }

        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - Internal normalization

    private static func normalizeObservedMs(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }

    private static func pairTurns(_ turns: [ChatTurn]) -> [(user: ChatTurn?, assistant: ChatTurn?)] {
        var pairs: [(user: ChatTurn?, assistant: ChatTurn?)] = []
        var pendingUser: ChatTurn? = nil
        for turn in turns {
            switch turn.role {
            case .user:
                if let previous = pendingUser {
                    pairs.append((user: previous, assistant: nil))
                }
                pendingUser = turn
            case .assistant:
                pairs.append((user: pendingUser, assistant: turn))
                pendingUser = nil
            }
        }
        if let trailing = pendingUser {
            pairs.append((user: trailing, assistant: nil))
        }
        return pairs
    }

    private static func extractResponseText(from assistant: ChatTurn?) -> String {
        guard let assistant else { return "" }
        if !assistant.text.isEmpty {
            return scrubInlineDataURIs(assistant.text)
        }
        let blockProse = assistant.blocks.compactMap { block -> String? in
            guard case .text(_, let content) = block, !content.isEmpty else { return nil }
            return content
        }
        return scrubInlineDataURIs(blockProse.joined(separator: "\n\n"))
    }

    private static func extractVisualSlots(from assistant: ChatTurn?) -> [SessionExportVisualSlot] {
        guard let assistant else { return [] }
        var slots: [SessionExportVisualSlot] = []

        for block in assistant.blocks {
            guard case .visual(let slot) = block else { continue }
            let kind = slot.activeKind == "video" ? "video" : "image"
            let artifact = kind == "video" ? slot.videoArtifact : slot.imageArtifact
            let prompt = scrubInlineDataURIs(slot.prompt.trimmingCharacters(in: .whitespacesAndNewlines))
            let continuity = slot.continuityMode?.trimmingCharacters(in: .whitespacesAndNewlines)
            let anchorIndex = slot.anchorSlotIndex.flatMap { $0 >= 0 ? $0 : nil }
            let anchorExplicit: Bool? = slot.anchorExplicit ? true : (anchorIndex != nil ? false : nil)
            let anchorSha = UnifiedDecode.normalizedAnchorSHA256(slot.anchorSha256)
            let artifactSha = artifact.flatMap { UnifiedDecode.normalizedAnchorSHA256($0.sha256) }
                ?? UnifiedDecode.normalizedAnchorSHA256(slot.importedSha256)
            let errorText = slot.error?.trimmingCharacters(in: .whitespacesAndNewlines)

            slots.append(SessionExportVisualSlot(
                slotID: slot.slotID.isEmpty ? "slot-\(slot.slotIndex)" : slot.slotID,
                slotIndex: slot.slotIndex,
                kind: kind,
                prompt: prompt.isEmpty ? nil : prompt,
                bytes: artifact.map { $0.data.count } ?? slot.importedBytes,
                sha256: artifactSha,
                continuityMode: (continuity?.isEmpty == false) ? continuity : nil,
                anchorSlotIndex: anchorIndex,
                anchorExplicit: anchorExplicit,
                anchorSha256: anchorSha,
                error: (errorText?.isEmpty == false) ? errorText : nil
            ))
        }

        // Non-slot visual artifacts (when narrative mode was off).
        var nextIndex = slots.count
        for artifact in assistant.artifacts {
            guard artifact.slotID == nil,
                  artifact.mediaType == "image" || artifact.mediaType == "video" else {
                continue
            }
            let kind = artifact.mediaType == "video" ? "video" : "image"
            let slotIndex = artifact.slotIndex ?? nextIndex
            nextIndex += 1
            let anchorIndex = artifact.anchorSlotIndex.flatMap { $0 >= 0 ? $0 : nil }
            let anchorExplicit: Bool? = artifact.anchorExplicit ? true : (anchorIndex != nil ? false : nil)
            let anchorSha = UnifiedDecode.normalizedAnchorSHA256(artifact.anchorSha256)
            let artifactSha = UnifiedDecode.normalizedAnchorSHA256(artifact.sha256)

            slots.append(SessionExportVisualSlot(
                slotID: artifact.id.isEmpty ? "slot-\(slotIndex)" : artifact.id,
                slotIndex: slotIndex,
                kind: kind,
                prompt: nil,
                bytes: artifact.data.count,
                sha256: artifactSha,
                continuityMode: nil,
                anchorSlotIndex: anchorIndex,
                anchorExplicit: anchorExplicit,
                anchorSha256: anchorSha,
                error: nil
            ))
        }

        return slots
    }

    private static func buildTelemetry(
        from raw: TurnTelemetry?,
        visualSlots: [SessionExportVisualSlot]
    ) -> (SessionExportTelemetry?, [TurnTelemetry.Pill]) {
        let src = raw ?? TurnTelemetry()
        let ttftMs = normalizeObservedMs(src.ttftMs)
        let ttfaMs = normalizeObservedMs(src.ttfaMs)
        let firstVisualMs = normalizeObservedMs(src.firstVisualMs)
        let totalMs = normalizeObservedMs(src.totalMs)

        let visualCount = max(max(src.visualSettledCount, 0), visualSlots.count)

        var slotI2VCount = 0
        var slotHasExplicitAnchor = false
        for slot in visualSlots {
            let isI2V = (slot.anchorSlotIndex != nil)
                || slot.continuityMode == NarrativeContinuityMode.i2vChain.rawValue
                || slot.anchorExplicit == true
            if isI2V { slotI2VCount += 1 }
            if slot.anchorExplicit == true { slotHasExplicitAnchor = true }
        }
        let hasExplicitFlag = slotHasExplicitAnchor || src.hasExplicitAnchor
        let hasI2VFlag = slotI2VCount > 0 || src.hasI2VChain || hasExplicitFlag
        let i2vCount = slotI2VCount > 0 ? slotI2VCount : (hasI2VFlag ? 1 : 0)

        let vramOut: SessionExportNativeAllocator? = src.nativeAllocator.map { envelope in
            let summary = envelope.summary
            let detailLines = Array(
                summary.help
                    .split(separator: "\n", omittingEmptySubsequences: false)
                    .dropFirst()
                    .map(String.init)
            )
            return SessionExportNativeAllocator(
                coverage: summary.coverage,
                covered: summary.covered,
                coveredStageCount: summary.coveredStageCount,
                totalStageCount: summary.totalStageCount,
                peakBytes: summary.covered ? summary.peakBytes : nil,
                label: summary.label,
                details: detailLines
            )
        } ?? src.importedNativeAllocator

        let behaviorSummary = BehaviorLogprobsSummary.from(
            requested: src.behaviorRequested,
            header: src.behaviorEvidenceHeader,
            evidence: src.behaviorEvidence
        )
        let behaviorOut: SessionExportBehaviorLogprobs? = behaviorSummary.map { summary in
            let isExposed = summary.status == "exposed"
            let format = isExposed ? src.behaviorEvidence?.format.flatMap { $0.isEmpty ? nil : $0 } : nil
            let sha = isExposed ? src.behaviorEvidence?.tokenizerSha256.flatMap { $0.isEmpty ? nil : $0 } : nil
            return SessionExportBehaviorLogprobs(
                status: summary.status,
                tokenCount: isExposed ? summary.tokenCount : nil,
                supervisedCount: isExposed ? summary.supervisedCount : nil,
                label: summary.label,
                format: format,
                tokenizerSha256: sha
            )
        } ?? src.importedBehaviorLogprobs

        let hasAnyMetric = ttftMs != nil
            || ttfaMs != nil
            || firstVisualMs != nil
            || totalMs != nil
            || visualCount > 0
            || i2vCount > 0
            || vramOut != nil
            || behaviorOut != nil

        guard hasAnyMetric else {
            return (nil, [])
        }

        let telemetryJSON = SessionExportTelemetry(
            ttftMs: ttftMs,
            ttfaMs: ttfaMs,
            firstVisualMs: firstVisualMs,
            totalMs: totalMs,
            visualSlots: visualCount,
            i2vCount: i2vCount,
            nativeAllocator: vramOut,
            behaviorLogprobs: behaviorOut
        )

        var effective = src
        effective.ttftMs = ttftMs
        effective.ttfaMs = ttfaMs
        effective.firstVisualMs = firstVisualMs
        effective.totalMs = totalMs
        effective.visualSettledCount = visualCount
        effective.hasI2VChain = hasI2VFlag
        effective.hasExplicitAnchor = hasExplicitFlag

        return (telemetryJSON, effective.formattedPills)
    }

    private static func buildReportWithPills(
        sessionID: String,
        model: String,
        turns: [ChatTurn],
        exportedAt: Date
    ) -> (SessionExportReport, [[TurnTelemetry.Pill]]) {
        let trimmedSession = sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        let pairs = pairTurns(turns)

        var exportedTurns: [SessionExportTurn] = []
        var perTurnPills: [[TurnTelemetry.Pill]] = []
        exportedTurns.reserveCapacity(pairs.count)
        perTurnPills.reserveCapacity(pairs.count)

        for (offset, pair) in pairs.enumerated() {
            let prompt = scrubInlineDataURIs(pair.user?.text ?? "")
            let response = extractResponseText(from: pair.assistant)
            let asrTrimmed = scrubInlineDataURIs(
                (pair.assistant?.transcript ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            )
            let visualSlots = extractVisualSlots(from: pair.assistant)
            let (telemetryJSON, pills) = buildTelemetry(
                from: pair.assistant?.telemetry,
                visualSlots: visualSlots
            )

            let rawInterruptedBytes = pair.assistant?.interruptedSpokenBytes
            let isInterrupted = (pair.assistant?.isInterrupted == true) || rawInterruptedBytes != nil
            let spokenBytes = rawInterruptedBytes.flatMap { $0 >= 0 ? $0 : nil }

            exportedTurns.append(SessionExportTurn(
                index: offset + 1,
                prompt: prompt,
                response: response,
                asrText: asrTrimmed.isEmpty ? nil : asrTrimmed,
                interrupted: isInterrupted ? true : nil,
                spokenBytes: isInterrupted ? spokenBytes : nil,
                visualSlots: visualSlots,
                telemetry: telemetryJSON
            ))
            perTurnPills.append(pills)
        }

        let report = SessionExportReport(
            schema: schema,
            exportedAt: formatISO8601UTC(exportedAt),
            sessionID: trimmedSession.isEmpty ? nil : trimmedSession,
            model: trimmedModel.isEmpty ? nil : trimmedModel,
            turnCount: exportedTurns.count,
            turns: exportedTurns
        )
        return (report, perTurnPills)
    }

    // MARK: - Pure client-side JSON import (macOS + Web schemas)

    static func importJSONString(_ text: String) throws -> ImportedSession {
        try importJSON(Data(text.utf8))
    }

    /// Strictly parses and validates a macOS (`umminfer-session-export-v1`) or
    /// Web (`umminfer-web-session-export-v1`) session JSON export without
    /// contacting the network or fabricating missing telemetry/media bytes.
    static func importJSON(_ data: Data) throws -> ImportedSession {
        guard !data.isEmpty else {
            throw SessionImportError.emptyPayload
        }
        guard data.count <= maxImportBytes else {
            throw SessionImportError.payloadTooLarge(bytes: data.count, limit: maxImportBytes)
        }

        let rawJSON: Any
        do {
            rawJSON = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw SessionImportError.invalidJSON(error.localizedDescription)
        }

        guard let root = rawJSON as? [String: Any] else {
            throw SessionImportError.invalidRootObject
        }

        guard let schemaValue = root["schema"] as? String,
              supportedImportSchemas.contains(schemaValue) else {
            throw SessionImportError.unsupportedSchema(root["schema"] as? String)
        }

        let exportedAt = try optionalTrimmedString(root["exported_at"], field: "exported_at")
        let sessionID = try optionalTrimmedString(root["session_id"], field: "session_id")
        let model = try optionalTrimmedString(root["model"], field: "model")
        let declaredTurnCount = try optionalNonNegativeInt(root["turn_count"], field: "turn_count")

        let hasTurnsKey = root["turns"] != nil
        let hasMessagesKey = root["messages"] != nil
        if hasTurnsKey && hasMessagesKey {
            throw SessionImportError.invalidField("cannot contain both 'turns' and 'messages'")
        }

        var restoredTurns: [ChatTurn] = []
        let logicalTurnCount: Int

        if hasTurnsKey {
            guard let rawTurns = root["turns"] as? [Any] else {
                throw SessionImportError.invalidField("turns must be an array")
            }
            if let declaredTurnCount, declaredTurnCount != rawTurns.count {
                throw SessionImportError.invalidField(
                    "turn_count (\(declaredTurnCount)) does not match turns length (\(rawTurns.count))"
                )
            }
            for (offset, item) in rawTurns.enumerated() {
                guard let entry = item as? [String: Any] else {
                    throw SessionImportError.invalidTurn(index: offset, reason: "turn must be an object")
                }
                let entryTurns = try parseTurnEntry(entry, index: offset)
                restoredTurns.append(contentsOf: entryTurns)
            }
            logicalTurnCount = rawTurns.count
        } else if hasMessagesKey {
            guard let rawMessages = root["messages"] as? [Any] else {
                throw SessionImportError.invalidField("messages must be an array")
            }
            for (offset, item) in rawMessages.enumerated() {
                guard let entry = item as? [String: Any] else {
                    throw SessionImportError.invalidTurn(index: offset, reason: "message must be an object")
                }
                let turn = try parseMessageEntry(entry, index: offset)
                restoredTurns.append(turn)
            }
            let pairedCount = pairTurns(restoredTurns).count
            if let declaredTurnCount, declaredTurnCount != pairedCount && declaredTurnCount != rawMessages.count {
                throw SessionImportError.invalidField(
                    "turn_count (\(declaredTurnCount)) does not match messages"
                )
            }
            logicalTurnCount = pairedCount
        } else {
            throw SessionImportError.missingConversationArray
        }

        let restoredMessages = restoredTurns.map { turn in
            ChatMessage(role: turn.role.rawValue, content: turn.text)
        }

        return ImportedSession(
            schema: schemaValue,
            exportedAt: exportedAt,
            sessionID: sessionID,
            model: model,
            turnCount: logicalTurnCount,
            turns: restoredTurns,
            messages: restoredMessages
        )
    }

    // MARK: - Strict JSON decoding helpers

    private static func strictBool(_ value: Any?, field: String) throws -> Bool? {
        guard let value, !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw SessionImportError.invalidField("\(field) must be a boolean")
        }
        return number.boolValue
    }

    private static func optionalNonNegativeInt(_ value: Any?, field: String) throws -> Int? {
        guard let value, !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number) else {
            throw SessionImportError.invalidField("\(field) must be a non-negative integer")
        }
        let int64Val = number.int64Value
        guard int64Val >= 0,
              int64Val <= Int64(Int.max),
              number.doubleValue >= 0 else {
            throw SessionImportError.invalidField("\(field) must be a non-negative integer")
        }
        return Int(int64Val)
    }

    private static func optionalNonNegativeInt64(_ value: Any?, field: String) throws -> Int64? {
        guard let value, !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number) else {
            throw SessionImportError.invalidField("\(field) must be a non-negative integer")
        }
        let int64Val = number.int64Value
        guard int64Val >= 0, number.doubleValue >= 0 else {
            throw SessionImportError.invalidField("\(field) must be a non-negative integer")
        }
        return int64Val
    }

    private static func optionalNonNegativeDouble(_ value: Any?, field: String) throws -> Double? {
        guard let value, !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else {
            throw SessionImportError.invalidField("\(field) must be a finite non-negative number")
        }
        let dbl = number.doubleValue
        guard dbl.isFinite, dbl >= 0 else {
            throw SessionImportError.invalidField("\(field) must be a finite non-negative number")
        }
        return dbl
    }

    private static func optionalTrimmedString(_ value: Any?, field: String) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        guard let str = value as? String else {
            throw SessionImportError.invalidField("\(field) must be a string")
        }
        let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func optionalStringStrict(_ value: Any?, field: String) throws -> String? {
        guard let value else { return nil }
        guard let str = value as? String else {
            throw SessionImportError.invalidField("\(field) must be a string")
        }
        return str
    }

    private static func optionalNullableString(_ value: Any?, field: String) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        guard let str = value as? String else {
            throw SessionImportError.invalidField("\(field) must be a string")
        }
        return str
    }

    private static func optionalValidatedSHA256(_ value: Any?, field: String) throws -> String? {
        guard let raw = try optionalNullableString(value, field: field) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return nil }
        guard let normalized = UnifiedDecode.normalizedAnchorSHA256(trimmed) else {
            throw SessionImportError.invalidField("\(field) must be a 64-character hex SHA-256 digest")
        }
        return normalized
    }

    private static func inferSlotIndex(from slotID: String, fallback: Int) -> Int {
        if slotID.hasPrefix("slot-") {
            let suffix = slotID.dropFirst("slot-".count)
            if !suffix.isEmpty,
               suffix.unicodeScalars.allSatisfy({ $0.value >= 0x30 && $0.value <= 0x39 }),
               let parsed = Int(suffix), parsed >= 0 {
                return parsed
            }
        }
        return fallback
    }

    private static func parseVisualSlots(_ raw: Any?, turnIndex: Int) throws -> [VisualSlotState] {
        guard let raw, !(raw is NSNull) else { return [] }
        guard let array = raw as? [Any] else {
            throw SessionImportError.invalidTurn(index: turnIndex, reason: "visual_slots must be an array")
        }
        var slots: [VisualSlotState] = []
        slots.reserveCapacity(array.count)

        for (slotOffset, item) in array.enumerated() {
            guard let obj = item as? [String: Any] else {
                throw SessionImportError.invalidTurn(
                    index: turnIndex,
                    reason: "visual_slots[\(slotOffset)] must be an object"
                )
            }
            let prefix = "turns[\(turnIndex)].visual_slots[\(slotOffset)]"
            let macSlotID = try optionalTrimmedString(obj["slot_id"], field: "\(prefix).slot_id")
            let webSlotID = try optionalTrimmedString(obj["id"], field: "\(prefix).id")
            let rawSlotIndex = try optionalNonNegativeInt(obj["slot_index"], field: "\(prefix).slot_index")

            guard let kindValue = obj["kind"] as? String,
                  kindValue == "image" || kindValue == "video" else {
                throw SessionImportError.invalidField("\(prefix).kind must be 'image' or 'video'")
            }

            let resolvedID = macSlotID ?? webSlotID ?? "slot-\(rawSlotIndex ?? slotOffset)"
            let resolvedIndex = rawSlotIndex ?? inferSlotIndex(from: resolvedID, fallback: slotOffset)
            let promptText = scrubInlineDataURIs(
                (try optionalNullableString(obj["prompt"], field: "\(prefix).prompt") ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            )
            let bytes = try optionalNonNegativeInt(obj["bytes"], field: "\(prefix).bytes")
            let sha256 = try optionalValidatedSHA256(obj["sha256"], field: "\(prefix).sha256")
            let continuityMode = try optionalTrimmedString(obj["continuity_mode"], field: "\(prefix).continuity_mode")
            let anchorSlotIndex = try optionalNonNegativeInt(obj["anchor_slot_index"], field: "\(prefix).anchor_slot_index")
            let anchorExplicit = try strictBool(obj["anchor_explicit"], field: "\(prefix).anchor_explicit") ?? false
            let anchorSha256 = try optionalValidatedSHA256(obj["anchor_sha256"], field: "\(prefix).anchor_sha256")
            let errorText = try optionalTrimmedString(obj["error"], field: "\(prefix).error")

            // Never fabricate binary imageArtifact/videoArtifact data bytes.
            slots.append(VisualSlotState(
                slotID: resolvedID,
                slotIndex: resolvedIndex,
                prompt: promptText,
                activeKind: kindValue,
                isLoading: false,
                error: errorText,
                imageArtifact: nil,
                videoArtifact: nil,
                swapToken: nil,
                continuityMode: continuityMode,
                anchorSlotIndex: anchorSlotIndex,
                anchorExplicit: anchorExplicit,
                anchorSha256: anchorSha256,
                importedBytes: bytes,
                importedSha256: sha256
            ))
        }
        return slots
    }

    private static func parseTelemetry(
        _ raw: Any?,
        visualSlots: [VisualSlotState],
        turnIndex: Int
    ) throws -> TurnTelemetry? {
        guard let raw, !(raw is NSNull) else { return nil }
        guard let obj = raw as? [String: Any] else {
            throw SessionImportError.invalidTurn(index: turnIndex, reason: "telemetry must be an object")
        }
        let prefix = "turns[\(turnIndex)].telemetry"

        let ttftMs = try optionalNonNegativeDouble(obj["ttft_ms"], field: "\(prefix).ttft_ms")
        let ttfaMs = try optionalNonNegativeDouble(obj["ttfa_ms"], field: "\(prefix).ttfa_ms")
        let firstVisualMs = try optionalNonNegativeDouble(obj["first_visual_ms"], field: "\(prefix).first_visual_ms")
        let totalMs = try optionalNonNegativeDouble(obj["total_ms"], field: "\(prefix).total_ms")
        let visualCount = try optionalNonNegativeInt(obj["visual_slots"], field: "\(prefix).visual_slots")
        let i2vCount = try optionalNonNegativeInt(obj["i2v_count"], field: "\(prefix).i2v_count")

        let importedVRAM = try parseNativeAllocator(obj["native_allocator"], field: "\(prefix).native_allocator")
        let importedBehavior = try parseBehaviorLogprobs(obj["behavior_logprobs"], field: "\(prefix).behavior_logprobs")

        let slotHasExplicit = visualSlots.contains { $0.anchorExplicit }
        let slotHasI2V = visualSlots.contains {
            $0.anchorSlotIndex != nil
                || $0.continuityMode == NarrativeContinuityMode.i2vChain.rawValue
                || $0.anchorExplicit
        }
        let resolvedVisualCount = max(visualCount ?? 0, visualSlots.count)
        let resolvedHasI2V = (i2vCount ?? 0) > 0 || slotHasI2V || slotHasExplicit

        return TurnTelemetry(
            ttftMs: ttftMs,
            ttfaMs: ttfaMs,
            visualSettledCount: resolvedVisualCount,
            firstVisualMs: firstVisualMs,
            hasI2VChain: resolvedHasI2V,
            hasExplicitAnchor: slotHasExplicit,
            totalMs: totalMs,
            nativeAllocator: nil,
            behaviorRequested: false,
            behaviorEvidenceHeader: nil,
            behaviorEvidence: nil,
            importedNativeAllocator: importedVRAM,
            importedBehaviorLogprobs: importedBehavior
        )
    }

    private static func parseNativeAllocator(
        _ raw: Any?,
        field: String
    ) throws -> SessionExportNativeAllocator? {
        guard let raw, !(raw is NSNull) else { return nil }
        guard let obj = raw as? [String: Any] else {
            throw SessionImportError.invalidField("\(field) must be an object")
        }
        guard let coverage = obj["coverage"] as? String,
              coverage == "COVERED" || coverage == "PARTIAL" || coverage == "NOT_COVERED" else {
            throw SessionImportError.invalidField("\(field).coverage must be COVERED, PARTIAL, or NOT_COVERED")
        }
        guard let covered = try strictBool(obj["covered"], field: "\(field).covered") else {
            throw SessionImportError.invalidField("\(field).covered is required")
        }
        if covered && coverage == "NOT_COVERED" {
            throw SessionImportError.invalidField("\(field).covered cannot be true when coverage is NOT_COVERED")
        }
        if !covered && coverage != "NOT_COVERED" {
            throw SessionImportError.invalidField("\(field).covered cannot be false when coverage is \(coverage)")
        }

        let peakBytes = try optionalNonNegativeInt64(obj["peak_bytes"], field: "\(field).peak_bytes")
        if covered && peakBytes == nil {
            throw SessionImportError.invalidField("\(field).peak_bytes is required when covered is true")
        }
        if !covered && peakBytes != nil {
            throw SessionImportError.invalidField("\(field).peak_bytes must be null when not covered")
        }

        let coveredStageCount = try optionalNonNegativeInt(
            obj["covered_stage_count"], field: "\(field).covered_stage_count"
        ) ?? (covered ? 1 : 0)
        let totalStageCount = try optionalNonNegativeInt(
            obj["total_stage_count"], field: "\(field).total_stage_count"
        ) ?? coveredStageCount
        if coveredStageCount > totalStageCount {
            throw SessionImportError.invalidField("\(field).covered_stage_count exceeds total_stage_count")
        }

        var details: [String] = []
        if let rawDetails = obj["details"], !(rawDetails is NSNull) {
            guard let arr = rawDetails as? [Any] else {
                throw SessionImportError.invalidField("\(field).details must be an array of strings")
            }
            for item in arr {
                guard let line = item as? String else {
                    throw SessionImportError.invalidField("\(field).details must contain only strings")
                }
                details.append(line)
            }
        }

        let providedLabel = try optionalTrimmedString(obj["label"], field: "\(field).label")
        let canonicalLabel: String
        if covered, let peakBytes, let formatted = NativeAllocatorTelemetrySummary.formatBytes(peakBytes) {
            canonicalLabel = "VRAM: \(formatted) peak"
        } else {
            canonicalLabel = "VRAM: N/C"
        }

        return SessionExportNativeAllocator(
            coverage: coverage,
            covered: covered,
            coveredStageCount: coveredStageCount,
            totalStageCount: totalStageCount,
            peakBytes: covered ? peakBytes : nil,
            label: providedLabel ?? canonicalLabel,
            details: details
        )
    }

    private static func parseBehaviorLogprobs(
        _ raw: Any?,
        field: String
    ) throws -> SessionExportBehaviorLogprobs? {
        guard let raw, !(raw is NSNull) else { return nil }
        guard let obj = raw as? [String: Any] else {
            throw SessionImportError.invalidField("\(field) must be an object")
        }
        guard let status = obj["status"] as? String,
              status == "exposed" || status == "not_exposed" || status == "absent" else {
            throw SessionImportError.invalidField("\(field).status must be exposed, not_exposed, or absent")
        }
        let tokenCount = try optionalNonNegativeInt(obj["token_count"], field: "\(field).token_count")
        let supervisedCount = try optionalNonNegativeInt(obj["supervised_count"], field: "\(field).supervised_count")

        if status == "exposed" {
            guard let tc = tokenCount, let sc = supervisedCount else {
                throw SessionImportError.invalidField(
                    "\(field).token_count and supervised_count are required when status is exposed"
                )
            }
            guard sc <= tc else {
                throw SessionImportError.invalidField("\(field).supervised_count exceeds token_count")
            }
        } else {
            if tokenCount != nil || supervisedCount != nil {
                throw SessionImportError.invalidField(
                    "\(field).token_count and supervised_count must be null when status is \(status)"
                )
            }
        }

        let providedLabel = try optionalTrimmedString(obj["label"], field: "\(field).label")
        let format = try optionalTrimmedString(obj["format"], field: "\(field).format")
        let tokenizerSha256 = try optionalTrimmedString(obj["tokenizer_sha256"], field: "\(field).tokenizer_sha256")
        let canonicalLabel = status == "exposed"
            ? "Behavior: \(tokenCount!) tok (\(supervisedCount!) sup)"
            : "Behavior: \(status)"

        return SessionExportBehaviorLogprobs(
            status: status,
            tokenCount: status == "exposed" ? tokenCount : nil,
            supervisedCount: status == "exposed" ? supervisedCount : nil,
            label: providedLabel ?? canonicalLabel,
            format: status == "exposed" ? format : nil,
            tokenizerSha256: status == "exposed" ? tokenizerSha256 : nil
        )
    }

    private static func buildAssistantTurn(
        text: String,
        asrText: String?,
        interrupted: Bool,
        spokenBytes: Int?,
        visualSlots: [VisualSlotState],
        telemetry: TurnTelemetry?
    ) -> ChatTurn {
        var turn = ChatTurn(
            role: .assistant,
            text: text,
            state: interrupted ? .stopped : .complete
        )
        if let asrText, !asrText.isEmpty {
            turn.transcript = asrText
        }
        turn.isInterrupted = interrupted
        turn.interruptedSpokenBytes = interrupted ? spokenBytes : nil
        if !visualSlots.isEmpty {
            if !text.isEmpty {
                turn.blocks.append(.text(id: UUID(), content: text))
            }
            for slot in visualSlots {
                turn.blocks.append(.visual(slot))
            }
        }
        if let telemetry {
            turn.telemetry = telemetry
        }
        return turn
    }

    private static func parseMessageEntry(_ obj: [String: Any], index: Int) throws -> ChatTurn {
        guard let roleValue = obj["role"] as? String else {
            throw SessionImportError.invalidTurn(index: index, reason: "role must be a string")
        }
        guard roleValue == "user" || roleValue == "assistant" else {
            throw SessionImportError.invalidRole(roleValue)
        }
        let contentValue = try optionalStringStrict(obj["content"], field: "messages[\(index)].content")
        let textValue = try optionalStringStrict(obj["text"], field: "messages[\(index)].text")
        guard let rawText = contentValue ?? textValue else {
            throw SessionImportError.invalidTurn(index: index, reason: "missing string content/text")
        }
        let scrubbed = scrubInlineDataURIs(rawText)

        if roleValue == "user" {
            return ChatTurn(role: .user, text: scrubbed, state: .complete)
        }

        let asrText = try optionalNullableString(obj["asr_text"], field: "messages[\(index)].asr_text")
            .map { scrubInlineDataURIs($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        let interrupted = try strictBool(obj["interrupted"], field: "messages[\(index)].interrupted") ?? false
        let spokenBytes = try optionalNonNegativeInt(obj["spoken_bytes"], field: "messages[\(index)].spoken_bytes")
        let visualSlots = try parseVisualSlots(obj["visual_slots"], turnIndex: index)
        let telemetry = try parseTelemetry(obj["telemetry"], visualSlots: visualSlots, turnIndex: index)

        return buildAssistantTurn(
            text: scrubbed,
            asrText: asrText,
            interrupted: interrupted,
            spokenBytes: spokenBytes,
            visualSlots: visualSlots,
            telemetry: telemetry
        )
    }

    private static func parseTurnEntry(_ obj: [String: Any], index: Int) throws -> [ChatTurn] {
        if let rawIdx = try optionalNonNegativeInt(obj["index"], field: "turns[\(index)].index"),
           rawIdx < 1 {
            throw SessionImportError.invalidTurn(index: index, reason: "index must be >= 1")
        }

        let role: String?
        if let rawRole = obj["role"], !(rawRole is NSNull) {
            guard let roleStr = rawRole as? String else {
                throw SessionImportError.invalidTurn(index: index, reason: "role must be a string")
            }
            guard roleStr == "turn" || roleStr == "user" || roleStr == "assistant" else {
                throw SessionImportError.invalidRole(roleStr)
            }
            role = roleStr
        } else {
            role = nil
        }

        let promptRaw = try optionalStringStrict(obj["prompt"], field: "turns[\(index)].prompt")
        let responseRaw = try optionalStringStrict(obj["response"], field: "turns[\(index)].response")
        let contentRaw = try optionalStringStrict(obj["content"], field: "turns[\(index)].content")
        let textRaw = try optionalStringStrict(obj["text"], field: "turns[\(index)].text")

        guard promptRaw != nil || responseRaw != nil || contentRaw != nil || textRaw != nil else {
            throw SessionImportError.invalidTurn(index: index, reason: "missing prompt/response text")
        }

        if role == "user", let responseRaw, !responseRaw.isEmpty {
            throw SessionImportError.invalidTurn(
                index: index,
                reason: "user role turn cannot carry a non-empty response"
            )
        }
        if role == "assistant", let promptRaw, !promptRaw.isEmpty {
            throw SessionImportError.invalidTurn(
                index: index,
                reason: "assistant role turn cannot carry a non-empty prompt"
            )
        }

        let userText = scrubInlineDataURIs(
            promptRaw ?? (role == "user" ? (contentRaw ?? textRaw ?? "") : "")
        )
        let assistantText = scrubInlineDataURIs(
            responseRaw ?? (role == "assistant" ? (contentRaw ?? textRaw ?? "") : "")
        )
        let asrText = try optionalNullableString(obj["asr_text"], field: "turns[\(index)].asr_text")
            .map { scrubInlineDataURIs($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .flatMap { $0.isEmpty ? nil : $0 }
        let interrupted = try strictBool(obj["interrupted"], field: "turns[\(index)].interrupted") ?? false
        let spokenBytes = try optionalNonNegativeInt(obj["spoken_bytes"], field: "turns[\(index)].spoken_bytes")
        let visualSlots = try parseVisualSlots(obj["visual_slots"], turnIndex: index)
        let telemetry = try parseTelemetry(obj["telemetry"], visualSlots: visualSlots, turnIndex: index)

        var result: [ChatTurn] = []
        let hasAssistantContent = !assistantText.isEmpty
            || asrText != nil
            || interrupted
            || !visualSlots.isEmpty
            || telemetry != nil
            || role == "assistant"

        if !userText.isEmpty || (role == "user" && !hasAssistantContent) {
            result.append(ChatTurn(role: .user, text: userText, state: .complete))
        }
        if hasAssistantContent {
            result.append(buildAssistantTurn(
                text: assistantText,
                asrText: asrText,
                interrupted: interrupted,
                spokenBytes: spokenBytes,
                visualSlots: visualSlots,
                telemetry: telemetry
            ))
        }

        guard !result.isEmpty else {
            throw SessionImportError.invalidTurn(index: index, reason: "turn has no user or assistant content")
        }
        return result
    }
}
