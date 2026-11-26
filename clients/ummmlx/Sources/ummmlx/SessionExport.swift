// SPDX-License-Identifier: Apache-2.0
// Pure client-side session export (JSON + Markdown) for macOS ummmlx.
// Zero network calls, zero binary/base64 media bloat, zero fabricated metrics.

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

enum SessionExporter {
    static let schema = "umminfer-session-export-v1"

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
            let errorText = slot.error?.trimmingCharacters(in: .whitespacesAndNewlines)

            slots.append(SessionExportVisualSlot(
                slotID: slot.slotID.isEmpty ? "slot-\(slot.slotIndex)" : slot.slotID,
                slotIndex: slot.slotIndex,
                kind: kind,
                prompt: prompt.isEmpty ? nil : prompt,
                bytes: artifact.map { $0.data.count },
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
        }

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
        }

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
            let isInterrupted = rawInterruptedBytes != nil
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
}
