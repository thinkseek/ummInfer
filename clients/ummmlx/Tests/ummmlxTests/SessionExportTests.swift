// SPDX-License-Identifier: Apache-2.0

import CryptoKit
import Foundation
import XCTest
@testable import ummmlx

private func testSHA256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

final class SessionExportTests: XCTestCase {
    private let fixedDate = Date(timeIntervalSince1970: 1_795_684_274.25) // 2026-11-26T09:11:14.250Z

    func testFormatExtensions() {
        XCTAssertEqual(SessionExportFormat.json.fileExtension, "json")
        XCTAssertEqual(SessionExportFormat.markdown.fileExtension, "md")
    }

    func testFilenameSanitizationAndFallback() {
        XCTAssertEqual(SessionExporter.sanitizeSessionID(""), "untitled")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("   \t\n "), "untitled")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("../.."), "untitled")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("---___///"), "untitled")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("../../etc/passwd"), "etc-passwd")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("..\\..\\secret\\file"), "secret-file")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("  my session 01  "), "my-session-01")
        XCTAssertEqual(SessionExporter.sanitizeSessionID("ummmlx-abc123_def"), "ummmlx-abc123_def")

        let longID = String(repeating: "a", count: 63) + "-trailing-overflow"
        let sanitizedLong = SessionExporter.sanitizeSessionID(longID)
        XCTAssertLessThanOrEqual(sanitizedLong.count, 64)
        XCTAssertFalse(sanitizedLong.hasSuffix("-"))
        XCTAssertEqual(sanitizedLong, String(repeating: "a", count: 63))

        XCTAssertEqual(
            SessionExporter.defaultFilename(sessionID: "../../evil/path", format: .json),
            "umminfer-session-evil-path.json"
        )
        XCTAssertEqual(
            SessionExporter.defaultFilename(sessionID: "  ", format: .markdown),
            "umminfer-session-untitled.md"
        )
        XCTAssertEqual(
            SessionExporter.defaultFilename(sessionID: "srv_session-42", format: .markdown),
            "umminfer-session-srv_session-42.md"
        )
    }

    func testEmptySessionExportJSONAndMarkdown() throws {
        let jsonData = SessionExporter.buildJSONData(
            sessionID: "",
            model: "",
            turns: [],
            exportedAt: fixedDate
        )
        let root = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        )
        XCTAssertEqual(root["schema"] as? String, "umminfer-session-export-v1")
        XCTAssertEqual(root["exported_at"] as? String, "2026-11-26T09:11:14.250Z")
        XCTAssertTrue(root["session_id"] is NSNull)
        XCTAssertTrue(root["model"] is NSNull)
        XCTAssertEqual(root["turn_count"] as? Int, 0)
        let turns = try XCTUnwrap(root["turns"] as? [Any])
        XCTAssertTrue(turns.isEmpty)

        // Pretty-printed with deterministic sorted keys and trailing newline.
        let jsonText = String(decoding: jsonData, as: UTF8.self)
        XCTAssertTrue(jsonText.hasSuffix("\n"))
        XCTAssertEqual(
            jsonData,
            SessionExporter.buildJSONData(sessionID: "", model: "", turns: [], exportedAt: fixedDate)
        )

        let md = SessionExporter.buildMarkdown(
            sessionID: "",
            model: "",
            turns: [],
            exportedAt: fixedDate
        )
        XCTAssertTrue(md.contains("# ummInfer Session Export"))
        XCTAssertTrue(md.contains("- **Session ID:** _none_"))
        XCTAssertFalse(md.contains("**Model:**"))
        XCTAssertTrue(md.contains("- **Exported at:** `2026-11-26T09:11:14.250Z`"))
        XCTAssertTrue(md.contains("- **Turns:** 0"))
        XCTAssertTrue(md.contains("_No turns recorded._"))
    }

    func testMultiTurnExportVisualI2VBargeInVRAMAndBehaviorZeroFabrication() throws {
        let imageBytes = Data(repeating: 0x41, count: 2048)
        let videoBytes = Data(repeating: 0x42, count: 4096)
        let imageB64Sentinel = imageBytes.base64EncodedString()
        let videoB64Sentinel = videoBytes.base64EncodedString()
        let imageSHA = testSHA256Hex(imageBytes)
        let videoSHA = testSHA256Hex(videoBytes)

        let imageArtifact = GeneratedArtifact(
            id: "img-0",
            mediaType: "image",
            mimeType: "image/png",
            data: imageBytes,
            sha256: imageSHA,
            slotID: "slot-0",
            slotIndex: 0
        )
        let videoArtifact = GeneratedArtifact(
            id: "vid-1",
            mediaType: "video",
            mimeType: "video/mp4",
            data: videoBytes,
            sha256: videoSHA,
            slotID: "slot-1",
            slotIndex: 1,
            anchorSlotIndex: 0,
            anchorExplicit: true,
            anchorSha256: imageSHA
        )

        let slot0 = VisualSlotState(
            slotID: "slot-0",
            slotIndex: 0,
            prompt: "glowing paper lantern over misty river",
            activeKind: "image",
            isLoading: false,
            imageArtifact: imageArtifact
        )
        let slot1 = VisualSlotState(
            slotID: "slot-1",
            slotIndex: 1,
            prompt: "lantern drifting in river breeze",
            activeKind: "video",
            isLoading: false,
            videoArtifact: videoArtifact,
            continuityMode: NarrativeContinuityMode.i2vChain.rawValue,
            anchorSlotIndex: 0,
            anchorExplicit: true,
            anchorSha256: imageSHA
        )

        let exposedEvidence = BehaviorLogprobsEvidence(
            format: "umminfer-behavior-logprobs-v1",
            tokenizerSha256: "1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016",
            tokenIDs: [101, 202, 303, 404],
            logprobs: [-0.1, -0.2, -0.3, -0.05],
            mask: [1, 0, 1, 0]
        )

        // Turn 1: Multimodal narrative I2V chain + COVERED VRAM + exposed behavior + inline data URI scrub.
        let user1 = ChatTurn(
            role: .user,
            text: "Create a storyboard using data:image/png;base64,\(imageB64Sentinel) as reference.",
            state: .complete
        )
        var assistant1 = ChatTurn(
            role: .assistant,
            text: "Opening lantern scene and river motion.",
            state: .complete
        )
        assistant1.artifacts = [imageArtifact, videoArtifact]
        assistant1.blocks = [
            .text(id: UUID(), content: "Opening lantern scene"),
            .visual(slot0),
            .visual(slot1),
        ]
        assistant1.telemetry = TurnTelemetry(
            ttftMs: 265,
            ttfaMs: 380,
            visualSettledCount: 2,
            firstVisualMs: 4120,
            hasI2VChain: true,
            hasExplicitAnchor: true,
            totalMs: 8790,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm",
                        nativeStage: "qwen38",
                        coverage: "COVERED",
                        allocatedBytesPeak: 134_637_158, // ~128.4 MiB
                        epochID: 17
                    )
                ]
            ),
            behaviorRequested: true,
            behaviorEvidenceHeader: "EXPOSED",
            behaviorEvidence: exposedEvidence
        )

        // Turn 2: Voice barge-in with 42 B spoken + NOT_COVERED VRAM (null peak) + behavior not_exposed.
        let user2 = ChatTurn(role: .user, text: "Explain gravitational lensing.", state: .complete)
        var assistant2 = ChatTurn(
            role: .assistant,
            text: "Light bends around massive",
            state: .stopped
        )
        assistant2.interruptedSpokenBytes = 42
        assistant2.telemetry = TurnTelemetry(
            ttftMs: 120,
            ttfaMs: nil,
            visualSettledCount: 0,
            firstVisualMs: nil,
            hasI2VChain: false,
            totalMs: 950,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "NOT_COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm",
                        nativeStage: "qwen38",
                        coverage: "NOT_COVERED",
                        allocatedBytesPeak: nil,
                        reason: "batched_request_attribution_unavailable"
                    )
                ]
            ),
            behaviorRequested: true,
            behaviorEvidenceHeader: "not_exposed",
            behaviorEvidence: exposedEvidence // stray payload must NEVER leak counts
        )

        // Turn 3: Interrupted before audio (0 B spoken) + behavior absent + invalid negative/NaN timings omitted.
        let user3 = ChatTurn(role: .user, text: "Wait, stop.", state: .complete)
        var assistant3 = ChatTurn(role: .assistant, text: "", state: .stopped)
        assistant3.interruptedSpokenBytes = 0
        assistant3.telemetry = TurnTelemetry(
            ttftMs: -10,
            ttfaMs: .nan,
            visualSettledCount: 0,
            firstVisualMs: .infinity,
            hasI2VChain: false,
            totalMs: -.infinity,
            behaviorRequested: true,
            behaviorEvidenceHeader: nil,
            behaviorEvidence: nil
        )

        // Turn 4: Plain text turn, uninterrupted, default-OFF opt-in telemetry.
        let user4 = ChatTurn(role: .user, text: "2 + 2?", state: .complete)
        var assistant4 = ChatTurn(role: .assistant, text: "4.", state: .complete)
        assistant4.telemetry = TurnTelemetry(ttftMs: 90, totalMs: 310)

        let turns = [user1, assistant1, user2, assistant2, user3, assistant3, user4, assistant4]

        // --- JSON verification ---
        let jsonData = SessionExporter.buildJSONData(
            sessionID: "ummmlx-story-99",
            model: "qwen3-native",
            turns: turns,
            exportedAt: fixedDate
        )
        let jsonString = String(decoding: jsonData, as: UTF8.self)

        // Zero binary/base64 bloat anywhere in JSON.
        XCTAssertFalse(jsonString.contains(imageB64Sentinel))
        XCTAssertFalse(jsonString.contains(videoB64Sentinel))
        XCTAssertFalse(jsonString.contains("data:image/png;base64,"))
        XCTAssertTrue(jsonString.contains("[omitted-base64]"))

        let root = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        )
        XCTAssertEqual(root["schema"] as? String, "umminfer-session-export-v1")
        XCTAssertEqual(root["exported_at"] as? String, "2026-11-26T09:11:14.250Z")
        XCTAssertEqual(root["session_id"] as? String, "ummmlx-story-99")
        XCTAssertEqual(root["model"] as? String, "qwen3-native")
        XCTAssertEqual(root["turn_count"] as? Int, 4)

        let jsonTurns = try XCTUnwrap(root["turns"] as? [[String: Any]])
        XCTAssertEqual(jsonTurns.count, 4)

        // Verify Turn 1 JSON.
        let t1 = jsonTurns[0]
        XCTAssertEqual(t1["index"] as? Int, 1)
        XCTAssertEqual(t1["prompt"] as? String, "Create a storyboard using [omitted-base64] as reference.")
        XCTAssertEqual(t1["response"] as? String, "Opening lantern scene and river motion.")
        XCTAssertNil(t1["interrupted"])
        XCTAssertNil(t1["spoken_bytes"])

        let t1Slots = try XCTUnwrap(t1["visual_slots"] as? [[String: Any]])
        XCTAssertEqual(t1Slots.count, 2)
        XCTAssertEqual(t1Slots[0]["slot_id"] as? String, "slot-0")
        XCTAssertEqual(t1Slots[0]["slot_index"] as? Int, 0)
        XCTAssertEqual(t1Slots[0]["kind"] as? String, "image")
        XCTAssertEqual(t1Slots[0]["bytes"] as? Int, 2048)
        XCTAssertEqual(t1Slots[0]["sha256"] as? String, imageSHA)
        XCTAssertNil(t1Slots[0]["data"])
        XCTAssertNil(t1Slots[0]["base64"])

        XCTAssertEqual(t1Slots[1]["slot_id"] as? String, "slot-1")
        XCTAssertEqual(t1Slots[1]["slot_index"] as? Int, 1)
        XCTAssertEqual(t1Slots[1]["kind"] as? String, "video")
        XCTAssertEqual(t1Slots[1]["bytes"] as? Int, 4096)
        XCTAssertEqual(t1Slots[1]["sha256"] as? String, videoSHA)
        XCTAssertEqual(t1Slots[1]["continuity_mode"] as? String, "i2v_chain")
        XCTAssertEqual(t1Slots[1]["anchor_slot_index"] as? Int, 0)
        XCTAssertEqual(t1Slots[1]["anchor_explicit"] as? Bool, true)
        XCTAssertEqual(t1Slots[1]["anchor_sha256"] as? String, imageSHA)
        XCTAssertNil(t1Slots[1]["data"])

        let t1Tel = try XCTUnwrap(t1["telemetry"] as? [String: Any])
        XCTAssertEqual(t1Tel["ttft_ms"] as? Double, 265)
        XCTAssertEqual(t1Tel["ttfa_ms"] as? Double, 380)
        XCTAssertEqual(t1Tel["first_visual_ms"] as? Double, 4120)
        XCTAssertEqual(t1Tel["total_ms"] as? Double, 8790)
        XCTAssertEqual(t1Tel["visual_slots"] as? Int, 2)
        XCTAssertEqual(t1Tel["i2v_count"] as? Int, 1)

        let t1Vram = try XCTUnwrap(t1Tel["native_allocator"] as? [String: Any])
        XCTAssertEqual(t1Vram["coverage"] as? String, "COVERED")
        XCTAssertEqual(t1Vram["covered"] as? Bool, true)
        XCTAssertEqual(t1Vram["covered_stage_count"] as? Int, 1)
        XCTAssertEqual(t1Vram["total_stage_count"] as? Int, 1)
        XCTAssertEqual(t1Vram["peak_bytes"] as? Int, 134_637_158)
        XCTAssertEqual(t1Vram["label"] as? String, "VRAM: 128.4 MiB peak")

        let t1Behavior = try XCTUnwrap(t1Tel["behavior_logprobs"] as? [String: Any])
        XCTAssertEqual(t1Behavior["status"] as? String, "exposed")
        XCTAssertEqual(t1Behavior["token_count"] as? Int, 4)
        XCTAssertEqual(t1Behavior["supervised_count"] as? Int, 2)
        XCTAssertEqual(t1Behavior["label"] as? String, "Behavior: 4 tok (2 sup)")
        XCTAssertEqual(t1Behavior["format"] as? String, "umminfer-behavior-logprobs-v1")
        XCTAssertEqual(
            t1Behavior["tokenizer_sha256"] as? String,
            "1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016"
        )

        // Verify Turn 2 JSON (interrupted + NOT_COVERED + not_exposed, zero fabrication).
        let t2 = jsonTurns[1]
        XCTAssertEqual(t2["index"] as? Int, 2)
        XCTAssertEqual(t2["interrupted"] as? Bool, true)
        XCTAssertEqual(t2["spoken_bytes"] as? Int, 42)
        let t2Tel = try XCTUnwrap(t2["telemetry"] as? [String: Any])
        XCTAssertEqual(t2Tel["ttft_ms"] as? Double, 120)
        XCTAssertNil(t2Tel["ttfa_ms"])
        XCTAssertEqual(t2Tel["total_ms"] as? Double, 950)

        let t2Vram = try XCTUnwrap(t2Tel["native_allocator"] as? [String: Any])
        XCTAssertEqual(t2Vram["coverage"] as? String, "NOT_COVERED")
        XCTAssertEqual(t2Vram["covered"] as? Bool, false)
        XCTAssertTrue(t2Vram["peak_bytes"] is NSNull, "NOT_COVERED peak_bytes must be null, never 0")
        XCTAssertEqual(t2Vram["label"] as? String, "VRAM: N/C")

        let t2Behavior = try XCTUnwrap(t2Tel["behavior_logprobs"] as? [String: Any])
        XCTAssertEqual(t2Behavior["status"] as? String, "not_exposed")
        XCTAssertTrue(t2Behavior["token_count"] is NSNull, "not_exposed token_count must be null, never 0")
        XCTAssertTrue(t2Behavior["supervised_count"] is NSNull, "not_exposed supervised_count must be null, never 0")
        XCTAssertEqual(t2Behavior["label"] as? String, "Behavior: not_exposed")
        XCTAssertNil(t2Behavior["format"])
        XCTAssertNil(t2Behavior["tokenizer_sha256"])

        // Verify Turn 3 JSON (interrupted before audio + behavior absent + invalid timings omitted).
        let t3 = jsonTurns[2]
        XCTAssertEqual(t3["index"] as? Int, 3)
        XCTAssertEqual(t3["interrupted"] as? Bool, true)
        XCTAssertEqual(t3["spoken_bytes"] as? Int, 0)
        let t3Tel = try XCTUnwrap(t3["telemetry"] as? [String: Any])
        XCTAssertNil(t3Tel["ttft_ms"])
        XCTAssertNil(t3Tel["ttfa_ms"])
        XCTAssertNil(t3Tel["first_visual_ms"])
        XCTAssertNil(t3Tel["total_ms"])
        let t3Behavior = try XCTUnwrap(t3Tel["behavior_logprobs"] as? [String: Any])
        XCTAssertEqual(t3Behavior["status"] as? String, "absent")
        XCTAssertTrue(t3Behavior["token_count"] is NSNull)
        XCTAssertTrue(t3Behavior["supervised_count"] is NSNull)
        XCTAssertEqual(t3Behavior["label"] as? String, "Behavior: absent")

        // Verify Turn 4 JSON (plain text, uninterrupted, default-OFF opt-ins absent).
        let t4 = jsonTurns[3]
        XCTAssertEqual(t4["index"] as? Int, 4)
        XCTAssertNil(t4["interrupted"])
        XCTAssertNil(t4["spoken_bytes"])
        let t4Tel = try XCTUnwrap(t4["telemetry"] as? [String: Any])
        XCTAssertEqual(t4Tel["ttft_ms"] as? Double, 90)
        XCTAssertEqual(t4Tel["total_ms"] as? Double, 310)
        XCTAssertNil(t4Tel["ttfa_ms"])
        XCTAssertNil(t4Tel["native_allocator"])
        XCTAssertNil(t4Tel["behavior_logprobs"])

        // --- Markdown verification ---
        let md = SessionExporter.buildMarkdown(
            sessionID: "ummmlx-story-99",
            model: "qwen3-native",
            turns: turns,
            exportedAt: fixedDate
        )
        XCTAssertFalse(md.contains(imageB64Sentinel))
        XCTAssertFalse(md.contains(videoB64Sentinel))
        XCTAssertFalse(md.contains("data:image/png;base64,"))
        XCTAssertTrue(md.contains("# ummInfer Session Export"))
        XCTAssertTrue(md.contains("- **Session ID:** `ummmlx-story-99`"))
        XCTAssertTrue(md.contains("- **Model:** `qwen3-native`"))
        XCTAssertTrue(md.contains("- **Turns:** 4"))
        XCTAssertTrue(md.contains("## Turn 1"))
        XCTAssertTrue(md.contains("#### Visual Slots"))
        XCTAssertTrue(md.contains("- `slot-0` (**Image**) — glowing paper lantern over misty river (`2 KiB`, `sha256: \(imageSHA.prefix(8))`)"))
        XCTAssertTrue(md.contains("- `slot-1` (**Video**, `Explicit I2V ← Slot 1 · \(imageSHA.prefix(8))`) — lantern drifting in river breeze (`4 KiB`, `sha256: \(videoSHA.prefix(8))`)"))
        XCTAssertTrue(md.contains("**Telemetry:** `TTFT 265 ms` · `TTFA 380 ms` · `2 visuals (4.12 s)` · `Explicit I2V` · `Total 8.79 s` · `VRAM: 128.4 MiB peak` · `Behavior: 4 tok (2 sup)`"))
        XCTAssertTrue(md.contains("### Assistant · Interrupted · 42 B spoken"))
        XCTAssertTrue(md.contains("**Telemetry:** `TTFT 120 ms` · `Total 950 ms` · `VRAM: N/C` · `Behavior: not_exposed`"))
        XCTAssertTrue(md.contains("### Assistant · Interrupted before audio"))
        XCTAssertTrue(md.contains("**Telemetry:** `Behavior: absent`"))
        XCTAssertTrue(md.contains("**Telemetry:** `TTFT 90 ms` · `Total 310 ms`"))
    }

    @MainActor
    func testChatModelExportHelpersAndStreamingState() throws {
        let model = ChatModel(audioStream: NoOpAudioStreamPlayer())
        model.model = "qwen3-native"
        XCTAssertTrue(model.turns.isEmpty)
        XCTAssertFalse(model.isStreaming)
        XCTAssertEqual(model.exportSessionFilename(format: .json), "umminfer-session-untitled.json")
        XCTAssertEqual(model.exportSessionFilename(format: .markdown), "umminfer-session-untitled.md")

        let data = model.exportSessionData(format: .json, exportedAt: fixedDate)
        let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(root["schema"] as? String, "umminfer-session-export-v1")
        XCTAssertEqual(root["turn_count"] as? Int, 0)

        model.noteSessionExported(filename: "umminfer-session-untitled.json")
        XCTAssertEqual(model.status, "Exported umminfer-session-untitled.json")
        XCTAssertNil(model.error)

        model.noteSessionExportFailed("Disk full")
        XCTAssertEqual(model.status, "Export error")
        XCTAssertEqual(model.error, "Could not export session: Disk full")
    }

    // MARK: - Cross-client JSON import tests

    func testRoundTripExportImportMacOSSchema() throws {
        let imageBytes = Data(repeating: 0x51, count: 2048)
        let videoBytes = Data(repeating: 0x52, count: 4096)
        let imageSHA = testSHA256Hex(imageBytes)
        let videoSHA = testSHA256Hex(videoBytes)

        let imageArtifact = GeneratedArtifact(
            id: "img-0",
            mediaType: "image",
            mimeType: "image/png",
            data: imageBytes,
            sha256: imageSHA,
            slotID: "slot-0",
            slotIndex: 0
        )
        let videoArtifact = GeneratedArtifact(
            id: "vid-1",
            mediaType: "video",
            mimeType: "video/mp4",
            data: videoBytes,
            sha256: videoSHA,
            slotID: "slot-1",
            slotIndex: 1,
            anchorSlotIndex: 0,
            anchorExplicit: true,
            anchorSha256: imageSHA
        )
        let slot0 = VisualSlotState(
            slotID: "slot-0",
            slotIndex: 0,
            prompt: "lantern keyframe over misty river",
            activeKind: "image",
            isLoading: false,
            imageArtifact: imageArtifact
        )
        let slot1 = VisualSlotState(
            slotID: "slot-1",
            slotIndex: 1,
            prompt: "drifting lantern with water ripples",
            activeKind: "video",
            isLoading: false,
            videoArtifact: videoArtifact,
            continuityMode: NarrativeContinuityMode.i2vChain.rawValue,
            anchorSlotIndex: 0,
            anchorExplicit: true,
            anchorSha256: imageSHA
        )

        let exposedEvidence = BehaviorLogprobsEvidence(
            format: "umminfer-behavior-logprobs-v1",
            tokenizerSha256: "1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016",
            tokenIDs: [10, 20, 30, 40],
            logprobs: [-0.1, -0.2, -0.3, -0.05],
            mask: [1, 0, 1, 0]
        )

        let u1 = ChatTurn(role: .user, text: "Tell a two-beat river story.", state: .complete)
        var a1 = ChatTurn(role: .assistant, text: "A warm lantern drifts past the pines.", state: .complete)
        a1.transcript = "Tell a two-beat river story."
        a1.blocks = [.text(id: UUID(), content: a1.text), .visual(slot0), .visual(slot1)]
        a1.telemetry = TurnTelemetry(
            ttftMs: 240,
            ttfaMs: 360,
            visualSettledCount: 2,
            firstVisualMs: 3800,
            hasI2VChain: true,
            hasExplicitAnchor: true,
            totalMs: 8200,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm",
                        nativeStage: "qwen38",
                        coverage: "COVERED",
                        allocatedBytesPeak: 134_637_158,
                        epochID: 9
                    )
                ]
            ),
            behaviorRequested: true,
            behaviorEvidenceHeader: "exposed",
            behaviorEvidence: exposedEvidence
        )

        let u2 = ChatTurn(role: .user, text: "Interrupt test", state: .complete)
        var a2 = ChatTurn(role: .assistant, text: "Partial spoken", state: .stopped)
        a2.interruptedSpokenBytes = 42
        a2.telemetry = TurnTelemetry(
            ttftMs: 110,
            totalMs: 850,
            nativeAllocator: NativeAllocatorTelemetryEnvelope(
                coverage: "NOT_COVERED",
                stages: [
                    NativeAllocatorStageTelemetry(
                        stage: "vlm",
                        coverage: "NOT_COVERED",
                        allocatedBytesPeak: nil,
                        reason: "batched_request_attribution_unavailable"
                    )
                ]
            ),
            behaviorRequested: true,
            behaviorEvidenceHeader: "not_exposed",
            behaviorEvidence: nil
        )

        let u3 = ChatTurn(role: .user, text: "Cancel before speech", state: .complete)
        var a3 = ChatTurn(role: .assistant, text: "", state: .stopped)
        a3.interruptedSpokenBytes = 0
        a3.telemetry = TurnTelemetry(
            behaviorRequested: true,
            behaviorEvidenceHeader: nil,
            behaviorEvidence: nil
        )

        let exported = SessionExporter.buildJSONData(
            sessionID: "ummmlx-roundtrip-01",
            model: "qwen3-native",
            turns: [u1, a1, u2, a2, u3, a3],
            exportedAt: fixedDate
        )

        let imported = try SessionExporter.importJSON(exported)
        XCTAssertEqual(imported.schema, SessionExporter.schema)
        XCTAssertEqual(imported.exportedAt, "2026-11-26T09:11:14.250Z")
        XCTAssertEqual(imported.sessionID, "ummmlx-roundtrip-01")
        XCTAssertEqual(imported.model, "qwen3-native")
        XCTAssertEqual(imported.turnCount, 3)
        XCTAssertEqual(imported.turns.count, 6)
        XCTAssertEqual(imported.messages.count, 6)
        XCTAssertEqual(imported.messages[0], ChatMessage(role: "user", content: "Tell a two-beat river story."))
        XCTAssertEqual(imported.messages[1], ChatMessage(role: "assistant", content: "A warm lantern drifts past the pines."))

        // Verify assistant 1 visual slots have honest metadata and ZERO fabricated binary bytes.
        let importedA1 = imported.turns[1]
        XCTAssertEqual(importedA1.role, .assistant)
        XCTAssertEqual(importedA1.state, .complete)
        XCTAssertEqual(importedA1.transcript, "Tell a two-beat river story.")
        XCTAssertTrue(importedA1.artifacts.isEmpty, "Import must never fabricate binary GeneratedArtifact data")
        let restoredSlots = importedA1.blocks.compactMap(\.visualSlot)
        XCTAssertEqual(restoredSlots.count, 2)
        XCTAssertEqual(restoredSlots[0].slotID, "slot-0")
        XCTAssertEqual(restoredSlots[0].slotIndex, 0)
        XCTAssertEqual(restoredSlots[0].activeKind, "image")
        XCTAssertNil(restoredSlots[0].imageArtifact)
        XCTAssertNil(restoredSlots[0].videoArtifact)
        XCTAssertEqual(restoredSlots[0].importedBytes, 2048)
        XCTAssertEqual(restoredSlots[0].importedSha256, imageSHA)
        XCTAssertEqual(restoredSlots[0].placeholderLabel, "Imported image (2 KiB · sha256: \(imageSHA.prefix(8)))")

        XCTAssertEqual(restoredSlots[1].slotID, "slot-1")
        XCTAssertEqual(restoredSlots[1].slotIndex, 1)
        XCTAssertEqual(restoredSlots[1].activeKind, "video")
        XCTAssertNil(restoredSlots[1].videoArtifact)
        XCTAssertEqual(restoredSlots[1].continuityMode, "i2v_chain")
        XCTAssertEqual(restoredSlots[1].anchorSlotIndex, 0)
        XCTAssertEqual(restoredSlots[1].anchorExplicit, true)
        XCTAssertEqual(restoredSlots[1].anchorSha256, imageSHA)
        XCTAssertEqual(restoredSlots[1].importedBytes, 4096)
        XCTAssertEqual(restoredSlots[1].importedSha256, videoSHA)

        // Verify telemetry summaries restored honestly without fabricating raw arrays.
        XCTAssertNil(importedA1.telemetry.nativeAllocator)
        XCTAssertNil(importedA1.telemetry.behaviorEvidence)
        XCTAssertEqual(importedA1.telemetry.ttftMs, 240)
        XCTAssertEqual(importedA1.telemetry.ttfaMs, 360)
        XCTAssertEqual(importedA1.telemetry.firstVisualMs, 3800)
        XCTAssertEqual(importedA1.telemetry.totalMs, 8200)
        XCTAssertEqual(
            importedA1.telemetry.formattedPills.map(\.label),
            [
                "TTFT 240 ms",
                "TTFA 360 ms",
                "2 visuals (3.80 s)",
                "Explicit I2V",
                "Total 8.20 s",
                "VRAM: 128.4 MiB peak",
                "Behavior: 4 tok (2 sup)",
            ]
        )

        // Verify assistant 2 interrupted + NOT_COVERED + not_exposed without fabrication.
        let importedA2 = imported.turns[3]
        XCTAssertEqual(importedA2.state, .stopped)
        XCTAssertTrue(importedA2.isInterrupted)
        XCTAssertEqual(importedA2.interruptedSpokenBytes, 42)
        XCTAssertEqual(importedA2.interruptedBadgeText, "Interrupted · 42 B spoken")
        XCTAssertEqual(importedA2.telemetry.importedNativeAllocator?.covered, false)
        XCTAssertNil(importedA2.telemetry.importedNativeAllocator?.peakBytes)
        XCTAssertEqual(importedA2.telemetry.importedBehaviorLogprobs?.status, "not_exposed")
        XCTAssertNil(importedA2.telemetry.importedBehaviorLogprobs?.tokenCount)
        XCTAssertNil(importedA2.telemetry.importedBehaviorLogprobs?.supervisedCount)

        // Verify assistant 3 interrupted before audio + behavior absent.
        let importedA3 = imported.turns[5]
        XCTAssertEqual(importedA3.state, .stopped)
        XCTAssertTrue(importedA3.isInterrupted)
        XCTAssertEqual(importedA3.interruptedSpokenBytes, 0)
        XCTAssertEqual(importedA3.interruptedBadgeText, "Interrupted before audio")
        XCTAssertEqual(importedA3.telemetry.importedBehaviorLogprobs?.status, "absent")
        XCTAssertNil(importedA3.telemetry.importedBehaviorLogprobs?.tokenCount)

        // Re-exporting the imported turns produces identical canonical JSON bytes!
        let reexported = SessionExporter.buildJSONData(
            sessionID: imported.sessionID ?? "",
            model: imported.model ?? "",
            turns: imported.turns,
            exportedAt: fixedDate
        )
        XCTAssertEqual(reexported, exported)
    }

    func testEmptySessionRoundTrip() throws {
        let emptyData = SessionExporter.buildJSONData(
            sessionID: "",
            model: "",
            turns: [],
            exportedAt: fixedDate
        )
        let imported = try SessionExporter.importJSON(emptyData)
        XCTAssertEqual(imported.schema, "umminfer-session-export-v1")
        XCTAssertNil(imported.sessionID)
        XCTAssertNil(imported.model)
        XCTAssertEqual(imported.turnCount, 0)
        XCTAssertTrue(imported.turns.isEmpty)
        XCTAssertTrue(imported.messages.isEmpty)
    }

    func testCrossClientImportWebSchemaV1() throws {
        let slot0SHA = "abcdef01" + String(repeating: "23456789", count: 7)
        let anchorSHA = "3a9d5bbe" + String(repeating: "4f12c890", count: 7)
        let tokenizerSHA = "1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016"
        let webPayload: [String: Any] = [
            "schema": "umminfer-web-session-export-v1",
            "exported_at": "2026-11-26T09:00:00.000Z",
            "session_id": "ummweb-a1b2c3d4e5f60708",
            "model": "qwen3-native",
            "turn_count": 3,
            "turns": [
                [
                    "index": 1,
                    "role": "turn",
                    "prompt": "Create a two-beat storyboard of a glowing lantern.",
                    "response": "A warm lantern floats above the misty river.",
                    "visual_slots": [
                        [
                            "id": "slot-0",
                            "kind": "image",
                            "prompt": "establishing keyframe of glowing paper lantern",
                            "bytes": 131_072,
                            "sha256": slot0SHA,
                        ],
                        [
                            "id": "slot-1",
                            "kind": "video",
                            "prompt": "animate river breeze and reflections",
                            "bytes": 1_572_864,
                            "continuity_mode": "i2v_chain",
                            "anchor_slot_index": 0,
                            "anchor_explicit": true,
                            "anchor_sha256": anchorSHA,
                        ],
                    ],
                    "telemetry": [
                        "ttft_ms": 210,
                        "ttfa_ms": 380,
                        "first_visual_ms": 3200,
                        "total_ms": 9100,
                        "visual_slots": 2,
                        "i2v_count": 1,
                        "native_allocator": [
                            "coverage": "COVERED",
                            "covered": true,
                            "covered_stage_count": 1,
                            "total_stage_count": 1,
                            "peak_bytes": 134_637_158,
                            "label": "VRAM: 128.4 MiB peak",
                            "details": ["vlm/qwen38: COVERED · peak 128.4 MiB · epoch #17"],
                        ],
                        "behavior_logprobs": [
                            "status": "exposed",
                            "token_count": 5,
                            "supervised_count": 3,
                            "label": "Behavior: 5 tok (3 sup)",
                            "format": "umminfer-behavior-logprobs-v1",
                            "tokenizer_sha256": tokenizerSHA,
                        ],
                    ],
                ],
                [
                    "index": 2,
                    "role": "turn",
                    "prompt": "Tell me about ocean tides.",
                    "response": "Ocean tides are driven by",
                    "asr_text": "Tell me about ocean tides.",
                    "interrupted": true,
                    "spoken_bytes": 24,
                    "visual_slots": [],
                ],
                [
                    // Web emits role:"user" when response is empty even on an interrupted turn.
                    "index": 3,
                    "role": "user",
                    "prompt": "Look at data:image/png;base64,AAAA1234== please",
                    "response": "",
                    "interrupted": true,
                    "spoken_bytes": 0,
                    "visual_slots": [],
                ],
            ],
        ]

        let data = try JSONSerialization.data(withJSONObject: webPayload, options: [.sortedKeys])
        let imported = try SessionExporter.importJSON(data)

        XCTAssertEqual(imported.schema, "umminfer-web-session-export-v1")
        XCTAssertEqual(imported.sessionID, "ummweb-a1b2c3d4e5f60708")
        XCTAssertEqual(imported.model, "qwen3-native")
        XCTAssertEqual(imported.turnCount, 3)
        XCTAssertEqual(imported.turns.count, 6)

        // Turn 1: visual slot IDs mapped from Web's "id" without "slot_index".
        let a1 = imported.turns[1]
        let slots = a1.blocks.compactMap(\.visualSlot)
        XCTAssertEqual(slots.count, 2)
        XCTAssertEqual(slots[0].slotID, "slot-0")
        XCTAssertEqual(slots[0].slotIndex, 0)
        XCTAssertEqual(slots[0].activeKind, "image")
        XCTAssertEqual(slots[0].importedBytes, 131_072)
        XCTAssertEqual(slots[0].importedSha256, slot0SHA)
        XCTAssertNil(slots[0].imageArtifact)

        XCTAssertEqual(slots[1].slotID, "slot-1")
        XCTAssertEqual(slots[1].slotIndex, 1)
        XCTAssertEqual(slots[1].activeKind, "video")
        XCTAssertEqual(slots[1].continuityMode, "i2v_chain")
        XCTAssertEqual(slots[1].anchorSlotIndex, 0)
        XCTAssertEqual(slots[1].anchorExplicit, true)
        XCTAssertEqual(slots[1].anchorSha256, anchorSHA)
        XCTAssertEqual(slots[1].importedBytes, 1_572_864)
        XCTAssertNil(slots[1].videoArtifact)

        XCTAssertEqual(
            a1.telemetry.formattedPills.map(\.label),
            [
                "TTFT 210 ms",
                "TTFA 380 ms",
                "2 visuals (3.20 s)",
                "Explicit I2V",
                "Total 9.10 s",
                "VRAM: 128.4 MiB peak",
                "Behavior: 5 tok (3 sup)",
            ]
        )

        // Turn 2: interrupted with spoken_bytes=24 + ASR transcript + zero fabricated telemetry.
        let a2 = imported.turns[3]
        XCTAssertEqual(a2.transcript, "Tell me about ocean tides.")
        XCTAssertEqual(a2.state, .stopped)
        XCTAssertEqual(a2.interruptedSpokenBytes, 24)
        XCTAssertTrue(a2.telemetry.formattedPills.isEmpty)

        // Turn 3: inline data URI scrubbed and interrupted before audio preserved.
        let u3 = imported.turns[4]
        let a3 = imported.turns[5]
        XCTAssertEqual(u3.text, "Look at [omitted-base64] please")
        XCTAssertFalse(u3.text.contains("data:image/png;base64"))
        XCTAssertEqual(a3.state, .stopped)
        XCTAssertEqual(a3.interruptedSpokenBytes, 0)
        XCTAssertEqual(a3.interruptedBadgeText, "Interrupted before audio")
    }

    func testImportMessagesArrayShape() throws {
        let json = """
        {
          "schema": "umminfer-web-session-export-v1",
          "session_id": "msg-sess-7",
          "model": "qwen3-native",
          "messages": [
            { "role": "user", "content": "Hello from messages array." },
            {
              "role": "assistant",
              "content": "Hi there!",
              "telemetry": {
                "ttft_ms": 145,
                "total_ms": 420
              }
            }
          ]
        }
        """
        let imported = try SessionExporter.importJSONString(json)
        XCTAssertEqual(imported.schema, "umminfer-web-session-export-v1")
        XCTAssertEqual(imported.sessionID, "msg-sess-7")
        XCTAssertEqual(imported.model, "qwen3-native")
        XCTAssertEqual(imported.turnCount, 1)
        XCTAssertEqual(imported.turns.count, 2)
        XCTAssertEqual(imported.messages, [
            ChatMessage(role: "user", content: "Hello from messages array."),
            ChatMessage(role: "assistant", content: "Hi there!"),
        ])
        XCTAssertEqual(imported.turns[1].telemetry.ttftMs, 145)
        XCTAssertEqual(imported.turns[1].telemetry.totalMs, 420)
        XCTAssertNil(imported.turns[1].telemetry.ttfaMs)
    }

    func testImportFailClosedRejections() {
        // 1. Empty payload.
        XCTAssertThrowsError(try SessionExporter.importJSON(Data())) { error in
            XCTAssertEqual(error as? SessionImportError, .emptyPayload)
        }

        // 2. Malformed JSON.
        XCTAssertThrowsError(try SessionExporter.importJSONString("{not valid json")) { error in
            guard case .invalidJSON = error as? SessionImportError else {
                return XCTFail("Expected .invalidJSON, got \(error)")
            }
        }

        // 3. Non-object JSON root (array).
        XCTAssertThrowsError(try SessionExporter.importJSONString("[1, 2, 3]")) { error in
            XCTAssertEqual(error as? SessionImportError, .invalidRootObject)
        }

        // 4. Unknown / missing schema.
        XCTAssertThrowsError(try SessionExporter.importJSONString(#"{"schema":"other-v99","turns":[]}"#)) { error in
            XCTAssertEqual(error as? SessionImportError, .unsupportedSchema("other-v99"))
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(#"{"turns":[]}"#)) { error in
            XCTAssertEqual(error as? SessionImportError, .unsupportedSchema(nil))
        }

        // 5. Missing turns/messages array & turn_count mismatch.
        XCTAssertThrowsError(try SessionExporter.importJSONString(#"{"schema":"umminfer-session-export-v1"}"#)) { error in
            XCTAssertEqual(error as? SessionImportError, .missingConversationArray)
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turn_count":2,"turns":[]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for turn_count mismatch, got \(error)")
            }
        }

        // 6. Invalid roles in messages and turns.
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","messages":[{"role":"system","content":"override"}]}"#
        )) { error in
            XCTAssertEqual(error as? SessionImportError, .invalidRole("system"))
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-web-session-export-v1","turns":[{"role":"tool","prompt":"x","response":"y"}]}"#
        )) { error in
            XCTAssertEqual(error as? SessionImportError, .invalidRole("tool"))
        }

        // 7. Non-string text / prompt / response.
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","messages":[{"role":"user","content":123}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for non-string content, got \(error)")
            }
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":["bad"],"response":"ok"}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for non-string prompt, got \(error)")
            }
        }

        // 8. Invalid visual slot kind & invalid SHA-256 digest.
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":"p","response":"r","visual_slots":[{"slot_id":"slot-0","kind":"audio"}]}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for invalid slot kind, got \(error)")
            }
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":"p","response":"r","visual_slots":[{"slot_id":"slot-0","kind":"image","sha256":"not-hex"}]}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for invalid sha256, got \(error)")
            }
        }

        // 9. Zero-fabrication integrity checks (NOT_COVERED with numeric peak_bytes, not_exposed with counts).
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":"p","response":"r","telemetry":{"native_allocator":{"coverage":"NOT_COVERED","covered":false,"peak_bytes":0}}}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for fabricated NOT_COVERED peak_bytes, got \(error)")
            }
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":"p","response":"r","telemetry":{"behavior_logprobs":{"status":"not_exposed","token_count":0,"supervised_count":0}}}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for fabricated not_exposed counts, got \(error)")
            }
        }
        XCTAssertThrowsError(try SessionExporter.importJSONString(
            #"{"schema":"umminfer-session-export-v1","turns":[{"prompt":"p","response":"r","telemetry":{"behavior_logprobs":{"status":"exposed","token_count":2,"supervised_count":5}}}]}"#
        )) { error in
            guard case .invalidField = error as? SessionImportError else {
                return XCTFail("Expected .invalidField for supervised_count > token_count, got \(error)")
            }
        }

        // 10. > 2 MiB payload rejected BEFORE decoding (even if bytes are not valid JSON).
        let oversized = Data(repeating: 0x7B, count: SessionExporter.maxImportBytes + 1)
        XCTAssertThrowsError(try SessionExporter.importJSON(oversized)) { error in
            XCTAssertEqual(
                error as? SessionImportError,
                .payloadTooLarge(bytes: SessionExporter.maxImportBytes + 1, limit: SessionExporter.maxImportBytes)
            )
        }
    }

    @MainActor
    func testChatModelImportZeroNetworkAndPostImportWorkflow() throws {
        ImportGuardURLProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ImportGuardURLProtocol.self]
        let session = URLSession(configuration: config)
        let client = ChatClient(session: session)
        let model = ChatModel(client: client, audioStream: NoOpAudioStreamPlayer())
        model.endpoint = "http://10.9.10.1/v1"
        model.model = "original-model"

        var outboundRequests: [[ChatMessage]] = []
        model.onRequest = { messages in
            outboundRequests.append(messages)
        }

        let webJSON = """
        {
          "schema": "umminfer-web-session-export-v1",
          "exported_at": "2026-11-26T09:15:00.000Z",
          "session_id": "web-imported-42",
          "model": "qwen3-native",
          "turn_count": 1,
          "turns": [
            {
              "index": 1,
              "role": "turn",
              "prompt": "What is 6 * 7?",
              "response": "42.",
              "visual_slots": [],
              "telemetry": {
                "ttft_ms": 100,
                "total_ms": 250,
                "visual_slots": 0,
                "i2v_count": 0
              }
            }
          ]
        }
        """

        let imported = try model.importSessionJSON(Data(webJSON.utf8), filename: "web-42.json")
        XCTAssertEqual(imported.turnCount, 1)
        XCTAssertEqual(ImportGuardURLProtocol.requestCount, 0, "Import must never trigger HTTP traffic")
        XCTAssertTrue(outboundRequests.isEmpty, "Import must never call ChatModel.onRequest")
        XCTAssertEqual(model.sessionID, "web-imported-42")
        XCTAssertEqual(model.model, "qwen3-native")
        XCTAssertEqual(model.turns.count, 2)
        XCTAssertEqual(model.turns[0].text, "What is 6 * 7?")
        XCTAssertEqual(model.turns[1].text, "42.")
        XCTAssertEqual(model.status, "Imported web-42.json (1 turn)")
        XCTAssertNil(model.error)

        // Invalid import while idle leaves existing imported state completely untouched!
        XCTAssertThrowsError(try model.importSessionJSON(Data("{broken".utf8)))
        XCTAssertEqual(model.sessionID, "web-imported-42")
        XCTAssertEqual(model.model, "qwen3-native")
        XCTAssertEqual(model.turns.count, 2)
        XCTAssertEqual(model.turns[0].text, "What is 6 * 7?")
        XCTAssertEqual(model.status, "Import error")
        XCTAssertNotNil(model.error)

        // Export works cleanly after import.
        let exportedAfterImport = model.exportSessionData(format: .json, exportedAt: fixedDate)
        let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: exportedAfterImport) as? [String: Any])
        XCTAssertEqual(root["schema"] as? String, "umminfer-session-export-v1")
        XCTAssertEqual(root["session_id"] as? String, "web-imported-42")
        XCTAssertEqual(root["turn_count"] as? Int, 1)

        // Follow-up history replay includes imported turns cleanly.
        let nextHistory = ChatModel.requestMessages(turns: model.turns, prompt: "Add 1.")
        XCTAssertEqual(nextHistory, [
            ChatMessage(role: "user", content: "What is 6 * 7?"),
            ChatMessage(role: "assistant", content: "42."),
            ChatMessage(role: "user", content: "Add 1."),
        ])

        // New Chat clears imported state cleanly.
        model.newChat()
        XCTAssertTrue(model.turns.isEmpty)
        XCTAssertEqual(model.sessionID, "")
        XCTAssertEqual(model.status, "Ready")
        XCTAssertNil(model.error)
    }

    @MainActor
    func testChatModelImportRejectedWhileStreamingOrLiveVoiceWithoutMutation() async throws {
        ImportGuardURLProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ImportGuardURLProtocol.self]
        let session = URLSession(configuration: config)
        let client = ChatClient(session: session)
        let realtime = RealtimeClient(
            transportFactory: { StubIdleRealtimeTransport() },
            audioCapturer: StubIdleAudioCapturer()
        )
        let model = ChatModel(
            client: client,
            audioStream: NoOpAudioStreamPlayer(),
            realtime: realtime
        )
        model.endpoint = "http://10.9.10.2/v1"
        model.model = "active-model"

        let validImport = Data("""
        {
          "schema": "umminfer-session-export-v1",
          "session_id": "incoming-sess",
          "model": "incoming-model",
          "turn_count": 1,
          "turns": [{ "index": 1, "prompt": "New", "response": "Turn", "visual_slots": [] }]
        }
        """.utf8)

        // 1. Streaming / running busy guard.
        model.prompt = "Keep this active turn"
        model.send()
        XCTAssertTrue(model.isRunning)
        XCTAssertTrue(model.isStreaming)
        XCTAssertTrue(model.isSessionImportBlocked)
        let turnsBeforeStreamingImport = model.turns.map(\.id)
        let statusBeforeStreamingImport = model.status

        XCTAssertThrowsError(try model.importSessionJSON(validImport)) { error in
            guard case .sessionBusy = error as? SessionImportError else {
                return XCTFail("Expected .sessionBusy while streaming, got \(error)")
            }
        }
        XCTAssertEqual(model.turns.map(\.id), turnsBeforeStreamingImport, "Streaming turns must remain untouched")
        XCTAssertEqual(model.turns.first?.text, "Keep this active turn")
        XCTAssertEqual(model.model, "active-model")
        XCTAssertEqual(model.sessionID, "")
        XCTAssertEqual(model.status, statusBeforeStreamingImport)

        model.stop()
        try await Task.sleep(nanoseconds: 50_000_000)
        model.newChat()
        XCTAssertFalse(model.isSessionImportBlocked)

        // 2. Live Voice busy guard.
        model.startLiveVoice()
        XCTAssertTrue(model.isLiveVoiceActive)
        XCTAssertTrue(model.isSessionImportBlocked)
        let turnsBeforeVoiceImport = model.turns.map(\.id)

        XCTAssertThrowsError(try model.importSessionJSON(validImport)) { error in
            guard case .sessionBusy = error as? SessionImportError else {
                return XCTFail("Expected .sessionBusy while Live Voice is active, got \(error)")
            }
        }
        XCTAssertEqual(model.turns.map(\.id), turnsBeforeVoiceImport, "Live Voice turns must remain untouched")
        XCTAssertEqual(model.model, "active-model")
        XCTAssertEqual(model.sessionID, "")

        model.stopLiveVoice()
        XCTAssertFalse(model.isSessionImportBlocked)

        // Once idle again, the same valid payload imports cleanly.
        let imported = try model.importSessionJSON(validImport)
        XCTAssertEqual(imported.sessionID, "incoming-sess")
        XCTAssertEqual(model.sessionID, "incoming-sess")
        XCTAssertEqual(model.model, "incoming-model")
        XCTAssertEqual(model.turns.count, 2)
    }
}

// MARK: - Test doubles for zero-network & busy-state import tests

private final class ImportGuardURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _requestCount = 0

    static var requestCount: Int {
        lock.withLock { _requestCount }
    }

    static func reset() {
        lock.withLock { _requestCount = 0 }
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasPrefix("10.9.10.") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.withLock { Self._requestCount += 1 }
        // Intentionally hold open until cancelled to exercise the streaming busy guard.
    }

    override func stopLoading() {}
}

private final class StubIdleRealtimeTransport: RealtimeTransport, @unchecked Sendable {
    func connect(url: URL) async throws {}
    func send(text: String) async throws {}
    func receive() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { _ in }
    }
    func close() async {}
}

private final class StubIdleAudioCapturer: RealtimeAudioCapturing, @unchecked Sendable {
    func startCapture() -> AsyncThrowingStream<Data, Error> {
        AsyncThrowingStream { _ in }
    }
    func stopCapture() {}
}
