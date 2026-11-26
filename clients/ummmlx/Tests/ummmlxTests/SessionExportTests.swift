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
}
