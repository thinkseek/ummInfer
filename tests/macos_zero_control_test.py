#!/usr/bin/env python3
"""Offline source contracts for the macOS natural zero-control defaults."""

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SWIFT_ROOT = REPO_ROOT / "clients" / "ummmlx"
APP = SWIFT_ROOT / "Sources" / "ummmlx" / "App.swift"
MODEL = SWIFT_ROOT / "Sources" / "ummmlx" / "ChatModel.swift"
CLIENT = SWIFT_ROOT / "Sources" / "ummmlx" / "ChatClient.swift"
AUTOMATION = SWIFT_ROOT / "Sources" / "ummmlx" / "Automation.swift"
UNIFIED_TESTS = SWIFT_ROOT / "Tests" / "ummmlxTests" / "ChatClientUnifiedTests.swift"


def source(path):
    return path.read_text(encoding="utf-8")


def section(text, start, end):
    if start not in text or end not in text:
        raise AssertionError(f"missing source section markers: {start!r} / {end!r}")
    return text.split(start, 1)[1].split(end, 1)[0]


class MacOSZeroControlSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = source(APP)
        cls.model = source(MODEL)
        cls.client = source(CLIENT)
        cls.automation = source(AUTOMATION)
        cls.unified_tests = source(UNIFIED_TESTS)

    def test_advanced_defaults_collapsed_and_contains_developer_controls(self):
        self.assertIn("@State private var showAdvanced = false", self.app)
        self.assertIn('Button(showAdvanced ? "Hide Advanced" : "Advanced")', self.app)
        self.assertIn("showAdvanced.toggle()", self.app)
        self.assertIn('.accessibilityIdentifier("advanced-options")', self.app)

        settings = section(self.app, "private var settings:", "private func labeledField")
        self.assertIn("if showAdvanced {", settings)
        visible = settings.split("if showAdvanced {", 1)[0]
        advanced = settings.split("if showAdvanced {", 1)[1]
        for control in ("endpoint", "model", "advanced-options"):
            self.assertIn(f'"{control}"', visible)
        for control in (
            "image-output", "video-output", "fastVideo", "audio-output",
            "live-voice-echo-suppression", "videoFramePolicy", "videoFrameBudget",
        ):
            self.assertNotIn(f'"{control}"', visible)
            self.assertIn(f'"{control}"', advanced)
        for binding in (
            "$model.maxTokensText", "$model.temperatureText", "$model.topPText",
            "$model.seedText", "$model.videoMotionThresholdText",
            "$model.videoSceneThresholdText",
        ):
            self.assertNotIn(binding, visible)
            self.assertIn(binding, advanced)

    def test_live_voice_input_stays_visible_and_echo_updates_remain_available(self):
        header = section(self.app, "private var header:", "private var settings:")
        composer = section(self.app, "private var composer:", "private func scrollToEnd")
        self.assertIn('"live-voice"', header)
        self.assertNotIn("audio-output", header)
        self.assertNotIn("live-voice-echo-suppression", header)
        for control in ("prompt", "mic", "send"):
            self.assertIn(f'"{control}"', composer)

        settings = section(self.app, "private var settings:", "private func labeledField")
        advanced = settings.split("if showAdvanced {", 1)[1]
        self.assertIn("Toggle(isOn: liveVoiceEchoSuppressionBinding)", advanced)
        self.assertIn(".disabled(model.isRunning || model.isListening)", advanced)
        self.assertIn("works before connecting and live mid-response", advanced)
        self.assertIn("setLiveVoiceEchoSuppression($0)", self.app)

    def test_temporal_preset_still_reveals_advanced_controls(self):
        preset = section(self.app, "private var showcasePresetsView:", "private func telemetryStrip")
        self.assertIn("if preset.videoFramePolicy == .temporal", preset)
        self.assertIn("showAdvanced = true", preset)
        self.assertIn("promptFocused = true", preset)

    def test_natural_defaults_and_existing_wire_hooks(self):
        for declaration in (
            "@Published var imageOutput = true",
            "@Published var videoOutput = true",
            "@Published var maxTokensText = \"512\"",
            "@Published var narrativeContinuityMode: NarrativeContinuityMode = .i2vChain",
            "@Published var narrativeMode = true",
            "@Published var narrativeVisualMode = \"auto\"",
            "@Published var narrativeMaxSlots = 3",
            "@Published var narrativeAllowTextOnly = true",
            "@Published var audioOutput = true",
            "@Published var fastVideoMode = false",
        ):
            self.assertIn(declaration, self.model)
        for hook in (
            "options.maxTokens = maxTok", "options.seed = seed",
            "options.image = img", "options.video = vid",
            "options.narrative = NarrativeOptions(",
        ):
            self.assertIn(hook, self.model)
        for hook in (
            'body["modalities"] = options.modalities',
            'object["continuity_mode"] = narrative.continuityMode.rawValue',
            'if let topP = options.topP { body["top_p"] = topP }',
            'if let seed = options.seed { body["seed"] = seed }',
        ):
            self.assertIn(hook, self.client)

    def test_automation_pins_legacy_request_defaults(self):
        prefix = self.automation.split("// Server-free fault regression.", 1)[0]
        for assignment in (
            "model.imageOutput = false",
            "model.videoOutput = false",
            "model.audioOutput = false",
            "model.narrativeMode = true",
            "model.narrativeContinuityMode = .off",
            'model.maxTokensText = "256"',
        ):
            self.assertIn(assignment, prefix)
        multimodal = section(self.automation, "if multimodal {", "} else if audioOnly {")
        self.assertIn("model.imageOutput = true", multimodal)
        self.assertIn("model.audioOutput = true", multimodal)
        audio_only = section(self.automation, "} else if audioOnly {", "} else if !prompt.isEmpty {")
        self.assertIn("model.audioOutput = true", audio_only)

    def test_no_preset_captured_send_covers_natural_wire_defaults(self):
        name = "func testNaturalDefaultChatModelSendWithoutPresetCapturesAutomaticNarrative()"
        test = section(self.unified_tests, name, "\n    @MainActor")
        self.assertNotIn("applyShowcasePreset", test)
        self.assertIn("UnifiedSSEProtocol.bodies()", test)
        for assertion in (
            'body["max_tokens"] as? Int, 512',
            'body["modalities"] as? [String], ["text", "image", "audio", "video"]',
            'audio["voice"] as? String, "vivian"',
            'audio["format"] as? String, "wav"',
            'audio["stream"] as? Bool, true',
            'narrative["visual_mode"] as? String, "auto"',
            'narrative["max_slots"] as? Int, 3',
            'narrative["continuity_mode"] as? String, "i2v_chain"',
        ):
            self.assertIn(assertion, test)


if __name__ == "__main__":
    unittest.main()
