// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import AVFoundation
import AVKit
import UniformTypeIdentifiers

@main
struct UmmMLXApp: App {
    @StateObject private var model = ChatModel()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("ummmlx") {
            RootView(model: model)
        }
        .defaultSize(width: 900, height: 680)
        .windowResizability(.contentMinSize)
        .commands {
            SessionCommands(model: model)
        }
    }
}

struct SessionCommands: Commands {
    @ObservedObject var model: ChatModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New chat") {
                model.newChat()
            }
            .keyboardShortcut("n", modifiers: .command)
        }
        CommandGroup(replacing: .importExport) {
            Button("Import JSON…") {
                SessionFilePanels.importSession(into: model)
            }
            .keyboardShortcut("o", modifiers: .command)
            .disabled(model.isSessionImportBlocked)
            Divider()
            Button("Export JSON…") {
                SessionFilePanels.exportSession(from: model, format: .json)
            }
            .keyboardShortcut("e", modifiers: .command)
            .disabled(model.turns.isEmpty || model.isStreaming)
            Button("Export Markdown…") {
                SessionFilePanels.exportSession(from: model, format: .markdown)
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(model.turns.isEmpty || model.isStreaming)
        }
    }
}

@MainActor
enum SessionFilePanels {
    static func importSession(into model: ChatModel) {
        guard !model.isSessionImportBlocked else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            if let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
               fileSize > SessionExporter.maxImportBytes {
                let err = SessionImportError.payloadTooLarge(
                    bytes: fileSize,
                    limit: SessionExporter.maxImportBytes
                )
                model.noteSessionImportFailed(err.localizedDescription)
                return
            }
            let data = try Data(contentsOf: url)
            _ = try model.importSessionJSON(data, filename: url.lastPathComponent)
        } catch is SessionImportError {
            // ChatModel.importSessionJSON already surfaced the validation error.
        } catch {
            model.noteSessionImportFailed(error.localizedDescription)
        }
    }

    static func exportSession(from model: ChatModel, format: SessionExportFormat) {
        guard !model.turns.isEmpty, !model.isStreaming else { return }
        let data = model.exportSessionData(format: format)
        let panel = NSSavePanel()
        panel.nameFieldStringValue = model.exportSessionFilename(format: format)
        let markdownType = UTType(filenameExtension: "md") ?? .plainText
        panel.allowedContentTypes = [format == .json ? .json : markdownType]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try data.write(to: url, options: .atomic)
            model.noteSessionExported(filename: url.lastPathComponent)
        } catch {
            model.noteSessionExportFailed(error.localizedDescription)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        !ProcessInfo.processInfo.arguments.contains("--automation")
    }
}

struct RootView: View {
    @ObservedObject var model: ChatModel
    @AppStorage("appearance") private var appearance = "system"
    @State private var automatedAppearance: String?
    @State private var automation: Automation?
    @State private var preview: GeneratedArtifact?
    @State private var showAdvanced = false
    @FocusState private var promptFocused: Bool
    @Environment(\.colorScheme) private var colorScheme
    private let isAutomation = ProcessInfo.processInfo.arguments.contains("--automation")

    private var theme: Theme { Theme(dark: colorScheme == .dark) }
    private var preferredScheme: ColorScheme? {
        switch automatedAppearance ?? appearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }
    private var appearanceBinding: Binding<String> {
        guard isAutomation else { return $appearance }
        return Binding(
            get: { automatedAppearance ?? appearance },
            set: { automatedAppearance = $0 }
        )
    }
    private var liveVoiceEchoSuppressionBinding: Binding<Bool> {
        Binding(
            get: { model.liveVoiceEchoSuppression },
            set: { model.setLiveVoiceEchoSuppression($0) }
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            settings
            Divider().overlay(theme.line)
            transcript
            if let error = model.error { errorBanner(error) }
            Divider().overlay(theme.line)
            composer
        }
        .frame(minWidth: 660, minHeight: 500)
        .background(theme.background)
        .foregroundStyle(theme.text)
        .tint(theme.accent)
        .preferredColorScheme(preferredScheme)
        .sheet(item: $preview) { artifact in
            fullImage(artifact)
        }
        .onAppear {
            if automation == nil {
                automation = Automation.requested(model: model, appearance: appearanceBinding)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Text("ummmlx")
                .font(.system(size: 16, weight: .semibold))
            Circle().fill((model.isRunning || model.isLiveVoiceActive) ? theme.accent : theme.tertiary)
                .frame(width: 7, height: 7)
                .accessibilityHidden(true)
            Text(model.status)
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.secondary)
                .accessibilityIdentifier("status")
            if !model.sessionID.isEmpty {
                Text("session \(model.sessionShort)")
                    .font(.system(size: theme.captionSize, design: .monospaced))
                    .foregroundStyle(theme.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .accessibilityIdentifier("session")
                    .help("Full session id: \(model.sessionID)")
            }
            Button {
                model.isLiveVoiceActive ? model.stopLiveVoice() : model.startLiveVoice()
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: model.isLiveVoiceActive
                          ? "waveform.circle.fill" : "waveform.circle")
                        .accessibilityHidden(true)
                    Text(model.isLiveVoiceActive ? "End Live Voice" : "Live Voice")
                        .font(.system(size: theme.captionSize, weight: .medium))
                }
            }
            .buttonStyle(.bordered)
            .tint(model.isLiveVoiceActive ? theme.danger : theme.accent)
            .controlSize(.small)
            .disabled(model.isRunning || model.isListening)
            .help("Opt-in full-duplex WebSocket voice session with server VAD, playback ack, and barge-in")
            .accessibilityIdentifier("live-voice")
            if model.isStreamingAudioPlaying {
                HStack(spacing: 5) {
                    Image(systemName: "waveform")
                        .foregroundStyle(theme.accent)
                        .accessibilityHidden(true)
                    Text("Speaking")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                }
                .accessibilityIdentifier("voice-speaking")
                Button("Stop Voice") { model.stopStreamingAudio() }
                    .controlSize(.small)
                    .accessibilityIdentifier("voice-stop")
            }
            Spacer()
            Picker("Theme", selection: appearanceBinding) {
                Text("System").tag("system")
                Text("Light").tag("light")
                Text("Dark").tag("dark")
            }
            .pickerStyle(.menu)
            .fixedSize()
            Button("Import JSON") {
                importSession()
            }
            .fixedSize()
            .keyboardShortcut("o", modifiers: .command)
            .disabled(model.isSessionImportBlocked)
            .help("Import a previously exported ummInfer macOS or Web session JSON file without contacting the server")
            .accessibilityIdentifier("importSessionJSON")
            Menu("Export") {
                Button("Export JSON…") {
                    exportSession(format: .json)
                }
                .keyboardShortcut("e", modifiers: .command)
                .disabled(model.turns.isEmpty || model.isStreaming)
                .accessibilityIdentifier("exportSessionJSON")
                Button("Export Markdown…") {
                    exportSession(format: .markdown)
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(model.turns.isEmpty || model.isStreaming)
                .accessibilityIdentifier("exportSessionMarkdown")
            }
            .fixedSize()
            .disabled(model.turns.isEmpty || model.isStreaming)
            .help("Export session transcript, visual metadata, and observed telemetry as JSON or Markdown")
            .accessibilityIdentifier("exportSession")
            Button("New chat") { model.newChat() }
                .keyboardShortcut("n", modifiers: .command)
                .accessibilityIdentifier("new-chat")
        }
        .padding(16)
        .background(theme.surface)
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .bottom, spacing: 12) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Base URL")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                    TextField("http://127.0.0.1:43301/v1", text: $model.endpoint)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .accessibilityLabel("Base URL")
                        .accessibilityIdentifier("endpoint")
                }
                .frame(maxWidth: .infinity)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                VStack(alignment: .leading, spacing: 5) {
                    Text("Model")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                    TextField("Model served by ummInfer", text: $model.model)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .accessibilityLabel("Model")
                        .accessibilityIdentifier("model")
                }
                .frame(width: 210)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                Button(showAdvanced ? "Hide Advanced" : "Advanced") {
                    showAdvanced.toggle()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.bottom, 2)
                .accessibilityIdentifier("advanced-options")
            }
            if showAdvanced {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        Toggle("Image", isOn: $model.imageOutput)
                            .toggleStyle(.checkbox)
                            .accessibilityIdentifier("image-output")
                        Toggle("Video", isOn: $model.videoOutput)
                            .toggleStyle(.checkbox)
                            .accessibilityIdentifier("video-output")
                        Toggle("Fast Video", isOn: $model.fastVideoMode)
                            .toggleStyle(.checkbox)
                            .help("Enable Wan2.2 TeaCache (0.05) for sub-10s video generation")
                            .accessibilityIdentifier("fastVideo")
                        Toggle(isOn: $model.audioOutput) {
                            Text("Voice").font(.system(size: theme.captionSize))
                        }
                        .toggleStyle(.checkbox)
                        .help("Stream spoken audio while text and visual legs render")
                        .accessibilityIdentifier("audio-output")
                    }
                    HStack(spacing: 10) {
                        labeledField("Max tok", text: $model.maxTokensText, width: 62)
                        labeledField("Temp", text: $model.temperatureText, width: 48)
                        labeledField("top_p", text: $model.topPText, width: 52, placeholder: "1.0")
                        labeledField("Seed", text: $model.seedText, width: 72, placeholder: "auto")
                        labeledField("Voice", text: $model.audioVoice, width: 76)
                        Button(model.refAudioURL == nil ? "Ref WAV…" : "Ref: \(model.refAudioURL!.lastPathComponent)") {
                            if model.refAudioURL != nil {
                                model.refAudioURL = nil
                            } else {
                                model.refAudioURL = pickFile(types: [.wav])
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    HStack(spacing: 10) {
                        Picker("Img", selection: $model.imageSize) {
                            Text("512×512").tag("512x512")
                            Text("1024×1024").tag("1024x1024")
                        }
                        .frame(width: 120)
                        labeledField("Steps", text: $model.imageStepsText, width: 46, placeholder: "4")
                        Button(model.editImageURL == nil ? "Edit Img…" : "Edit: \(model.editImageURL!.lastPathComponent)") {
                            if model.editImageURL != nil {
                                model.editImageURL = nil
                            } else {
                                model.editImageURL = pickFile(types: [.png, .jpeg, .webP])
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        Divider().frame(height: 18)
                        Picker("Vid", selection: $model.videoSize) {
                            Text("480×320").tag("480x320")
                            Text("704×480").tag("704x480")
                            Text("1280×704").tag("1280x704")
                        }
                        .frame(width: 120)
                        labeledField("Frames", text: $model.videoFramesText, width: 46)
                        labeledField("Steps", text: $model.videoStepsText, width: 46)
                        labeledField("CFG", text: $model.videoGuidanceText, width: 46)
                    }
                    HStack(spacing: 10) {
                        Picker("Input frames", selection: $model.videoFramePolicy) {
                            Text("Uniform").tag("uniform")
                            Text("Temporal").tag("temporal")
                        }
                        .frame(width: 190)
                        .help("Frame selection for attached video understanding, not generated-video Fast Video")
                        .accessibilityIdentifier("videoFramePolicy")
                        Picker("Budget", selection: $model.videoFrameBudget) {
                            Text("Default (8, omitted)").tag("")
                            Text("4 · lower cost").tag("4")
                            Text("8 · explicit").tag("8")
                            Text("16 · higher cost").tag("16")
                        }
                        .frame(width: 190)
                        .help("Optional decoded-frame budget for attached video understanding. 4 lowers prefill/ViT cost; 16 increases it. Budget controls selection cost and coverage, not answer quality.")
                        .accessibilityIdentifier("videoFrameBudget")
                        labeledField("Motion ≥", text: $model.videoMotionThresholdText,
                                     width: 52, placeholder: "0.02")
                            .disabled(model.videoFramePolicy != "temporal")
                            .accessibilityIdentifier("videoMotionThreshold")
                        labeledField("Scene ≥", text: $model.videoSceneThresholdText,
                                     width: 52, placeholder: "0.25")
                            .disabled(model.videoFramePolicy != "temporal")
                            .accessibilityIdentifier("videoSceneThreshold")
                        Text("Budget changes selection cost, not quality; thresholds are finite values in [0,1].")
                            .font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.tertiary)
                    }
                    HStack(spacing: 10) {
                        Toggle("Narrative", isOn: $model.narrativeMode)
                            .toggleStyle(.checkbox)
                            .accessibilityIdentifier("narrative")
                        Picker("Visual", selection: $model.narrativeVisualMode) {
                            Text("Auto").tag("auto")
                            Text("Image").tag("image")
                            Text("Video").tag("video")
                        }
                        .frame(width: 132)
                        .disabled(!model.narrativeMode || model.isRunning || model.isLiveVoiceActive)
                        Stepper("Slots \(model.narrativeMaxSlots)",
                                value: $model.narrativeMaxSlots, in: 1...6)
                            .fixedSize()
                            .disabled(!model.narrativeMode || model.isRunning || model.isLiveVoiceActive)
                        Toggle("Smart Text-Only", isOn: $model.narrativeAllowTextOnly)
                            .toggleStyle(.checkbox)
                            .disabled(!model.narrativeMode || model.isRunning || model.isLiveVoiceActive)
                            .help("In Narrative mode, allow pure-text responses for math, code, or factual prompts without redundant visuals")
                            .accessibilityIdentifier("narrativeAllowTextOnly")
                        Text("Interleaved visual beats need Image or Video enabled.")
                            .font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.tertiary)
                    }
                    HStack(spacing: 10) {
                        Picker("Continuity", selection: $model.narrativeContinuityMode) {
                            ForEach(NarrativeContinuityMode.allCases, id: \.self) { mode in
                                Text(mode.displayLabel).tag(mode)
                            }
                        }
                        .frame(width: 196)
                        .disabled(!model.narrativeMode || model.isRunning || model.isLiveVoiceActive)
                        .help("Cross-slot visual continuity across narrative visual beats")
                        .accessibilityIdentifier("narrativeContinuity")
                        labeledField("Style anchor", text: $model.narrativeStyleAnchor,
                                     width: 240, placeholder: "Optional style cue (≤256B)")
                            .disabled(!model.narrativeMode || model.isRunning || model.isLiveVoiceActive)
                            .help("Optional cross-slot style cue (max 256 UTF-8 bytes)")
                            .accessibilityIdentifier("narrativeStyleAnchor")
                        Toggle("Native VRAM telemetry", isOn: $model.nativeAllocatorTelemetry)
                            .toggleStyle(.checkbox)
                            .help("Opt in to per-request native CUDA allocator peak telemetry (telemetry.native_allocator=true); omitted when off")
                            .accessibilityIdentifier("nativeAllocatorTelemetry")
                        Toggle("Behavior logprobs", isOn: $model.behaviorLogprobs)
                            .toggleStyle(.checkbox)
                            .help("Opt in to per-request sampled behavior token/logprob evidence (behavior.logprobs=true; requires streaming); omitted when off")
                            .accessibilityIdentifier("behaviorLogprobs")
                    }
                }
                .disabled(model.isRunning || model.isLiveVoiceActive)
                .padding(.top, 4)
                Toggle(isOn: liveVoiceEchoSuppressionBinding) {
                    Text("Echo suppression").font(.system(size: theme.captionSize))
                }
                .toggleStyle(.checkbox)
                .disabled(model.isRunning || model.isListening)
                .help("Server-side echo suppression for Full-Duplex Live Voice (works before connecting and live mid-response)")
                .accessibilityIdentifier("live-voice-echo-suppression")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(theme.surface)
    }

    private func labeledField(_ label: String, text: Binding<String>, width: CGFloat,
                              placeholder: String = "") -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    private func pickFile(types: [UTType]) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func saveArtifact(_ artifact: GeneratedArtifact, defaultName: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = defaultName
        if panel.runModal() == .OK, let url = panel.url {
            try? artifact.data.write(to: url)
        }
    }

    private func importSession() {
        SessionFilePanels.importSession(into: model)
    }

    private func exportSession(format: SessionExportFormat) {
        SessionFilePanels.exportSession(from: model, format: format)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if model.turns.isEmpty {
                    VStack(spacing: 20) {
                        VStack(spacing: 10) {
                            Image(systemName: "text.bubble")
                                .font(.system(size: 30, weight: .light))
                                .foregroundStyle(theme.accent)
                            Text("Chat with your ummInfer unified server")
                                .font(.system(size: theme.sectionSize, weight: .medium))
                            Text("Text · image · audio · video · voice input")
                                .font(.system(size: theme.bodySize))
                                .foregroundStyle(theme.secondary)
                        }
                        showcasePresetsView
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 72)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 24)
                } else {
                    LazyVStack(spacing: 16) {
                        ForEach(model.turns) { turn in
                            messageRow(turn).id(turn.id)
                        }
                    }
                    .padding(20)
                }
            }
            .onChange(of: model.turns.count) { _, _ in scrollToEnd(proxy) }
            .onChange(of: model.turns.last?.text) { _, _ in scrollToEnd(proxy) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func messageRow(_ turn: ChatTurn) -> some View {
        HStack(alignment: .top) {
            if turn.role == .user { Spacer(minLength: 90) }
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 7) {
                    Text(turn.role == .user ? "You" : "ummInfer")
                        .font(.system(size: theme.captionSize, weight: .medium))
                        .foregroundStyle(theme.secondary)
                    if turn.state == .streaming && turn.text.isEmpty
                        && turn.artifacts.isEmpty && turn.blocks.isEmpty {
                        ProgressView().controlSize(.small)
                    }
                    if let interruptedBadge = turn.interruptedBadgeText {
                        Text(interruptedBadge)
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(theme.danger)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(theme.danger.opacity(0.10))
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(theme.danger.opacity(0.35)))
                            .accessibilityIdentifier("turn-interrupted-badge")
                            .accessibilityLabel(interruptedBadge)
                    } else if turn.state == .stopped {
                        Text("Stopped").font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.secondary)
                    }
                    if turn.state == .failed {
                        Text("Incomplete").font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.danger)
                    }
                }
                if turn.blocks.isEmpty {
                    if !turn.text.isEmpty {
                        textBubble(turn.text, role: turn.role)
                    }
                } else {
                    ForEach(turn.blocks) { block in
                        switch block {
                        case .text(_, let content):
                            if !content.isEmpty {
                                textBubble(content, role: .assistant)
                            }
                        case .visual(let slot):
                            visualSlotCard(slot, turnID: turn.id)
                        }
                    }
                }
                if !turn.transcript.isEmpty {
                    Text("Server transcript: \(turn.transcript)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                        .accessibilityIdentifier("server-transcript")
                }
                let liveAudioActive = model.streamingAudioTurnID == turn.id && model.isStreamingAudioPlaying
                let hasAudioArtifact = turn.artifacts.contains { $0.mediaType == "audio" }
                // Slot artifacts render inside their block card; audio and
                // unattributed artifacts keep the plain row.
                let rowArtifacts = turn.artifacts.filter { $0.slotID == nil }
                if !rowArtifacts.isEmpty || liveAudioActive {
                    HStack(alignment: .top, spacing: 10) {
                        ForEach(rowArtifacts) { artifact in
                            artifactView(artifact, liveStreaming: liveAudioActive && artifact.mediaType == "audio")
                                .id(artifact.id)
                        }
                        if liveAudioActive && !hasAudioArtifact {
                            StreamingAudioView(theme: theme) {
                                model.stopStreamingAudio()
                            }
                        }
                    }
                }
                ForEach(turn.legErrors, id: \.self) { detail in
                    Text("Output leg: \(detail)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.danger)
                }
                let telemetryPills = turn.role == .assistant ? turn.telemetry.formattedPills : []
                if !telemetryPills.isEmpty {
                    telemetryStrip(telemetryPills)
                }
            }
            .frame(maxWidth: 660, alignment: .leading)
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier(turn.role == .user ? "user-message" : "assistant-message")
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
    }

    private var showcasePresetsView: some View {
        let columns = [
            GridItem(.flexible(), spacing: 10),
            GridItem(.flexible(), spacing: 10)
        ]
        return LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
            ForEach(ShowcasePreset.allCases) { preset in
                Button {
                    model.applyShowcasePreset(preset)
                    if preset.videoFramePolicy == .temporal {
                        showAdvanced = true
                    }
                    promptFocused = true
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 6) {
                            Image(systemName: preset.iconName)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(theme.accent)
                                .accessibilityHidden(true)
                            Text(preset.title)
                                .font(.system(size: theme.bodySize, weight: .semibold))
                                .foregroundStyle(theme.text)
                                .lineLimit(1)
                        }
                        Text(preset.subtitle)
                            .font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.secondary)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
                    .overlay(RoundedRectangle(cornerRadius: theme.cardRadius).stroke(theme.line))
                }
                .buttonStyle(.plain)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                .accessibilityIdentifier("showcase-preset-\(preset.rawValue)")
            }
        }
        .frame(maxWidth: 740)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("showcase-presets")
    }

    private func telemetryStrip(_ pills: [TurnTelemetry.Pill]) -> some View {
        HStack(spacing: 6) {
            ForEach(pills) { pill in
                let isAccent = pill.kind == .i2v
                let pillView = Text(pill.label)
                    .font(.system(size: 11, weight: isAccent ? .semibold : .regular, design: .monospaced))
                    .foregroundStyle(isAccent ? theme.accent : theme.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(isAccent ? theme.accent.opacity(0.12) : theme.surface)
                    .clipShape(Capsule())
                    .overlay(Capsule().stroke(isAccent ? theme.accent.opacity(0.4) : theme.line))
                    .fixedSize(horizontal: true, vertical: false)
                    .accessibilityIdentifier("telemetry-pill-\(pill.kind.rawValue)")
                if let helpText = pill.help {
                    pillView.help(helpText)
                } else {
                    pillView
                }
            }
        }
        .padding(.top, 2)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("turn-telemetry")
    }

    @ViewBuilder
    private func textBubble(_ text: String, role: ChatTurn.Role) -> some View {
        Group {
            if role == .assistant {
                MarkdownTextView(text: text, theme: theme)
            } else {
                Text(text)
                    .font(.system(size: theme.bodySize))
                    .lineSpacing(3.5)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .background(role == .user ? theme.ink : theme.surface)
        .foregroundStyle(role == .user ? theme.onInk : theme.text)
        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
    }

    /// One inline narrative visual card: prompt caption, the active render, the
    /// in-place swap button, and a save action.
    private func visualSlotCard(_ slot: VisualSlotState, turnID: UUID) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: slot.activeKind == "video" ? "video" : "photo")
                    .foregroundStyle(theme.accent)
                    .accessibilityHidden(true)
                if let badge = slot.continuityBadgeText {
                    let isChain = slot.isChainedBadge
                    let badgeLabel = Text(badge)
                        .font(.system(size: 11, weight: slot.anchorExplicit ? .semibold : .medium))
                        .foregroundStyle(isChain ? theme.accent : theme.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(isChain ? theme.accent.opacity(slot.anchorExplicit ? 0.16 : 0.12) : theme.hover)
                        .clipShape(Capsule())
                        .overlay(Capsule().stroke(isChain ? theme.accent.opacity(slot.anchorExplicit ? 0.55 : 0.35) : theme.line))
                        .accessibilityIdentifier("visual-slot-continuity")
                    if let helpText = slot.continuityBadgeHelp {
                        badgeLabel.help(helpText)
                    } else {
                        badgeLabel
                    }
                }
                Text(slot.prompt.isEmpty ? "Visual \(slot.slotIndex + 1)" : slot.prompt)
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
            }
            if slot.activeKind == "video" {
                if let artifact = slot.videoArtifact {
                    VideoArtifactView(artifact: artifact, theme: theme) {
                        saveArtifact(artifact, defaultName: "video.mp4")
                    }
                } else {
                    visualPlaceholder(loading: slot.isLoading, label: slot.placeholderLabel)
                }
            } else if let artifact = slot.imageArtifact,
                      let image = NSImage(data: artifact.data) {
                Button {
                    preview = artifact
                } label: {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 220, height: 220)
                        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
                        .overlay(RoundedRectangle(cornerRadius: theme.cardRadius)
                            .stroke(theme.line))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Narrative image")
                .accessibilityIdentifier("visual-slot-image")
                .help("Open the full image")
            } else {
                visualPlaceholder(loading: slot.isLoading, label: slot.placeholderLabel)
            }
            HStack(spacing: 10) {
                Button(slot.activeKind == "video" ? "→ Image"
                       : (slot.imageArtifact != nil ? "Animate → Video" : "→ Video")) {
                    model.swapVisualSlot(
                        turnID: turnID, slotID: slot.slotID,
                        toKind: slot.activeKind == "image" ? "video" : "image")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .accessibilityIdentifier("visual-swap")
                if slot.activeKind == "image", let artifact = slot.imageArtifact {
                    Button("Save PNG") {
                        saveArtifact(artifact, defaultName: "image.png")
                    }
                    .buttonStyle(.link)
                    .font(.system(size: theme.captionSize))
                    .accessibilityIdentifier("visual-save")
                }
                Spacer(minLength: 0)
            }
            if let error = slot.error {
                HStack(alignment: .top, spacing: 5) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(theme.danger)
                        .accessibilityHidden(true)
                    Text(error)
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.danger)
                        .textSelection(.enabled)
                        .accessibilityIdentifier("visual-slot-error")
                }
            }
        }
        .padding(11)
        .background(theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
        .overlay(RoundedRectangle(cornerRadius: theme.cardRadius)
            .stroke(slot.error != nil ? theme.danger.opacity(0.45) : theme.line))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("visual-slot")
    }

    @ViewBuilder
    private func visualPlaceholder(loading: Bool, label: String) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: theme.cardRadius)
                .fill(theme.background)
                .frame(width: 220, height: 128)
                .overlay(RoundedRectangle(cornerRadius: theme.cardRadius)
                    .stroke(theme.line))
            if loading {
                ProgressView().controlSize(.small)
            } else {
                Text(label)
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.tertiary)
            }
        }
    }

    @ViewBuilder
    private func artifactView(_ artifact: GeneratedArtifact,
                              liveStreaming: Bool = false) -> some View {
        if artifact.mediaType == "image",
           let image = NSImage(data: artifact.data) {
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    preview = artifact
                } label: {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 148, height: 148)
                        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
                        .overlay(RoundedRectangle(cornerRadius: theme.cardRadius)
                            .stroke(theme.line))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Generated image")
                .accessibilityIdentifier("artifact-image")
                .help("Open the full image")
                Button("Save PNG") { saveArtifact(artifact, defaultName: "image.png") }
                    .buttonStyle(.link)
                    .font(.system(size: theme.captionSize))
            }
        } else if artifact.mediaType == "audio" {
            VStack(alignment: .leading, spacing: 4) {
                AudioPlayerView(
                    artifact: artifact,
                    theme: theme,
                    liveStreaming: liveStreaming,
                    onStopLiveStream: { model.stopStreamingAudio() }
                )
                Button("Save WAV") { saveArtifact(artifact, defaultName: "speech.wav") }
                    .buttonStyle(.link)
                    .font(.system(size: theme.captionSize))
            }
        } else if artifact.mediaType == "video" || artifact.mimeType == "video/mp4" {
            VideoArtifactView(artifact: artifact, theme: theme) {
                saveArtifact(artifact, defaultName: "video.mp4")
            }
        } else {
            Text("Unsupported artifact \(artifact.mimeType)")
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.secondary)
        }
    }

    private func fullImage(_ artifact: GeneratedArtifact) -> some View {
        VStack(spacing: 12) {
            if let image = NSImage(data: artifact.data) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 900, maxHeight: 680)
            }
            HStack {
                Text("\(artifact.mimeType) · \(artifact.data.count) bytes")
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                Spacer()
                Button("Close") { preview = nil }
                    .keyboardShortcut(.escape, modifiers: [])
            }
        }
        .padding(18)
        .frame(minWidth: 620, minHeight: 480)
        .background(theme.background)
    }

    private func errorBanner(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(theme.danger)
                .accessibilityHidden(true)
            Text(text)
                .font(.system(size: theme.bodySize))
                .foregroundStyle(theme.danger)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(theme.danger.opacity(0.08))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("error")
    }

    private var composer: some View {
        VStack(spacing: 8) {
            if model.isLiveVoiceActive {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform.and.mic")
                            .foregroundStyle(theme.accent)
                            .accessibilityHidden(true)
                        Text(model.liveVoicePartialTranscript.isEmpty
                             ? "Live Voice active — speak anytime to interrupt."
                             : model.liveVoicePartialTranscript)
                            .font(.system(size: theme.bodySize))
                            .foregroundStyle(theme.secondary)
                            .lineLimit(2)
                            .accessibilityIdentifier("live-voice-transcript")
                        Spacer()
                        Button("End") { model.stopLiveVoice() }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .tint(theme.danger)
                            .accessibilityIdentifier("live-voice-end")
                    }
                    HStack(spacing: 8) {
                        let clampedLevel = min(max(model.liveVoiceMicLevel, 0.0), 1.0)
                        HStack(spacing: 6) {
                            Text("Mic")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(theme.secondary)
                            ZStack(alignment: .leading) {
                                Capsule()
                                    .fill(theme.hover)
                                    .overlay(Capsule().stroke(theme.line))
                                Capsule()
                                    .fill(theme.accent)
                                    .frame(width: 64 * CGFloat(clampedLevel))
                            }
                            .frame(width: 64, height: 6)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(theme.background)
                        .clipShape(Capsule())
                        .overlay(Capsule().stroke(theme.line))
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("Live microphone input level")
                        .accessibilityValue("\(Int((clampedLevel * 100).rounded())) percent")
                        .accessibilityIdentifier("live-voice-mic-meter")
                        if let echoText = model.liveVoiceEchoSuppressionStatusText {
                            let echoOn = model.liveVoiceConfirmedEchoSuppression == true
                            Text(echoText)
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(echoOn ? theme.accent : theme.danger)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background((echoOn ? theme.accent : theme.danger).opacity(0.12))
                                .clipShape(Capsule())
                                .overlay(Capsule().stroke((echoOn ? theme.accent : theme.danger).opacity(0.4)))
                                .accessibilityIdentifier("live-voice-echo-status")
                        }
                        Text(model.liveVoiceBargeInStatusText)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(theme.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(theme.background)
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(theme.line))
                            .accessibilityIdentifier("live-voice-barge-in-count")
                        Spacer()
                    }
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("live-voice-status-strip")
                }
            }
            if model.isListening || !model.partialTranscript.isEmpty {
                HStack(spacing: 7) {
                    Image(systemName: "waveform")
                        .foregroundStyle(theme.accent)
                        .accessibilityHidden(true)
                    Text(model.partialTranscript.isEmpty
                         ? "Listening…" : model.partialTranscript)
                        .font(.system(size: theme.bodySize))
                        .foregroundStyle(theme.secondary)
                        .lineLimit(2)
                        .accessibilityIdentifier("partial-transcript")
                    Spacer()
                }
            }
            if let attached = model.attachedImageURL {
                HStack(spacing: 6) {
                    Image(systemName: "photo")
                        .foregroundStyle(theme.accent)
                    Text("VLM image: \(attached.lastPathComponent)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                    Button("×") { model.attachedImageURL = nil }
                        .buttonStyle(.plain)
                    Spacer()
                }
            }
            if let attachedVid = model.attachedVideoURL {
                HStack(spacing: 6) {
                    Image(systemName: "film")
                        .foregroundStyle(theme.accent)
                    Text("VLM video: \(attachedVid.lastPathComponent)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                    Button("×") { model.attachedVideoURL = nil }
                        .buttonStyle(.plain)
                    Spacer()
                }
            }
            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    if let url = pickFile(types: [.png, .jpeg, .webP]) {
                        model.attachedImageURL = url
                        model.attachedVideoURL = nil
                    }
                } label: {
                    Image(systemName: "photo.badge.plus")
                        .font(.system(size: 14))
                        .frame(width: 24, height: 22)
                }
                .buttonStyle(.bordered)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                .help("Attach an image for VLM visual QA")
                Button {
                    if let url = pickFile(types: [.mpeg4Movie, .quickTimeMovie, .movie]) {
                        model.attachedVideoURL = url
                        model.attachedImageURL = nil
                    }
                } label: {
                    Image(systemName: "video.badge.plus")
                        .font(.system(size: 14))
                        .frame(width: 24, height: 22)
                }
                .buttonStyle(.bordered)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                .help("Attach a video clip for native VLM video understanding")
                TextField("Write a prompt", text: $model.prompt, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: theme.bodySize))
                    .focused($promptFocused)
                    .onSubmit { model.send() }
                    .disabled(model.isLiveVoiceActive)
                    .accessibilityLabel("Prompt")
                    .accessibilityIdentifier("prompt")
                Button {
                    model.isListening ? model.stopListening() : model.startListening()
                } label: {
                    Image(systemName: model.isListening
                          ? "stop.circle.fill" : "mic.fill")
                        .font(.system(size: 15))
                        .frame(width: 26, height: 22)
                }
                .buttonStyle(.bordered)
                .tint(model.isListening ? theme.danger : theme.accent)
                .disabled(model.isRunning || model.isLiveVoiceActive)
                .accessibilityLabel(model.isListening ? "Stop recording" : "Record")
                .accessibilityIdentifier("mic")
                if model.isRunning {
                    Button("Stop") { model.stop() }
                        .buttonStyle(.borderedProminent)
                        .tint(theme.danger)
                        .keyboardShortcut(".", modifiers: .command)
                        .accessibilityIdentifier("stop")
                } else {
                    Button("Send") { model.send() }
                        .buttonStyle(.borderedProminent)
                        .tint(theme.ink)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(model.isLiveVoiceActive
                                  || (model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                      && model.attachedImageURL == nil
                                      && model.attachedVideoURL == nil))
                        .accessibilityIdentifier("send")
                }
            }
            HStack(spacing: 4) {
                if model.isLiveVoiceActive {
                    Text("Live Voice streams 16 kHz PCM16 over WebSocket to the configured ummInfer server.")
                } else {
                    Text("⌘Return to send ·")
                    Text("Voice uses Apple's on-device model; audio never leaves this Mac.")
                }
            }
            .font(.system(size: theme.captionSize))
            .foregroundStyle(theme.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(theme.surface)
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let id = model.turns.last?.id else { return }
        withAnimation(.easeOut(duration: 0.12)) { proxy.scrollTo(id, anchor: .bottom) }
    }
}

@MainActor
final class AudioPlayerModel: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var isPlaying = false
    @Published private(set) var duration: TimeInterval = 0
    private var player: AVAudioPlayer?

    func load(_ data: Data) {
        guard player == nil else { return }
        let player = try? AVAudioPlayer(data: data)
        player?.delegate = self
        player?.prepareToPlay()
        self.player = player
        duration = player?.duration ?? 0
    }

    func toggle() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
        } else {
            player.play()
            isPlaying = true
        }
    }

    func stop() {
        player?.stop()
        player?.currentTime = 0
        isPlaying = false
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer,
                                                 successfully flag: Bool) {
        Task { @MainActor in self.isPlaying = false }
    }
}

struct StreamingAudioView: View {
    let theme: Theme
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Button("Stop") { onStop() }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("audio-stream-stop")
            Text("Playing stream…")
                .font(.system(size: theme.captionSize, design: .monospaced))
                .foregroundStyle(theme.accent)
                .accessibilityIdentifier("audio-stream-status")
            Text("PCM16")
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.tertiary)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
    }
}

struct AudioPlayerView: View {
    let artifact: GeneratedArtifact
    let theme: Theme
    var liveStreaming = false
    var onStopLiveStream: () -> Void = {}
    @StateObject private var player = AudioPlayerModel()

    private var durationText: String {
        String(format: "%.1f s", player.duration)
    }

    var body: some View {
        HStack(spacing: 8) {
            Button(player.isPlaying ? "Pause" : "Play") {
                onStopLiveStream()
                player.toggle()
            }
            .buttonStyle(.bordered)
            .disabled(player.duration <= 0 || liveStreaming)
            .accessibilityIdentifier("audio-play")
            Button("Stop") {
                onStopLiveStream()
                player.stop()
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("audio-stop")
            Text(liveStreaming ? "Live · \(durationText)" : durationText)
                .font(.system(size: theme.captionSize, design: .monospaced))
                .foregroundStyle(liveStreaming ? theme.accent : theme.secondary)
                .accessibilityIdentifier("audio-duration")
            Text("WAV")
                .font(.system(size: theme.captionSize))
                .foregroundStyle(theme.tertiary)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
        .onAppear { player.load(artifact.data) }
    }
}

@MainActor
final class VideoPlayerModel: ObservableObject {
    @Published var player: AVPlayer?
    private var fileURL: URL?

    func load(_ artifact: GeneratedArtifact) {
        guard player == nil else { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ummmlx-\(artifact.id).mp4")
        do {
            try artifact.data.write(to: url, options: .atomic)
            fileURL = url
            player = AVPlayer(url: url)
        } catch {
            player = nil
        }
    }

    deinit {
        if let fileURL {
            try? FileManager.default.removeItem(at: fileURL)
        }
    }
}

struct AVPlayerViewRepresentable: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.player = player
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

struct VideoArtifactView: View {
    let artifact: GeneratedArtifact
    let theme: Theme
    let onSave: () -> Void
    @StateObject private var model = VideoPlayerModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let player = model.player {
                AVPlayerViewRepresentable(player: player)
                    .frame(width: 320, height: 214)
                    .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
                    .overlay(RoundedRectangle(cornerRadius: theme.cardRadius)
                        .stroke(theme.line))
            } else {
                ProgressView()
                    .frame(width: 320, height: 214)
            }
            HStack(spacing: 8) {
                Text(String(format: "MP4 · %.1f KB", Double(artifact.data.count) / 1024.0))
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                Button("Save MP4", action: onSave)
                    .buttonStyle(.link)
                    .font(.system(size: theme.captionSize))
            }
        }
        .onAppear { model.load(artifact) }
    }
}
