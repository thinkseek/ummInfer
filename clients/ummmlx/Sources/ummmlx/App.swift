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
    @State private var showOptions = false
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
            Circle().fill(model.isRunning ? theme.accent : theme.tertiary)
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
            Toggle(isOn: $model.audioOutput) {
                Text("Voice").font(.system(size: theme.captionSize))
            }
            .toggleStyle(.checkbox)
            .help("Stream spoken audio while text and visual legs render")
            .accessibilityIdentifier("audio-output")
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
            Button("New chat") { model.newChat() }
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
                HStack(spacing: 10) {
                    Toggle("Image", isOn: $model.imageOutput)
                        .toggleStyle(.checkbox)
                        .accessibilityIdentifier("image-output")
                    Toggle("Video", isOn: $model.videoOutput)
                        .toggleStyle(.checkbox)
                        .accessibilityIdentifier("video-output")
                }
                .padding(.bottom, 4)
                Button(showOptions ? "Hide Options" : "Options") {
                    showOptions.toggle()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.bottom, 2)
            }
            if showOptions {
                VStack(alignment: .leading, spacing: 8) {
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
                        Toggle("Narrative", isOn: $model.narrativeMode)
                            .toggleStyle(.checkbox)
                            .accessibilityIdentifier("narrative")
                        Picker("Visual", selection: $model.narrativeVisualMode) {
                            Text("Auto").tag("auto")
                            Text("Image").tag("image")
                            Text("Video").tag("video")
                        }
                        .frame(width: 132)
                        Stepper("Slots \(model.narrativeMaxSlots)",
                                value: $model.narrativeMaxSlots, in: 1...6)
                            .fixedSize()
                        Text("Interleaved visual beats need Image or Video enabled.")
                            .font(.system(size: theme.captionSize))
                            .foregroundStyle(theme.tertiary)
                    }
                }
                .padding(.top, 4)
            }
        }
        .disabled(model.isRunning)
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

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if model.turns.isEmpty {
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
                    .frame(maxWidth: .infinity)
                    .padding(.top, 110)
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
                    if turn.state == .stopped {
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
            }
            .frame(maxWidth: 660, alignment: .leading)
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier(turn.role == .user ? "user-message" : "assistant-message")
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
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
                    visualPlaceholder(loading: slot.isLoading, label: "No video")
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
                visualPlaceholder(loading: slot.isLoading, label: "No image")
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
                Text(error)
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.danger)
                    .textSelection(.enabled)
                    .accessibilityIdentifier("visual-slot-error")
            }
        }
        .padding(11)
        .background(theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
        .overlay(RoundedRectangle(cornerRadius: theme.cardRadius).stroke(theme.line))
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
            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    model.attachedImageURL = pickFile(types: [.png, .jpeg, .webP])
                } label: {
                    Image(systemName: "photo.badge.plus")
                        .font(.system(size: 14))
                        .frame(width: 24, height: 22)
                }
                .buttonStyle(.bordered)
                .disabled(model.isRunning)
                .help("Attach an image for VLM visual QA")
                TextField("Write a prompt", text: $model.prompt, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: theme.bodySize))
                    .onSubmit { model.send() }
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
                .disabled(model.isRunning)
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
                        .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                  && model.attachedImageURL == nil)
                        .accessibilityIdentifier("send")
                }
            }
            HStack(spacing: 4) {
                Text("⌘Return to send ·")
                Text("Voice uses Apple's on-device model; audio never leaves this Mac.")
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
