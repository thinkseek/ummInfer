// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import AVFoundation

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
                .font(.system(size: theme.titleSize, weight: .semibold))
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
        HStack(alignment: .bottom, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Base URL")
                    .font(.system(size: theme.captionSize))
                    .foregroundStyle(theme.secondary)
                TextField("http://127.0.0.1:30000/v1", text: $model.endpoint)
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
            .frame(width: 260)
            VStack(alignment: .leading, spacing: 3) {
                Toggle("Image", isOn: $model.imageOutput)
                    .toggleStyle(.checkbox)
                    .accessibilityIdentifier("image-output")
                Toggle("Audio", isOn: $model.audioOutput)
                    .toggleStyle(.checkbox)
                    .accessibilityIdentifier("audio-output")
            }
            .frame(width: 92, alignment: .leading)
            .padding(.bottom, 2)
        }
        .disabled(model.isRunning)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(theme.surface)
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
                        Text("Text · image · audio · on-device voice input")
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
                    if turn.state == .streaming && turn.text.isEmpty && turn.artifacts.isEmpty {
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
                if !turn.text.isEmpty {
                    Text(turn.text)
                        .font(.system(size: theme.bodySize))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 10)
                        .background(turn.role == .user ? theme.ink : theme.surface)
                        .foregroundStyle(turn.role == .user ? theme.onInk : theme.text)
                        .clipShape(RoundedRectangle(cornerRadius: theme.cardRadius))
                }
                if !turn.transcript.isEmpty {
                    Text("Server transcript: \(turn.transcript)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.secondary)
                        .accessibilityIdentifier("server-transcript")
                }
                if !turn.artifacts.isEmpty {
                    HStack(alignment: .top, spacing: 10) {
                        ForEach(turn.artifacts) { artifact in
                            artifactView(artifact).id(artifact.id)
                        }
                    }
                }
                ForEach(turn.legErrors, id: \.self) { detail in
                    Text("Output leg: \(detail)")
                        .font(.system(size: theme.captionSize))
                        .foregroundStyle(theme.danger)
                }
            }
            .frame(maxWidth: 620, alignment: .leading)
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier(turn.role == .user ? "user-message" : "assistant-message")
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
    }

    @ViewBuilder
    private func artifactView(_ artifact: GeneratedArtifact) -> some View {
        if artifact.mediaType == "image",
           let image = NSImage(data: artifact.data) {
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
        } else if artifact.mediaType == "audio" {
            AudioPlayerView(artifact: artifact, theme: theme)
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
            HStack(alignment: .bottom, spacing: 10) {
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
                        .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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

struct AudioPlayerView: View {
    let artifact: GeneratedArtifact
    let theme: Theme
    @StateObject private var player = AudioPlayerModel()

    private var durationText: String {
        String(format: "%.1f s", player.duration)
    }

    var body: some View {
        HStack(spacing: 8) {
            Button(player.isPlaying ? "Pause" : "Play") { player.toggle() }
                .buttonStyle(.bordered)
                .disabled(player.duration <= 0)
                .accessibilityIdentifier("audio-play")
            Button("Stop") { player.stop() }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("audio-stop")
            Text(durationText)
                .font(.system(size: theme.captionSize, design: .monospaced))
                .foregroundStyle(theme.secondary)
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
