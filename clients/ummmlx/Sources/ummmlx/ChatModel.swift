// SPDX-License-Identifier: Apache-2.0

import Foundation

struct ChatTurn: Identifiable {
    enum Role: String {
        case user
        case assistant
    }
    enum State {
        case complete
        case streaming
        case stopped
        case failed
    }

    let id = UUID()
    let role: Role
    var text: String
    var state: State
    /// Generated image/audio artifacts of this turn. Never sent back as history.
    var artifacts: [GeneratedArtifact] = []
    /// Server ASR transcript when the request carried audio. Empty otherwise.
    var transcript = ""
    /// Non-terminal optional-leg failures reported by the server.
    var legErrors: [String] = []
}

@MainActor
final class ChatModel: ObservableObject {
    @Published var endpoint: String
    @Published var model: String
    @Published var prompt = ""
    @Published private(set) var turns: [ChatTurn] = []
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Ready"
    @Published private(set) var error: String?
    /// Server-echoed session id. The UI shows its first eight characters.
    @Published private(set) var sessionID = ""
    @Published var imageOutput = false
    @Published var audioOutput = false
    @Published var videoOutput = false
    @Published var maxTokensText = "256"
    @Published var temperatureText = "0"
    @Published var topPText = ""
    @Published var seedText = ""
    @Published var imageSize = "512x512"
    @Published var imageStepsText = ""
    @Published var editImageURL: URL?
    @Published var audioVoice = "vivian"
    @Published var refAudioURL: URL?
    @Published var videoSize = "480x320"
    @Published var videoFramesText = "13"
    @Published var videoStepsText = "15"
    @Published var videoGuidanceText = "5.0"
    @Published var attachedImageURL: URL?
    @Published var partialTranscript = ""
    @Published private(set) var isListening = false
    @Published private(set) var speechStatus = ""
    @Published private(set) var streamingAudioTurnID: UUID?
    @Published private(set) var isStreamingAudioPlaying = false

    let speech: SpeechController
    let audioStream: AudioStreamPlaying

    private let client: ChatClient
    private var requestTask: Task<Void, Never>?
    private var activeTurn: UUID?
    private var stopping = false
    private var receivedDelta = false
    private var requestSessionID = ""
    var onFirstDelta: (() -> Void)?
    /// Automation hook: the exact messages sent for one request.
    var onRequest: (([ChatMessage]) -> Void)?

    init(speech: SpeechController = SpeechController(),
         client: ChatClient? = nil,
         audioStream: AudioStreamPlaying? = nil) {
        self.speech = speech
        self.client = client ?? ChatClient()
        // Automation runs are silent by default so a gate run never makes noise.
        // --automation-audio opts into the live player for a manual listen.
        let automation = ProcessInfo.processInfo.arguments.contains("--automation")
        let audible = ProcessInfo.processInfo.arguments.contains("--automation-audio")
        let resolvedAudio = audioStream ?? (
            automation && !audible ? NoOpAudioStreamPlayer() : LiveAudioStreamPlayer()
        )
        self.audioStream = resolvedAudio
        let defaults = UserDefaults.standard
        endpoint = defaults.string(forKey: "endpoint") ?? "http://127.0.0.1:30000/v1"
        model = defaults.string(forKey: "model") ?? "Qwen2.5-0.5B-Instruct-4bit"
        requestSessionID = Self.newSessionID()
        resolvedAudio.onStateChange = { [weak self] playing in
            guard let self else { return }
            self.isStreamingAudioPlaying = playing
            if !playing {
                self.streamingAudioTurnID = nil
            }
        }
    }

    static func newSessionID() -> String {
        "ummmlx-" + UUID().uuidString
            .replacingOccurrences(of: "-", with: "").prefix(16)
    }

    /// First eight characters of the server id. The client-generated prefix is
    /// dropped so the chip does not read "ummmlx-" for every session.
    var sessionShort: String {
        var text = sessionID
        if text.hasPrefix("ummmlx-") { text.removeFirst("ummmlx-".count) }
        return String(text.prefix(8))
    }

    private static func imageMimeType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "webp": return "image/webp"
        default: return "image/png"
        }
    }

    private static func dataURL(from url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return "data:\(imageMimeType(for: url));base64,\(data.base64EncodedString())"
    }

    func send() {
        guard !isRunning else { return }
        error = nil
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || attachedImageURL != nil else {
            error = "Enter a prompt."
            status = "Input required"
            return
        }
        do {
            _ = try ChatClient.chatURL(base: endpoint)
            if model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ChatClientError.modelRequired
            }
        } catch {
            self.error = error.localizedDescription
            status = "Configuration error"
            return
        }

        let sentImageURL = attachedImageURL
        var attachedImageDataURL: String?
        var attachedImageBytes: Data?
        if let sentImageURL {
            do {
                let data = try Data(contentsOf: sentImageURL)
                attachedImageBytes = data
                attachedImageDataURL = "data:\(Self.imageMimeType(for: sentImageURL));base64,\(data.base64EncodedString())"
            } catch {
                self.error = "Could not read attached image: \(error.localizedDescription)"
                status = "Input error"
                return
            }
        }

        if !ProcessInfo.processInfo.arguments.contains("--automation") {
            UserDefaults.standard.set(endpoint, forKey: "endpoint")
            UserDefaults.standard.set(model, forKey: "model")
        }
        var history = Self.requestMessages(turns: turns, prompt: text)
        if let attachedImageDataURL, !history.isEmpty {
            let last = history.removeLast()
            history.append(ChatMessage(role: last.role, content: last.content,
                                       imageDataURL: attachedImageDataURL))
        }
        onRequest?(history)

        var options = ChatStreamOptions()
        if requestSessionID.isEmpty { requestSessionID = Self.newSessionID() }
        options.sessionID = requestSessionID
        if let maxTok = Int(maxTokensText.trimmingCharacters(in: .whitespaces)), maxTok > 0 {
            options.maxTokens = maxTok
        }
        if let temp = Double(temperatureText.trimmingCharacters(in: .whitespaces)), temp >= 0 {
            options.temperature = temp
        }
        if let topP = Double(topPText.trimmingCharacters(in: .whitespaces)),
           topP > 0, topP <= 1 {
            options.topP = topP
        }
        if let seed = Int64(seedText.trimmingCharacters(in: .whitespaces)) {
            options.seed = seed
        }
        if imageOutput {
            var img = ImageOptions(size: imageSize)
            if let steps = Int(imageStepsText.trimmingCharacters(in: .whitespaces)), steps > 0 {
                img.steps = steps
            }
            if let editImageURL {
                do {
                    img.imageDataURL = try Self.dataURL(from: editImageURL)
                } catch {
                    self.error = "Could not read edit image: \(error.localizedDescription)"
                    status = "Input error"
                    return
                }
            }
            options.image = img
        }
        if audioOutput {
            var aud = AudioOptions(voice: audioVoice.trimmingCharacters(in: .whitespaces).isEmpty
                                   ? "vivian" : audioVoice.trimmingCharacters(in: .whitespaces))
            if let refAudioURL {
                do {
                    aud.refAudio = try Data(contentsOf: refAudioURL).base64EncodedString()
                } catch {
                    self.error = "Could not read reference WAV: \(error.localizedDescription)"
                    status = "Input error"
                    return
                }
            }
            options.audio = aud
        }
        if videoOutput {
            var vid = VideoOptions(size: videoSize)
            if let frames = Int(videoFramesText.trimmingCharacters(in: .whitespaces)), frames > 0 {
                vid.numFrames = frames
            }
            if let steps = Int(videoStepsText.trimmingCharacters(in: .whitespaces)), steps > 0 {
                vid.steps = steps
            }
            if let cfg = Double(videoGuidanceText.trimmingCharacters(in: .whitespaces)), cfg >= 0 {
                vid.guidanceScale = cfg
            }
            options.video = vid
        }

        stopStreamingAudio()
        if audioOutput && !imageOutput && !videoOutput {
            audioStream.beginStream()
        }

        var userTurn = ChatTurn(role: .user, text: text.isEmpty ? "[Image]" : text, state: .complete)
        if let attachedImageBytes, let sentImageURL {
            userTurn.artifacts.append(GeneratedArtifact(
                id: "input-\(UUID().uuidString)",
                mediaType: "image",
                mimeType: Self.imageMimeType(for: sentImageURL),
                data: attachedImageBytes,
                sha256: ""
            ))
        }
        attachedImageURL = nil
        turns.append(userTurn)
        let assistant = ChatTurn(role: .assistant, text: "", state: .streaming)
        turns.append(assistant)
        activeTurn = assistant.id
        prompt = ""
        isRunning = true
        stopping = false
        receivedDelta = false
        status = "Connecting…"

        requestTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.client.stream(
                    base: self.endpoint, model: self.model, messages: history,
                    options: options
                ) { delta in
                    guard self.activeTurn == assistant.id,
                          let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                        return
                    }
                    self.turns[index].text += delta
                    if !self.receivedDelta {
                        self.receivedDelta = true
                        self.status = "Streaming…"
                        self.onFirstDelta?()
                    }
                } onEvent: { event in
                    guard self.activeTurn == assistant.id,
                          let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                        return
                    }
                    switch event.kind {
                    case .asrFinal:
                        if !event.text.isEmpty { self.turns[index].transcript = event.text }
                    case .legError:
                        let detail = event.error.isEmpty
                            ? "\(event.stage) output failed" : event.error
                        self.turns[index].legErrors.append(detail)
                        self.status = "Completed with leg error"
                    default:
                        break
                    }
                } onArtifact: { artifact in
                    guard self.activeTurn == assistant.id,
                          let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                        return
                    }
                    self.turns[index].artifacts.append(artifact)
                } onAudioChunk: { chunk in
                    guard self.activeTurn == assistant.id else { return }
                    self.streamingAudioTurnID = assistant.id
                    self.audioStream.enqueue(chunk)
                }
                guard self.activeTurn == assistant.id,
                      let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                    return
                }
                self.audioStream.finishStream()
                self.turns[index].text = result.text
                self.turns[index].state = .complete
                if !result.sessionID.isEmpty { self.sessionID = result.sessionID }
                self.status = result.finishReason.isEmpty
                    ? "Completed" : "Completed · \(result.finishReason)"
                self.finish(assistant.id)
            } catch {
                guard self.activeTurn == assistant.id else { return }
                self.stopStreamingAudio()
                let cancelled = self.stopping || Task.isCancelled ||
                    (error as? URLError)?.code == .cancelled || error is CancellationError
                if let index = self.turns.firstIndex(where: { $0.id == assistant.id }) {
                    if self.turns[index].text.isEmpty && self.turns[index].artifacts.isEmpty {
                        self.turns.remove(at: index)
                    } else {
                        self.turns[index].state = cancelled ? .stopped : .failed
                    }
                }
                if cancelled {
                    self.status = "Stopped"
                } else {
                    self.status = "Error"
                    self.error = error.localizedDescription
                }
                self.finish(assistant.id)
            }
        }
    }

    static func requestMessages(turns: [ChatTurn], prompt: String) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        var pendingUser: ChatTurn?
        for turn in turns {
            switch turn.role {
            case .user:
                pendingUser = turn
            case .assistant:
                if let pendingUser, turn.state == .complete {
                    messages.append(ChatMessage(role: "user", content: pendingUser.text))
                    messages.append(ChatMessage(role: "assistant", content: turn.text))
                }
                pendingUser = nil
            }
        }
        messages.append(ChatMessage(role: "user", content: prompt))
        return messages
    }

    // MARK: - Local speech input (Apple on-device model only)

    func startListening() {
        guard !isRunning, !isListening else { return }
        stopStreamingAudio()
        error = nil
        speech.onPartial = { [weak self] text in
            self?.partialTranscript = text
        }
        speech.onFinal = { [weak self] text in
            guard let self else { return }
            self.prompt = text
            self.partialTranscript = ""
            self.isListening = false
            self.speechStatus = "On-device transcript ready"
            if self.status == "Speech error" { self.status = "Ready" }
        }
        speech.onFailure = { [weak self] message in
            guard let self else { return }
            self.isListening = false
            self.partialTranscript = ""
            self.error = message
            self.status = "Speech error"
        }
        // The recognizer can end without a final result (silence). Release
        // the composer instead of leaving the mic latched on.
        speech.onIdle = { [weak self] in
            guard let self, self.isListening else { return }
            self.isListening = false
            self.partialTranscript = ""
            self.speechStatus = "No speech recognized"
        }
        partialTranscript = ""
        speechStatus = "Listening on-device…"
        isListening = true
        speech.startMicrophone()
    }

    func stopListening() {
        speech.cancel()
        isListening = false
        partialTranscript = ""
        speechStatus = "Speech stopped"
    }

    /// File recognition used by the repeatable automation gate. Never contacts
    /// the network or the umm server.
    @discardableResult
    func transcribeFile(_ url: URL) async throws -> String {
        speechStatus = "Recognizing on-device…"
        error = nil
        do {
            let text = try await speech.transcribeFile(url)
            prompt = text
            partialTranscript = ""
            speechStatus = "On-device transcript ready"
            return text
        } catch {
            speechStatus = "Speech error"
            self.error = error.localizedDescription
            throw error
        }
    }

    func stopStreamingAudio() {
        audioStream.stop()
        isStreamingAudioPlaying = false
        streamingAudioTurnID = nil
    }

    func stop() {
        stopStreamingAudio()
        guard isRunning else { return }
        stopping = true
        status = "Stopping…"
        requestTask?.cancel()
    }

    func newChat() {
        stopStreamingAudio()
        requestTask?.cancel()
        requestTask = nil
        activeTurn = nil
        turns.removeAll()
        isRunning = false
        stopping = false
        receivedDelta = false
        error = nil
        status = "Ready"
        sessionID = ""
        requestSessionID = Self.newSessionID()
        onFirstDelta = nil
    }

    private func finish(_ id: UUID) {
        guard activeTurn == id else { return }
        requestTask = nil
        activeTurn = nil
        isRunning = false
        stopping = false
    }
}
