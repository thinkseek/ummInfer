// SPDX-License-Identifier: Apache-2.0

import Foundation

/// One interleaved visual beat of an assistant turn. It holds both rendered
/// kinds so a swap can reuse the cached artifact without a network call.
struct VisualSlotState: Identifiable, Equatable {
    let id = UUID()
    let slotID: String
    let slotIndex: Int
    var prompt: String
    /// Kind currently shown: "image" or "video".
    var activeKind: String
    var isLoading: Bool
    var error: String?
    var imageArtifact: GeneratedArtifact?
    var videoArtifact: GeneratedArtifact?
    /// Identifies the in-flight swap request, so a stale result is dropped
    /// after the user swapped again or a cached kind was shown instead.
    var swapToken: UUID?
    /// Optional cross-slot continuity metadata reported on `visual_slot` events.
    var continuityMode: String? = nil
    /// Optional 0-based anchor slot index when chained via I2V.
    var anchorSlotIndex: Int? = nil
    /// True when the slot used an explicit `source=slot-N` anchor directive.
    var anchorExplicit: Bool = false
    /// Optional verified 64-char hex SHA-256 digest of the resolved anchor image.
    var anchorSha256: String? = nil

    /// Short 8-character prefix of `anchorSha256` when a valid 64-char hex digest is present.
    var anchorShortDigest: String? {
        guard let normalized = UnifiedDecode.normalizedAnchorSHA256(anchorSha256) else {
            return nil
        }
        return String(normalized.prefix(8))
    }

    var isChainedBadge: Bool {
        guard let anchorSlotIndex, anchorSlotIndex >= 0 else { return false }
        return continuityMode == NarrativeContinuityMode.i2vChain.rawValue
            || continuityMode == nil
            || continuityMode?.isEmpty == true
            || anchorExplicit
    }

    var continuityBadgeText: String? {
        if isChainedBadge, let anchorSlotIndex {
            let slotLabel = "Slot \(anchorSlotIndex + 1)"
            if anchorExplicit {
                if let shortDigest = anchorShortDigest {
                    return "Explicit I2V ← \(slotLabel) · \(shortDigest)"
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

    /// Hover tooltip (`help`) describing the anchor binding and full SHA-256 digest.
    var continuityBadgeHelp: String? {
        guard isChainedBadge, let anchorSlotIndex else { return nil }
        let prefix = anchorExplicit
            ? "Explicit anchor slot-\(anchorSlotIndex)"
            : "Anchor slot-\(anchorSlotIndex)"
        if let normalized = UnifiedDecode.normalizedAnchorSHA256(anchorSha256) {
            return "\(prefix) · SHA-256: \(normalized)"
        }
        return anchorExplicit ? prefix : nil
    }
}

/// A run of assistant prose or an inline visual card. Prose after a visual
/// slot starts a new text block, so the card stays at its narrative beat.
enum TurnBlock: Identifiable {
    case text(id: UUID, content: String)
    case visual(VisualSlotState)

    var id: UUID {
        switch self {
        case .text(let id, _): return id
        case .visual(let slot): return slot.id
        }
    }

    var visualSlot: VisualSlotState? {
        switch self {
        case .text: return nil
        case .visual(let slot): return slot
        }
    }
}

/// Curated North-Star starter presets shown in the empty conversation state.
/// Mirrors `SHOWCASE_PRESETS` in `web/index.html`.
enum ShowcasePreset: String, CaseIterable, Identifiable, Equatable {
    case documentary = "documentary"
    case storyboardI2V = "storyboard_i2v"
    case smartTextOnly = "smart_text_only"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .documentary: return "Live Documentary"
        case .storyboardI2V: return "Storyboard I2V Chain"
        case .smartTextOnly: return "Smart Text-Only"
        }
    }

    var subtitle: String {
        switch self {
        case .documentary: return "Interleaved prose, diagram & voice"
        case .storyboardI2V: return "Slot 0 keyframe -> Slot 1 anchored I2V"
        case .smartTextOnly: return "Autonomous zero-visual abstention"
        }
    }

    var iconName: String {
        switch self {
        case .documentary: return "waveform.and.magnifyingglass"
        case .storyboardI2V: return "film.stack"
        case .smartTextOnly: return "function"
        }
    }

    var narrativeVisualMode: String { "auto" }

    var allowTextOnly: Bool {
        switch self {
        case .documentary: return true
        case .storyboardI2V: return false
        case .smartTextOnly: return true
        }
    }

    var continuityMode: NarrativeContinuityMode {
        switch self {
        case .documentary: return .off
        case .storyboardI2V: return .i2vChain
        case .smartTextOnly: return .off
        }
    }

    var styleAnchor: String {
        switch self {
        case .documentary: return ""
        case .storyboardI2V:
            return "cinematic 35mm film grain, warm lantern twilight, smooth camera motion"
        case .smartTextOnly: return ""
        }
    }

    var fastVideo: Bool {
        switch self {
        case .documentary: return true
        case .storyboardI2V: return true
        case .smartTextOnly: return false
        }
    }

    var enableAudio: Bool {
        switch self {
        case .documentary: return true
        case .storyboardI2V: return true
        case .smartTextOnly: return false
        }
    }

    var prompt: String {
        switch self {
        case .documentary:
            return "Explain how a black hole accretion disk produces relativistic jets and gravitational lensing. Speak clearly like a live documentary narrator, include one crisp diagram of the spacetime curvature, and one short motion clip of plasma spiraling into the event horizon."
        case .storyboardI2V:
            return "Create a two-beat cinematic storyboard of a glowing paper lantern drifting over a misty mountain river at dusk: first illustrate the establishing keyframe, then animate the river breeze and water reflections in motion."
        case .smartTextOnly:
            return "Prove concisely why the square root of 2 is irrational and explain the key step in two sentences."
        }
    }
}

/// Real client-observed timing and visual telemetry for one assistant turn.
/// Values remain nil unless the corresponding SSE event was actually observed.
struct TurnTelemetry: Equatable {
    struct Pill: Equatable, Identifiable {
        enum Kind: String, Equatable {
            case ttft
            case ttfa
            case visual
            case i2v
            case total
        }
        let kind: Kind
        let label: String
        var id: String { kind.rawValue }
    }

    var ttftMs: Double? = nil
    var ttfaMs: Double? = nil
    var visualSettledCount: Int = 0
    var firstVisualMs: Double? = nil
    var hasI2VChain: Bool = false
    var hasExplicitAnchor: Bool = false
    var totalMs: Double? = nil

    static func formatDuration(_ ms: Double) -> String? {
        guard ms.isFinite, ms >= 0 else { return nil }
        let rounded = Int(ms.rounded())
        if rounded < 1000 {
            return "\(rounded) ms"
        }
        return String(format: "%.2f s", ms / 1000.0)
    }

    var formattedPills: [Pill] {
        var pills: [Pill] = []
        if let ttftMs, let formatted = Self.formatDuration(ttftMs) {
            pills.append(Pill(kind: .ttft, label: "TTFT \(formatted)"))
        }
        if let ttfaMs, let formatted = Self.formatDuration(ttfaMs) {
            pills.append(Pill(kind: .ttfa, label: "TTFA \(formatted)"))
        }
        if visualSettledCount > 0 {
            let countLabel = visualSettledCount == 1
                ? "1 visual" : "\(visualSettledCount) visuals"
            if let firstVisualMs, let formatted = Self.formatDuration(firstVisualMs) {
                pills.append(Pill(kind: .visual, label: "\(countLabel) (\(formatted))"))
            } else {
                pills.append(Pill(kind: .visual, label: countLabel))
            }
        }
        if hasI2VChain || hasExplicitAnchor {
            pills.append(Pill(kind: .i2v, label: hasExplicitAnchor ? "Explicit I2V" : "I2V"))
        }
        if let totalMs, let formatted = Self.formatDuration(totalMs) {
            pills.append(Pill(kind: .total, label: "Total \(formatted)"))
        }
        return pills
    }
}

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
    /// Interleaved render order. `text` and `artifacts` stay populated for
    /// history replay and code that predates blocks.
    var blocks: [TurnBlock] = []
    /// Real client-observed streaming telemetry for assistant turns.
    var telemetry = TurnTelemetry()
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
    /// Global voice: streams TTS concurrently with text and visual legs.
    @Published var audioOutput = false
    @Published var videoOutput = false
    @Published var narrativeMode = true
    /// "auto", "image" or "video".
    @Published var narrativeVisualMode = "auto"
    @Published var narrativeMaxSlots = 3
    @Published var narrativeAllowTextOnly = true
    @Published var narrativeContinuityMode: NarrativeContinuityMode = .off
    @Published var narrativeStyleAnchor = ""
    @Published var maxTokensText = "256"
    @Published var temperatureText = "0"
    @Published var topPText = ""
    @Published var seedText = ""
    @Published var imageSize = "512x512"
    @Published var imageStepsText = ""
    @Published var editImageURL: URL?
    @Published var audioVoice = "vivian"
    @Published var refAudioURL: URL?
    @Published var videoSize = "704x480"
    @Published var videoFramesText = "29"
    @Published var videoStepsText = "25"
    @Published var videoGuidanceText = "5.0"
    /// Attached-video understanding controls, independent of Fast Video output.
    @Published var videoFramePolicy = "uniform"
    @Published var videoMotionThresholdText = "0.02"
    @Published var videoSceneThresholdText = "0.25"
    @Published var fastVideoMode = false
    @Published var attachedImageURL: URL?
    @Published var attachedVideoURL: URL?
    @Published var partialTranscript = ""
    @Published private(set) var isListening = false
    @Published private(set) var speechStatus = ""
    @Published private(set) var streamingAudioTurnID: UUID?
    @Published private(set) var isStreamingAudioPlaying = false
    @Published private(set) var isLiveVoiceActive = false
    @Published private(set) var liveVoiceState: RealtimeSessionState = .idle
    @Published private(set) var liveVoicePartialTranscript = ""
    @Published private(set) var liveVoiceLastAckCursor = 0
    @Published var liveVoiceSilenceMs = 400

    let speech: SpeechController
    let audioStream: AudioStreamPlaying
    let realtime: RealtimeClient

    private let client: ChatClient
    private var requestTask: Task<Void, Never>?
    private var activeTurn: UUID?
    private var activeSettledVisualKeys: Set<String> = []
    private var stopping = false
    private var receivedDelta = false
    private var requestSessionID = ""
    private var liveVoiceAssistantTurns: [String: UUID] = [:]
    var onFirstDelta: (() -> Void)?
    /// Automation hook: the exact messages sent for one request.
    var onRequest: (([ChatMessage]) -> Void)?

    init(speech: SpeechController = SpeechController(),
         client: ChatClient? = nil,
         audioStream: AudioStreamPlaying? = nil,
         realtime: RealtimeClient? = nil) {
        self.speech = speech
        self.client = client ?? ChatClient()
        self.realtime = realtime ?? RealtimeClient()
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

    private static func videoMimeType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "webm": return "video/webm"
        case "mov": return "video/quicktime"
        default: return "video/mp4"
        }
    }

    private static let maxAttachedVideoBytes = 32 * 1024 * 1024

    private static func elapsedMilliseconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        let components = duration.components
        let ms = Double(components.seconds) * 1000.0 + Double(components.attoseconds) / 1e15
        return max(0, ms)
    }

    /// Applies a curated North-Star showcase preset to controls and composer
    /// prompt without firing a network request.
    func applyShowcasePreset(_ preset: ShowcasePreset) {
        guard !isRunning, !isLiveVoiceActive else { return }
        narrativeMode = true
        narrativeVisualMode = preset.narrativeVisualMode
        imageOutput = true
        videoOutput = true
        narrativeAllowTextOnly = preset.allowTextOnly
        narrativeContinuityMode = preset.continuityMode
        narrativeStyleAnchor = preset.styleAnchor
        fastVideoMode = preset.fastVideo
        audioOutput = preset.enableAudio
        prompt = preset.prompt
        error = nil
    }

    private static func dataURL(from url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return "data:\(imageMimeType(for: url));base64,\(data.base64EncodedString())"
    }

    func send() {
        guard !isRunning, !isLiveVoiceActive else { return }
        error = nil
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || attachedImageURL != nil || attachedVideoURL != nil else {
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
        let sentVideoURL = attachedImageURL == nil ? attachedVideoURL : nil
        let videoFrameSelection: VideoFrameSelectionOptions?
        do {
            videoFrameSelection = sentVideoURL == nil ? nil : try VideoFrameSelectionOptions.parse(
                policy: videoFramePolicy,
                motionText: videoMotionThresholdText,
                sceneText: videoSceneThresholdText)
        } catch {
            self.error = error.localizedDescription
            status = "Configuration error"
            return
        }
        let validatedStyleAnchor: String?
        do {
            validatedStyleAnchor = narrativeMode
                ? try NarrativeOptions.validateStyleAnchor(narrativeStyleAnchor)
                : nil
        } catch {
            self.error = error.localizedDescription
            status = "Configuration error"
            return
        }
        var attachedImageDataURL: String?
        var attachedImageBytes: Data?
        var attachedVideoDataURL: String?
        var attachedVideoBytes: Data?
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
        } else if let sentVideoURL {
            do {
                let data = try Data(contentsOf: sentVideoURL)
                guard data.count <= Self.maxAttachedVideoBytes else {
                    self.error = "Attached video exceeds the 32 MiB upload limit."
                    status = "Input error"
                    return
                }
                attachedVideoBytes = data
                attachedVideoDataURL = "data:\(Self.videoMimeType(for: sentVideoURL));base64,\(data.base64EncodedString())"
            } catch {
                self.error = "Could not read attached video: \(error.localizedDescription)"
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
        } else if let attachedVideoDataURL, !history.isEmpty {
            let last = history.removeLast()
            history.append(ChatMessage(role: last.role, content: last.content,
                                       videoDataURL: attachedVideoDataURL))
        }
        onRequest?(history)

        var options = ChatStreamOptions()
        if requestSessionID.isEmpty { requestSessionID = Self.newSessionID() }
        options.sessionID = requestSessionID
        options.videoFrameSelection = videoFrameSelection
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
            if fastVideoMode {
                options.videoTeaCacheThresh = 0.05
            }
        }
        // Narrative needs a visual modality on the wire, or the server rejects
        // the request. Without one the toggle is ignored instead of failing.
        if narrativeMode && (imageOutput || videoOutput) {
            var mode = narrativeVisualMode
            if mode == "image" && !imageOutput { mode = "auto" }
            if mode == "video" && !videoOutput { mode = "auto" }
            options.narrative = NarrativeOptions(
                enabled: true, visualMode: mode,
                maxSlots: min(max(narrativeMaxSlots, 1), 6),
                allowTextOnly: narrativeAllowTextOnly,
                continuityMode: narrativeContinuityMode,
                styleAnchor: validatedStyleAnchor)
        }

        stopStreamingAudio()
        if audioOutput {
            audioStream.beginStream()
        }

        let fallbackLabel = sentVideoURL != nil ? "[Video]" : "[Image]"
        var userTurn = ChatTurn(role: .user, text: text.isEmpty ? fallbackLabel : text, state: .complete)
        if let attachedImageBytes, let sentImageURL {
            userTurn.artifacts.append(GeneratedArtifact(
                id: "input-\(UUID().uuidString)",
                mediaType: "image",
                mimeType: Self.imageMimeType(for: sentImageURL),
                data: attachedImageBytes,
                sha256: ""
            ))
        } else if let attachedVideoBytes, let sentVideoURL {
            userTurn.artifacts.append(GeneratedArtifact(
                id: "input-\(UUID().uuidString)",
                mediaType: "video",
                mimeType: Self.videoMimeType(for: sentVideoURL),
                data: attachedVideoBytes,
                sha256: ""
            ))
        }
        attachedImageURL = nil
        attachedVideoURL = nil
        turns.append(userTurn)
        let assistant = ChatTurn(role: .assistant, text: "", state: .streaming)
        turns.append(assistant)
        activeTurn = assistant.id
        activeSettledVisualKeys = []
        prompt = ""
        isRunning = true
        stopping = false
        receivedDelta = false
        status = "Connecting…"
        let requestStart = ContinuousClock.now

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
                    if !delta.isEmpty && self.turns[index].telemetry.ttftMs == nil {
                        self.turns[index].telemetry.ttftMs = Self.elapsedMilliseconds(since: requestStart)
                    }
                    self.appendTextDelta(delta, to: index)
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
                    case .visualSlot:
                        let slotIndex = event.slotIndex ?? 0
                        let slotID = event.slotID ?? "slot-\(slotIndex)"
                        let isVideo = event.mediaType == "video"
                            || (event.mediaType.isEmpty && event.stage == "video")
                        self.recordAnchorTelemetry(
                            anchorSlotIndex: event.anchorSlotIndex,
                            anchorExplicit: event.anchorExplicit,
                            turnIndex: index)
                        if let existingIndex = self.turns[index].blocks.firstIndex(where: {
                            $0.visualSlot?.slotID == slotID
                        }), var existing = self.turns[index].blocks[existingIndex].visualSlot {
                            if let mode = event.continuityMode, !mode.isEmpty {
                                existing.continuityMode = mode
                            }
                            if let anchor = event.anchorSlotIndex, anchor >= 0 {
                                existing.anchorSlotIndex = anchor
                            }
                            if event.anchorExplicit { existing.anchorExplicit = true }
                            if let sha = event.anchorSha256 { existing.anchorSha256 = sha }
                            self.turns[index].blocks[existingIndex] = .visual(existing)
                        } else {
                            self.turns[index].blocks.append(.visual(VisualSlotState(
                                slotID: slotID, slotIndex: slotIndex,
                                prompt: event.prompt,
                                activeKind: isVideo ? "video" : "image",
                                isLoading: true,
                                continuityMode: event.continuityMode,
                                anchorSlotIndex: event.anchorSlotIndex,
                                anchorExplicit: event.anchorExplicit,
                                anchorSha256: event.anchorSha256)))
                        }
                    case .legError:
                        let detail = event.error.isEmpty
                            ? "\(event.stage) output failed" : event.error
                        // A slot failure belongs on its card, not also in the
                        // turn-level list.
                        let attributed = self.markSlotError(detail, event: event, turnIndex: index)
                        if !attributed {
                            self.turns[index].legErrors.append(detail)
                        }
                        self.status = "Completed with leg error"
                    case .complete:
                        for leg in event.legs {
                            self.applyLegSummary(leg, turnIndex: index)
                        }
                    default:
                        break
                    }
                } onArtifact: { artifact in
                    guard self.activeTurn == assistant.id,
                          let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                        return
                    }
                    self.turns[index].artifacts.append(artifact)
                    if artifact.mediaType == "audio" {
                        if self.turns[index].telemetry.ttfaMs == nil {
                            self.turns[index].telemetry.ttfaMs = Self.elapsedMilliseconds(since: requestStart)
                        }
                    } else if artifact.mediaType == "image" || artifact.mediaType == "video" {
                        let visualKey = artifact.slotID ?? artifact.id
                        if self.activeSettledVisualKeys.insert(visualKey).inserted {
                            self.turns[index].telemetry.visualSettledCount = self.activeSettledVisualKeys.count
                            if self.turns[index].telemetry.firstVisualMs == nil {
                                self.turns[index].telemetry.firstVisualMs = Self.elapsedMilliseconds(since: requestStart)
                            }
                        }
                    }
                    if let slotID = artifact.slotID {
                        self.attachVisualArtifact(artifact, slotID: slotID,
                                                  turnIndex: index)
                    }
                } onAudioChunk: { chunk in
                    guard self.activeTurn == assistant.id else { return }
                    if let index = self.turns.firstIndex(where: { $0.id == assistant.id }),
                       self.turns[index].telemetry.ttfaMs == nil {
                        self.turns[index].telemetry.ttfaMs = Self.elapsedMilliseconds(since: requestStart)
                    }
                    self.streamingAudioTurnID = assistant.id
                    self.audioStream.enqueue(chunk)
                }
                guard self.activeTurn == assistant.id,
                      let index = self.turns.firstIndex(where: { $0.id == assistant.id }) else {
                    return
                }
                self.audioStream.finishStream()
                self.turns[index].text = result.text
                self.settleVisualSlots(at: index)
                if self.turns[index].telemetry.totalMs == nil {
                    self.turns[index].telemetry.totalMs = Self.elapsedMilliseconds(since: requestStart)
                }
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
                    self.settleVisualSlots(at: index, markMissingOutput: false)
                    if self.turns[index].text.isEmpty
                        && self.turns[index].artifacts.isEmpty
                        && self.turns[index].blocks.isEmpty {
                        self.turns.remove(at: index)
                    } else {
                        if cancelled, self.turns[index].telemetry.totalMs == nil {
                            self.turns[index].telemetry.totalMs = Self.elapsedMilliseconds(since: requestStart)
                        }
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

    // MARK: - Interleaved blocks and in-place visual swap

    private func appendTextDelta(_ delta: String, to turnIndex: Int) {
        turns[turnIndex].text += delta
        if let last = turns[turnIndex].blocks.indices.last,
           case .text(let id, let content) = turns[turnIndex].blocks[last] {
            turns[turnIndex].blocks[last] = .text(id: id, content: content + delta)
        } else {
            turns[turnIndex].blocks.append(.text(id: UUID(), content: delta))
        }
    }

    private func recordAnchorTelemetry(anchorSlotIndex: Int?, anchorExplicit: Bool,
                                       turnIndex: Int) {
        guard let anchor = anchorSlotIndex, anchor >= 0 else { return }
        turns[turnIndex].telemetry.hasI2VChain = true
        if anchorExplicit {
            turns[turnIndex].telemetry.hasExplicitAnchor = true
        }
    }

    private func attachVisualArtifact(_ artifact: GeneratedArtifact, slotID: String,
                                      turnIndex: Int) {
        guard artifact.mediaType == "image" || artifact.mediaType == "video",
              let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
                  $0.visualSlot?.slotID == slotID
              }),
              var slot = turns[turnIndex].blocks[blockIndex].visualSlot else { return }
        if artifact.mediaType == "image" {
            slot.imageArtifact = artifact
        } else {
            slot.videoArtifact = artifact
        }
        if let anchor = artifact.anchorSlotIndex, anchor >= 0 {
            slot.anchorSlotIndex = anchor
        }
        if artifact.anchorExplicit {
            slot.anchorExplicit = true
        }
        if let sha = artifact.anchorSha256 {
            slot.anchorSha256 = sha
        }
        recordAnchorTelemetry(
            anchorSlotIndex: slot.anchorSlotIndex,
            anchorExplicit: slot.anchorExplicit,
            turnIndex: turnIndex)
        slot.activeKind = artifact.mediaType
        slot.isLoading = false
        slot.error = nil
        turns[turnIndex].blocks[blockIndex] = .visual(slot)
    }

    @discardableResult
    private func markSlotError(_ detail: String, event: UnifiedEvent,
                               turnIndex: Int) -> Bool {
        guard let slotID = event.slotID, !slotID.isEmpty else { return false }
        if let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
            $0.visualSlot?.slotID == slotID
        }), var slot = turns[turnIndex].blocks[blockIndex].visualSlot {
            if let anchor = event.anchorSlotIndex, anchor >= 0 {
                slot.anchorSlotIndex = anchor
            }
            if event.anchorExplicit {
                slot.anchorExplicit = true
            }
            if let sha = event.anchorSha256 {
                slot.anchorSha256 = sha
            }
            recordAnchorTelemetry(
                anchorSlotIndex: slot.anchorSlotIndex,
                anchorExplicit: slot.anchorExplicit,
                turnIndex: turnIndex)
            slot.error = detail
            slot.isLoading = false
            turns[turnIndex].blocks[blockIndex] = .visual(slot)
            return true
        }
        // If an explicit-anchor leg_error arrives without a prior visual_slot,
        // synthesize a failed slot card so the error is shown on-card rather
        // than silently dropped.
        guard event.anchorExplicit else { return false }
        let isVideo = event.mediaType == "video"
            || (event.mediaType.isEmpty && event.stage == "video")
        let slotIndex = event.slotIndex ?? 0
        recordAnchorTelemetry(
            anchorSlotIndex: event.anchorSlotIndex,
            anchorExplicit: event.anchorExplicit,
            turnIndex: turnIndex)
        turns[turnIndex].blocks.append(.visual(VisualSlotState(
            slotID: slotID, slotIndex: slotIndex,
            prompt: event.prompt,
            activeKind: isVideo ? "video" : "image",
            isLoading: false,
            error: detail,
            continuityMode: event.continuityMode ?? NarrativeContinuityMode.i2vChain.rawValue,
            anchorSlotIndex: event.anchorSlotIndex,
            anchorExplicit: true,
            anchorSha256: event.anchorSha256)))
        return true
    }

    private func applyLegSummary(_ leg: UnifiedLegSummary, turnIndex: Int) {
        guard let slotID = leg.slotID, !slotID.isEmpty,
              let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
                  $0.visualSlot?.slotID == slotID
              }),
              var slot = turns[turnIndex].blocks[blockIndex].visualSlot else { return }
        if let anchor = leg.anchorSlotIndex, anchor >= 0 {
            slot.anchorSlotIndex = anchor
        }
        if leg.anchorExplicit {
            slot.anchorExplicit = true
        }
        if let sha = leg.anchorSha256 {
            slot.anchorSha256 = sha
        }
        recordAnchorTelemetry(
            anchorSlotIndex: slot.anchorSlotIndex,
            anchorExplicit: slot.anchorExplicit,
            turnIndex: turnIndex)
        if leg.state == "failed", let err = leg.error, !err.isEmpty,
           slot.error == nil, slot.imageArtifact == nil, slot.videoArtifact == nil {
            slot.error = err
            slot.isLoading = false
        }
        turns[turnIndex].blocks[blockIndex] = .visual(slot)
    }

    /// Stops the spinner of any initial slot generation that will never
    /// finish. An active swap is left alone: its own task settles it.
    private func settleVisualSlots(at turnIndex: Int,
                                   markMissingOutput: Bool = true) {
        for blockIndex in turns[turnIndex].blocks.indices {
            guard var slot = turns[turnIndex].blocks[blockIndex].visualSlot,
                  slot.isLoading, slot.swapToken == nil else { continue }
            slot.isLoading = false
            if markMissingOutput, slot.imageArtifact == nil,
               slot.videoArtifact == nil, slot.error == nil {
                slot.error = "No output for this visual slot."
            }
            turns[turnIndex].blocks[blockIndex] = .visual(slot)
        }
    }

    /// Swaps one slot to the other visual kind in place. A cached artifact
    /// switches instantly; otherwise only this slot shows a spinner while
    /// `POST /v1/visual/render` runs.
    func swapVisualSlot(turnID: UUID, slotID: String, toKind: String) {
        guard toKind == "image" || toKind == "video",
              let turnIndex = turns.firstIndex(where: { $0.id == turnID }),
              let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
                  $0.visualSlot?.slotID == slotID
              }),
              var slot = turns[turnIndex].blocks[blockIndex].visualSlot else { return }

        if (toKind == "image" && slot.imageArtifact != nil)
            || (toKind == "video" && slot.videoArtifact != nil) {
            slot.activeKind = toKind
            slot.isLoading = false
            slot.error = nil
            // Drop any in-flight fetch for the kind just left.
            slot.swapToken = nil
            turns[turnIndex].blocks[blockIndex] = .visual(slot)
            return
        }
        guard !slot.isLoading else { return }
        let token = UUID()
        slot.activeKind = toKind
        slot.isLoading = true
        slot.error = nil
        slot.swapToken = token
        turns[turnIndex].blocks[blockIndex] = .visual(slot)

        var size: String?
        var steps: Int?
        var numFrames: Int?
        var guidance: Double?
        if toKind == "image" {
            size = imageSize
            if let value = Int(imageStepsText.trimmingCharacters(in: .whitespaces)),
               value > 0 { steps = value }
        } else {
            size = videoSize
            if let value = Int(videoStepsText.trimmingCharacters(in: .whitespaces)),
               value > 0 { steps = value }
            if let value = Int(videoFramesText.trimmingCharacters(in: .whitespaces)),
               value > 0 { numFrames = value }
            if let value = Double(videoGuidanceText.trimmingCharacters(in: .whitespaces)),
               value >= 0 { guidance = value }
        }
        let seed = Int(seedText.trimmingCharacters(in: .whitespaces))
        let teaCacheThresh: Double? = (toKind == "video" && fastVideoMode) ? 0.05 : nil
        let prompt = slot.prompt
        let base = endpoint
        // A video swap animates the slot's existing image as the first frame.
        var imageDataURL: String?
        if toKind == "video", let image = slot.imageArtifact {
            imageDataURL = "data:\(image.mimeType);base64,"
                + image.data.base64EncodedString()
        }
        Task { [weak self] in
            guard let self else { return }
            do {
                let data = try await self.client.renderVisual(
                    base: base, kind: toKind, prompt: prompt, size: size,
                    steps: steps, numFrames: numFrames,
                    guidanceScale: guidance, seed: seed,
                    imageDataURL: imageDataURL,
                    teaCacheThresh: teaCacheThresh)
                self.completeSwap(turnID: turnID, slotID: slotID,
                                  toKind: toKind, token: token, data: data)
            } catch {
                self.failSwap(turnID: turnID, slotID: slotID, toKind: toKind,
                              token: token, message: error.localizedDescription)
            }
        }
    }

    private func completeSwap(turnID: UUID, slotID: String, toKind: String,
                              token: UUID, data: Data) {
        guard let turnIndex = turns.firstIndex(where: { $0.id == turnID }),
              let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
                  $0.visualSlot?.slotID == slotID
              }),
              var slot = turns[turnIndex].blocks[blockIndex].visualSlot,
              slot.activeKind == toKind, slot.isLoading,
              slot.swapToken == token else { return }
        let artifact = GeneratedArtifact(
            id: "swap-\(slotID)-\(toKind)", mediaType: toKind,
            mimeType: toKind == "image" ? "image/png" : "video/mp4",
            data: data, sha256: ArtifactCollector.sha256Hex(data),
            slotID: slotID, slotIndex: slot.slotIndex)
        if toKind == "image" {
            slot.imageArtifact = artifact
        } else {
            slot.videoArtifact = artifact
        }
        slot.isLoading = false
        slot.error = nil
        slot.swapToken = nil
        turns[turnIndex].blocks[blockIndex] = .visual(slot)
        // Keep the legacy list complete; the row renderer skips slot artifacts.
        turns[turnIndex].artifacts.append(artifact)
    }

    private func failSwap(turnID: UUID, slotID: String, toKind: String,
                          token: UUID, message: String) {
        guard let turnIndex = turns.firstIndex(where: { $0.id == turnID }),
              let blockIndex = turns[turnIndex].blocks.firstIndex(where: {
                  $0.visualSlot?.slotID == slotID
              }),
              var slot = turns[turnIndex].blocks[blockIndex].visualSlot,
              slot.activeKind == toKind, slot.isLoading,
              slot.swapToken == token else { return }
        // Show the kind that still has an artifact, if any.
        if slot.imageArtifact != nil {
            slot.activeKind = "image"
        } else if slot.videoArtifact != nil {
            slot.activeKind = "video"
        }
        slot.isLoading = false
        slot.error = message
        slot.swapToken = nil
        turns[turnIndex].blocks[blockIndex] = .visual(slot)
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
        guard !isRunning, !isListening, !isLiveVoiceActive else { return }
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

    // MARK: - Full-duplex Live Voice (WebSocket /v1/realtime, Stage B playback_ack)

    func startLiveVoice() {
        guard !isLiveVoiceActive else { return }
        if isRunning { stop() }
        if isListening { stopListening() }
        stopStreamingAudio()
        error = nil

        let trimmedVoice = audioVoice.trimmingCharacters(in: .whitespacesAndNewlines)
        let parsedMaxTokens = Int(maxTokensText.trimmingCharacters(in: .whitespaces))
        let config = RealtimeSessionConfig(
            voice: trimmedVoice.isEmpty ? "vivian" : trimmedVoice,
            silenceMs: liveVoiceSilenceMs,
            playbackAck: true,
            maxNewTokens: parsedMaxTokens.flatMap { $0 > 0 ? $0 : nil }
        )

        do {
            try realtime.start(
                base: endpoint,
                model: model,
                config: config,
                audioStream: audioStream,
                onEvent: { [weak self] event in
                    self?.handleRealtimeEvent(event)
                },
                onError: { [weak self] err in
                    guard let self else { return }
                    self.isLiveVoiceActive = false
                    self.liveVoiceState = .error(err.localizedDescription)
                    self.liveVoicePartialTranscript = ""
                    self.liveVoiceAssistantTurns.removeAll()
                    self.streamingAudioTurnID = nil
                    self.isStreamingAudioPlaying = false
                    self.error = err.localizedDescription
                    self.status = "Live Voice error"
                },
                onDisconnect: { [weak self] in
                    guard let self else { return }
                    self.isLiveVoiceActive = false
                    self.liveVoiceState = .idle
                    self.liveVoicePartialTranscript = ""
                    self.liveVoiceAssistantTurns.removeAll()
                    self.streamingAudioTurnID = nil
                    self.isStreamingAudioPlaying = false
                    if self.error == nil {
                        self.status = "Ready"
                    }
                }
            )
        } catch {
            self.error = error.localizedDescription
            status = "Configuration error"
            return
        }

        if !ProcessInfo.processInfo.arguments.contains("--automation") {
            UserDefaults.standard.set(endpoint, forKey: "endpoint")
            UserDefaults.standard.set(model, forKey: "model")
        }
        isLiveVoiceActive = true
        liveVoiceState = .connecting
        liveVoicePartialTranscript = ""
        liveVoiceLastAckCursor = 0
        liveVoiceAssistantTurns.removeAll()
        status = "Live Voice · Connecting…"
    }

    func stopLiveVoice() {
        guard isLiveVoiceActive || realtime.isActive else { return }
        realtime.stop()
        stopStreamingAudio()
        isLiveVoiceActive = false
        liveVoiceState = .idle
        liveVoicePartialTranscript = ""
        liveVoiceAssistantTurns.removeAll()
        if error == nil {
            status = "Ready"
        }
    }

    private func ensureRealtimeAssistantTurn(responseID: String) -> Int {
        if let turnID = liveVoiceAssistantTurns[responseID],
           let existing = turns.firstIndex(where: { $0.id == turnID }) {
            return existing
        }
        let assistant = ChatTurn(role: .assistant, text: "", state: .streaming)
        turns.append(assistant)
        liveVoiceAssistantTurns[responseID] = assistant.id
        return turns.count - 1
    }

    private func handleRealtimeEvent(_ event: RealtimeServerEvent) {
        switch event {
        case .sessionUpdated:
            liveVoiceState = .listening
            status = "Live Voice · Listening"

        case .speechStarted:
            liveVoiceState = .userSpeaking
            liveVoicePartialTranscript = ""
            status = "Live Voice · Hearing speech…"

        case .speechStopped:
            status = "Live Voice · Transcribing…"

        case .transcriptionSegment(_, _, let text, let isFinal):
            if !isFinal {
                liveVoicePartialTranscript = text
            }

        case .transcriptionCompleted(_, let text):
            liveVoicePartialTranscript = ""
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                turns.append(ChatTurn(role: .user, text: trimmed, state: .complete))
            }

        case .responseTextDelta(let responseID, _, let text, _):
            let index = ensureRealtimeAssistantTurn(responseID: responseID)
            appendTextDelta(text, to: index)
            liveVoiceState = .responding
            status = "Live Voice · Responding…"

        case .audioDelta(let delta):
            let index = ensureRealtimeAssistantTurn(responseID: delta.responseID)
            streamingAudioTurnID = turns[index].id
            liveVoiceState = .responding
            status = "Live Voice · Speaking…"

        case .audioDone:
            break

        case .responseCompleted(let responseID, _, _):
            if let turnID = liveVoiceAssistantTurns[responseID],
               let index = turns.firstIndex(where: { $0.id == turnID }),
               turns[index].state == .streaming {
                turns[index].state = .complete
            }
            if !realtime.stateMachine.playbackAckEnabled && !isStreamingAudioPlaying {
                liveVoiceState = .listening
                status = "Live Voice · Listening"
            }

        case .responsePlaybackAck(let responseID, _, let cursor, let isFinal):
            liveVoiceLastAckCursor = cursor
            if isFinal {
                if let turnID = liveVoiceAssistantTurns.removeValue(forKey: responseID),
                   let index = turns.firstIndex(where: { $0.id == turnID }),
                   turns[index].state == .streaming {
                    turns[index].state = .complete
                }
                liveVoiceState = .listening
                status = "Live Voice · Listening"
            }

        case .responseCancelled(let cancellation):
            streamingAudioTurnID = nil
            isStreamingAudioPlaying = false
            if let turnID = liveVoiceAssistantTurns.removeValue(forKey: cancellation.responseID),
               let index = turns.firstIndex(where: { $0.id == turnID }) {
                if let ackText = cancellation.acknowledgedText {
                    if ackText.isEmpty && turns[index].artifacts.isEmpty {
                        turns.remove(at: index)
                    } else {
                        turns[index].text = ackText
                        turns[index].blocks = ackText.isEmpty
                            ? [] : [.text(id: UUID(), content: ackText)]
                        turns[index].state = .stopped
                    }
                } else if turns[index].text.isEmpty && turns[index].artifacts.isEmpty {
                    turns.remove(at: index)
                } else {
                    turns[index].state = .stopped
                }
            }
            liveVoiceState = .listening
            status = "Live Voice · Interrupted"

        case .responseFailed(let responseID, _, let detail):
            streamingAudioTurnID = nil
            if let responseID,
               let turnID = liveVoiceAssistantTurns.removeValue(forKey: responseID),
               let index = turns.firstIndex(where: { $0.id == turnID }) {
                if turns[index].text.isEmpty && turns[index].artifacts.isEmpty {
                    turns.remove(at: index)
                } else {
                    turns[index].state = .failed
                }
            }
            self.error = detail
            liveVoiceState = .listening
            status = "Live Voice · Error"

        case .error(_, let message):
            self.error = message
            liveVoiceState = .error(message)
            status = "Live Voice error"

        case .unknown:
            break
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
        stopLiveVoice()
        stopStreamingAudio()
        requestTask?.cancel()
        requestTask = nil
        activeTurn = nil
        activeSettledVisualKeys.removeAll()
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
