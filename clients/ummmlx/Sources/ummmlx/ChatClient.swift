// SPDX-License-Identifier: Apache-2.0
// Modified from Voxt internal/llm/openai_http.go and openai.go at commit
// 5645daf0480e9bf572b45a2bba8b8e2f1e642daa. See ThirdParty/Voxt/LICENSE
// and NOTICE. Changes: Swift concurrency, incremental parsing, strict limits,
// required [DONE], unified events, and the ummInfer request contract.

import Foundation

struct ChatMessage: Codable, Equatable {
    let role: String
    let content: String
    var imageDataURL: String? = nil
    var videoDataURL: String? = nil

    init(role: String, content: String, imageDataURL: String? = nil,
         videoDataURL: String? = nil) {
        self.role = role
        self.content = content
        self.imageDataURL = imageDataURL
        self.videoDataURL = videoDataURL
    }
}

struct ChatCompletion {
    let text: String
    let finishReason: String
    let sessionID: String
    let artifacts: [GeneratedArtifact]
}

struct ImageOptions: Equatable {
    var prompt: String?
    var size = "512x512"
    var steps: Int?
    var seed: Int?
    var imageDataURL: String?
    var outputFormat = "png"
}

struct AudioOptions: Equatable {
    var voice: String
    var format = "wav"
    var stream = true
    var refAudio: String?
}

struct VideoOptions: Equatable {
    var prompt: String?
    var size = "704x480"
    var seconds: Int?
    var fps: Int?
    var numFrames: Int?
    var steps: Int?
    var guidanceScale: Double?
    var seed: Int?
    /// First-frame data URL for native image-to-video animation (Wan2.2-TI2V).
    var imageDataURL: String?
    var outputFormat = "mp4"
}

/// Interleaved narrative mode: the server's VLM director inserts visual beats
/// into the prose and emits a `visual_slot` event for each one.
struct NarrativeOptions: Equatable {
    var enabled = true
    /// "auto", "image" or "video".
    var visualMode = "auto"
    /// 1...6; the server rejects anything else.
    var maxSlots = 3
    /// When true in auto mode, allows zero-visual-slot pure-text answers for
    /// math, code, or factual prompts.
    var allowTextOnly = true
}

struct ChatStreamOptions: Equatable {
    var sessionID: String?
    var maxTokens: Int?
    var temperature: Double?
    var topP: Double?
    var seed: Int64?
    var image: ImageOptions?
    var audio: AudioOptions?
    var video: VideoOptions?
    var narrative: NarrativeOptions?
    /// Unified servers hold multi-turn context themselves: the wire carries one
    /// user message plus session_history=true. Set by the capability fallback
    /// when a server rejects replayed history.
    var serverHistory = false

    var modalities: [String] {
        var values = ["text"]
        if image != nil { values.append("image") }
        if audio != nil { values.append("audio") }
        if video != nil { values.append("video") }
        return values
    }
}

enum ChatClientError: LocalizedError, Equatable {
    case invalidEndpoint(String)
    case modelRequired
    case http(Int, String)
    case invalidResponse(String)
    case streamTruncated
    case cancelled
    case unifiedError(String)

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint(let detail): return detail
        case .modelRequired: return "Model is required."
        case .http(let status, let detail): return "HTTP \(status): \(detail)"
        case .invalidResponse(let detail): return "Invalid server response: \(detail)"
        case .streamTruncated: return "The stream ended before [DONE]."
        case .cancelled: return "The server cancelled the request."
        case .unifiedError(let detail): return "The server reported an error: \(detail)"
        }
    }
}

struct SSEParser {
    private var line = Data()
    private var dataLines: [String] = []
    private var dataBytes = 0
    private let limit = 4 * 1024 * 1024

    mutating func feed(_ byte: UInt8) throws -> String? {
        if byte == 0x0A {
            return try consumeLine()
        }
        line.append(byte)
        if line.count > limit + (line.last == 0x0D ? 1 : 0) {
            throw ChatClientError.invalidResponse("an SSE line exceeds 4 MiB")
        }
        return nil
    }

    /// Line-oriented entry point. `bytes.lines` already splits the stream, so
    /// this skips the per-byte loop that made a large artifact line take
    /// seconds to traverse.
    mutating func feed(line text: String) throws -> String? {
        var value = text
        if value.hasSuffix("\r") { value.removeLast() }
        if value.utf8.count > limit {
            throw ChatClientError.invalidResponse("an SSE line exceeds 4 MiB")
        }
        if value.isEmpty { return try flush() }
        let parts = value.split(separator: ":", maxSplits: 1,
                                omittingEmptySubsequences: false)
        guard parts.first == "data" else { return nil }
        var payload = parts.count == 2 ? String(parts[1]) : ""
        if payload.first == " " { payload.removeFirst() }
        dataBytes += payload.utf8.count + (dataLines.isEmpty ? 0 : 1)
        if dataBytes > limit {
            throw ChatClientError.invalidResponse("an SSE event exceeds 4 MiB")
        }
        dataLines.append(payload)
        // AsyncLineSequence does not surface the blank separator line, so a
        // buffered multi-line event would run into the next one. The server
        // emits exactly one data line per event, so flush on the data line.
        return try flush()
    }


    mutating func finish() throws -> String? {
        if !line.isEmpty {
            if let event = try consumeLine() { return event }
        }
        return try flush()
    }

    private mutating func consumeLine() throws -> String? {
        defer { line.removeAll(keepingCapacity: true) }
        if line.last == 0x0D { line.removeLast() }
        if line.isEmpty { return try flush() }
        guard let text = String(data: line, encoding: .utf8) else {
            throw ChatClientError.invalidResponse("an SSE line is not UTF-8")
        }
        let parts = text.split(separator: ":", maxSplits: 1,
                               omittingEmptySubsequences: false)
        guard parts[0] == "data" else { return nil }
        var value = parts.count == 2 ? String(parts[1]) : ""
        if value.first == " " { value.removeFirst() }
        dataBytes += value.utf8.count + (dataLines.isEmpty ? 0 : 1)
        if dataBytes > limit {
            throw ChatClientError.invalidResponse("an SSE event exceeds 4 MiB")
        }
        dataLines.append(value)
        return nil
    }

    private mutating func flush() throws -> String? {
        guard !dataLines.isEmpty else {
            dataBytes = 0
            return nil
        }
        let event = dataLines.joined(separator: "\n")
        dataLines.removeAll(keepingCapacity: true)
        dataBytes = 0
        return event
    }
}

private final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate,
                                        @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

@MainActor
final class ChatClient {
    private let session: URLSession
    /// Text streams stay at 64 MiB. Mixed output carries base64 artifacts, so
    /// the wire is larger; the decoded 48/60 MiB checks remain the real limits.
    private let textResponseLimit = 64 * 1024 * 1024
    private let mixedResponseLimit = 96 * 1024 * 1024

    init(session: URLSession? = nil) {
        if let session {
            self.session = session
            return
        }
        let configuration = URLSessionConfiguration.ephemeral
        // Image generation measured 227 s on the NVIDIA Hopper (SM90) native chain; the text
        // path is unchanged in practice. Stop still cancels immediately.
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 900
        configuration.connectionProxyDictionary = [:]
        self.session = URLSession(configuration: configuration,
                                  delegate: NoRedirectDelegate(), delegateQueue: nil)
    }

    /// Plain HTTP is allowed for loopback, private literal addresses and any
    /// host named in UMMMLX_HTTP_HOSTS. Everything else must use HTTPS.
    static func plainHTTPHostAllowlist(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Set<String> {
        guard let raw = environment["UMMMLX_HTTP_HOSTS"] else { return [] }
        return Set(raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty })
    }

    static func chatURL(base: String,
                        allowedHTTPHosts: Set<String> = plainHTTPHostAllowlist()) throws -> URL {
        try endpointURL(base: base, endpoint: "/chat/completions",
                        allowedHTTPHosts: allowedHTTPHosts)
    }

    /// Direct single-stage visual swap: `POST <base>/visual/render`. Same URL
    /// security policy as the chat endpoint.
    static func visualRenderURL(base: String,
                                allowedHTTPHosts: Set<String> = plainHTTPHostAllowlist()) throws -> URL {
        try endpointURL(base: base, endpoint: "/visual/render",
                        allowedHTTPHosts: allowedHTTPHosts)
    }

    private static func endpointURL(base: String, endpoint: String,
                                    allowedHTTPHosts: Set<String>) throws -> URL {
        let text = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              let host = components.host?.lowercased(), !host.isEmpty,
              scheme == "http" || scheme == "https",
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else {
            throw ChatClientError.invalidEndpoint(
                "Use an HTTP(S) base URL without credentials, query, or fragment.")
        }
        if scheme == "http" && !isPlainHTTPHost(host, allowed: allowedHTTPHosts) {
            throw ChatClientError.invalidEndpoint(
                "Plain HTTP is allowed only on loopback, a private address, "
                + "or a host named in UMMMLX_HTTP_HOSTS. Use HTTPS for other hosts.")
        }
        var mutable = components
        var path = mutable.path
        while path.count > 1 && path.last == "/" { path.removeLast() }
        if path.isEmpty || path == "/" { path = "/v1" }
        mutable.path = path + endpoint
        guard let url = mutable.url else {
            throw ChatClientError.invalidEndpoint("The base URL is invalid.")
        }
        return url
    }

    static func isPlainHTTPHost(_ host: String, allowed: Set<String>) -> Bool {
        if isLoopback(host) { return true }
        if allowed.contains(host) { return true }
        if let octets = ipv4Octets(host) {
            if octets[0] == 10 { return true }
            if octets[0] == 172 && (16...31).contains(octets[1]) { return true }
            if octets[0] == 192 && octets[1] == 168 { return true }
            return false
        }
        // IPv6 unique local addresses (fc00::/7). Brackets are stripped by
        // URLComponents in some versions, so trim them defensively.
        let bare = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast()) : host
        if bare.contains(":") && (bare.hasPrefix("fc") || bare.hasPrefix("fd")) {
            return true
        }
        return false
    }

    private static func isLoopback(_ host: String) -> Bool {
        if host == "localhost" || host == "::1" || host == "[::1]" { return true }
        guard let octets = ipv4Octets(host) else { return false }
        return octets[0] == 127
    }

    private static func ipv4Octets(_ host: String) -> [Int]? {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [Int] = []
        for part in parts {
            guard !part.isEmpty, part.allSatisfy(\.isASCII),
                  part.allSatisfy(\.isNumber), let value = Int(part),
                  value >= 0, value <= 255 else { return nil }
            octets.append(value)
        }
        return octets
    }

    static func requestBody(model: String, messages: [ChatMessage],
                            options: ChatStreamOptions) -> [String: Any] {
        let encodedMessages: [[String: Any]] = messages.map { message in
            if let imageURL = message.imageDataURL, !imageURL.isEmpty {
                var parts: [[String: Any]] = []
                if !message.content.isEmpty {
                    parts.append(["type": "text", "text": message.content])
                }
                parts.append(["type": "image_url", "image_url": ["url": imageURL]])
                return ["role": message.role, "content": parts]
            }
            if let videoURL = message.videoDataURL, !videoURL.isEmpty {
                var parts: [[String: Any]] = []
                if !message.content.isEmpty {
                    parts.append(["type": "text", "text": message.content])
                }
                parts.append(["type": "video_url", "video_url": ["url": videoURL]])
                return ["role": message.role, "content": parts]
            }
            return ["role": message.role, "content": message.content]
        }
        var body: [String: Any] = [
            "model": model,
            "messages": encodedMessages,
            "max_tokens": options.maxTokens ?? 64,
            "temperature": options.temperature ?? 0,
            "stream": true
        ]
        if let topP = options.topP { body["top_p"] = topP }
        if let seed = options.seed { body["seed"] = seed }
        if let sessionID = options.sessionID, !sessionID.isEmpty {
            body["session_id"] = sessionID
        }
        if options.serverHistory { body["session_history"] = true }
        if let image = options.image {
            body["modalities"] = options.modalities
            var object: [String: Any] = [
                "size": image.size, "output_format": image.outputFormat
            ]
            if let prompt = image.prompt, !prompt.isEmpty { object["prompt"] = prompt }
            if let steps = image.steps { object["steps"] = steps }
            if let seed = image.seed { object["seed"] = seed }
            if let imageDataURL = image.imageDataURL, !imageDataURL.isEmpty {
                object["image"] = imageDataURL
            }
            body["image"] = object
        }
        if let audio = options.audio {
            body["modalities"] = options.modalities
            var object: [String: Any] = [
                "voice": audio.voice, "format": audio.format
            ]
            if let refAudio = audio.refAudio, !refAudio.isEmpty {
                object["ref_audio"] = refAudio
            }
            // Wave 10 makes TTS a global streaming leg: it runs from the first
            // clean sentence while image, video or narrative legs render.
            if audio.stream {
                object["stream"] = true
            }
            body["audio"] = object
        }
        if let video = options.video {
            body["modalities"] = options.modalities
            var object: [String: Any] = [
                "size": video.size, "output_format": video.outputFormat
            ]
            if let prompt = video.prompt, !prompt.isEmpty { object["prompt"] = prompt }
            if let seconds = video.seconds { object["seconds"] = seconds }
            if let fps = video.fps { object["fps"] = fps }
            if let numFrames = video.numFrames { object["num_frames"] = numFrames }
            if let steps = video.steps { object["steps"] = steps }
            if let guidance = video.guidanceScale { object["guidance_scale"] = guidance }
            if let seed = video.seed { object["seed"] = seed }
            if let imageDataURL = video.imageDataURL, !imageDataURL.isEmpty {
                object["image"] = imageDataURL
            }
            body["video"] = object
        }
        if let narrative = options.narrative {
            body["narrative"] = [
                "enabled": narrative.enabled,
                "visual_mode": narrative.visualMode,
                "max_slots": narrative.maxSlots,
                "allow_text_only": narrative.allowTextOnly
            ]
        }
        return body
    }

    /// Sends one chat request and adapts the wire shape to the server's
    /// capability on documented 400s. A 400 arrives before any event, so a
    /// retry cannot duplicate text or artifacts. At most three shapes are
    /// tried; the loop cannot alternate between them.
    func stream(base: String, model: String, messages: [ChatMessage],
                options: ChatStreamOptions = ChatStreamOptions(),
                onDelta: (String) -> Void,
                onEvent: ((UnifiedEvent) -> Void)? = nil,
                onArtifact: ((GeneratedArtifact) -> Void)? = nil,
                onAudioChunk: ((AudioStreamChunk) -> Void)? = nil) async throws -> ChatCompletion {
        var shape = options
        var wire = messages
        var historyUnavailable = false
        var attempts = 0
        while true {
            do {
                return try await attempt(
                    base: base, model: model, messages: wire, options: shape,
                    onDelta: onDelta, onEvent: onEvent, onArtifact: onArtifact,
                    onAudioChunk: onAudioChunk)
            } catch let ChatClientError.http(status, detail)
                        where status == 400 && attempts < 3 {
                attempts += 1
                // A non-unified server rejects unified-only fields. Drop the
                // session id and keep the replayed history.
                if detail.contains("session_id requires unified mode"),
                   shape.sessionID != nil {
                    shape.sessionID = nil
                    continue
                }
                // The deployment has no server-side history. Send the single
                // message without the flag; context is then not preserved.
                if detail.contains("session_history is not enabled"),
                   shape.serverHistory {
                    shape.serverHistory = false
                    historyUnavailable = true
                    continue
                }
                // The unified route accepts one text unit. It keeps multi-turn
                // context itself when session_history is set.
                if detail.contains("at most one text part"), !shape.serverHistory,
                   !historyUnavailable, wire.count > 1, let last = wire.last {
                    shape.serverHistory = true
                    wire = [last]
                    continue
                }
                throw ChatClientError.http(status, detail)
            }
        }
    }

    /// Renders one image or video for an in-place narrative slot swap.
    /// Returns the raw PNG or MP4 bytes.
    func renderVisual(base: String, kind: String, prompt: String,
                      size: String? = nil, steps: Int? = nil,
                      numFrames: Int? = nil, guidanceScale: Double? = nil,
                      seed: Int? = nil,
                      imageDataURL: String? = nil) async throws -> Data {
        guard kind == "image" || kind == "video" else {
            throw ChatClientError.invalidEndpoint(
                "A visual render kind must be image or video.")
        }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ChatClientError.invalidEndpoint("A visual render prompt is required.")
        }
        var request = URLRequest(url: try Self.visualRenderURL(base: base))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["kind": kind, "prompt": trimmed]
        if let size, !size.isEmpty { body["size"] = size }
        if let steps, steps > 0 { body["steps"] = steps }
        if let numFrames, numFrames > 0 { body["num_frames"] = numFrames }
        if let guidanceScale { body["guidance_scale"] = guidanceScale }
        if let seed { body["seed"] = seed }
        if let imageDataURL, !imageDataURL.isEmpty { body["image"] = imageDataURL }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ChatClientError.invalidResponse("missing HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ChatClientError.http(http.statusCode, Self.errorDetail(data))
        }
        guard !data.isEmpty else {
            throw ChatClientError.invalidResponse("the visual render returned no bytes")
        }
        return data
    }

    private func attempt(base: String, model: String, messages: [ChatMessage],
                         options: ChatStreamOptions,
                         onDelta: (String) -> Void,
                         onEvent: ((UnifiedEvent) -> Void)?,
                         onArtifact: ((GeneratedArtifact) -> Void)?,
                         onAudioChunk: ((AudioStreamChunk) -> Void)?) async throws -> ChatCompletion {
        let name = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw ChatClientError.modelRequired }
        let url = try Self.chatURL(base: base)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: Self.requestBody(model: name, messages: messages,
                                             options: options))

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ChatClientError.invalidResponse("missing HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = try await readLimited(bytes, limit: 64 * 1024)
            throw ChatClientError.http(http.statusCode, Self.errorDetail(body))
        }
        let contentType = http.value(forHTTPHeaderField: "Content-Type")?
            .split(separator: ";").first?.trimmingCharacters(in: .whitespaces)
            .lowercased()
        guard contentType == "text/event-stream" else {
            throw ChatClientError.invalidResponse("expected text/event-stream")
        }

        let mixed = options.image != nil || options.audio != nil || options.video != nil
        let wireLimit = mixed ? mixedResponseLimit : textResponseLimit
        var parser = SSEParser()
        var sequence = UnifiedSequenceGuard()
        var collector = ArtifactCollector()
        var streamedAudio = StreamedAudioAccumulator()
        var text = ""
        var finishReason = ""
        var sessionID = ""
        var artifacts: [GeneratedArtifact] = []
        var received = 0
        var done = false
        var unifiedSeen = false
        var terminalCount = 0

        func consume(_ event: String) throws -> Bool {
            if event == "[DONE]" { return true }
            let parsed: Any
            do {
                parsed = try JSONSerialization.jsonObject(with: Data(event.utf8))
            } catch {
                throw ChatClientError.invalidResponse("an SSE event is not JSON")
            }
            guard let root = parsed as? [String: Any] else {
                throw ChatClientError.invalidResponse("an SSE event is not an object")
            }
            if let error = root["error"], !(error is NSNull) {
                throw ChatClientError.http(500, Self.errorDetail(Data(event.utf8)))
            }
            if let rawUnified = root["unified"], !(rawUnified is NSNull) {
                guard let object = rawUnified as? [String: Any] else {
                    throw ChatClientError.invalidResponse("unified is not an object")
                }
                guard terminalCount == 0 else {
                    throw ChatClientError.invalidResponse(
                        "a unified event follows the terminal event")
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                let unified = try UnifiedDecode.decode(data)
                try sequence.accept(unified.seq)
                unifiedSeen = true
                if unified.kind.isTerminal { terminalCount = 1 }
                if !unified.sessionID.isEmpty { sessionID = unified.sessionID }
                if unified.kind == .artifact, let chunk = unified.artifact,
                   let completed = try collector.ingest(chunk) {
                    if completed.mimeType == "audio/pcm" {
                        let streamChunk = try streamedAudio.append(completed)
                        onAudioChunk?(streamChunk)
                    } else {
                        artifacts.append(completed)
                        onArtifact?(completed)
                    }
                }
                if unified.kind == .complete,
                   let wavArtifact = try streamedAudio.finalizeWAV() {
                    artifacts.append(wavArtifact)
                    onArtifact?(wavArtifact)
                }
                onEvent?(unified)
                switch unified.kind {
                case .error:
                    throw ChatClientError.unifiedError(
                        unified.error.isEmpty ? "the request failed" : unified.error)
                case .cancelled:
                    throw ChatClientError.cancelled
                default:
                    break
                }
            }
            if let rawChoices = root["choices"] {
                guard let choices = rawChoices as? [Any] else {
                    throw ChatClientError.invalidResponse("choices is not an array")
                }
                for rawChoice in choices {
                    guard let choice = rawChoice as? [String: Any] else {
                        throw ChatClientError.invalidResponse("a choice is not an object")
                    }
                    if let rawDelta = choice["delta"], !(rawDelta is NSNull) {
                        guard let delta = rawDelta as? [String: Any] else {
                            throw ChatClientError.invalidResponse("delta is not an object")
                        }
                        if let rawContent = delta["content"], !(rawContent is NSNull) {
                            guard let content = rawContent as? String else {
                                throw ChatClientError.invalidResponse("delta.content is not a string")
                            }
                            if !content.isEmpty {
                                text += content
                                if text.utf8.count > textResponseLimit {
                                    throw ChatClientError.invalidResponse("response text exceeds 64 MiB")
                                }
                                onDelta(content)
                            }
                        }
                    }
                    if let rawFinish = choice["finish_reason"], !(rawFinish is NSNull) {
                        guard let finish = rawFinish as? String else {
                            throw ChatClientError.invalidResponse("finish_reason is not a string")
                        }
                        if !finish.isEmpty { finishReason = finish }
                    }
                }
            }
            return false
        }

        // Reading the stream byte by byte costs one async iteration per byte.
        // An audio artifact line is tens of kilobytes, so that loop fell behind
        // the server and the player starved between chunks. Lines are the SSE
        // unit anyway, so iterate over them.
        stream: for try await line in bytes.lines {
            try Task.checkCancellation()
            received += line.utf8.count + 1
            if received > wireLimit {
                throw ChatClientError.invalidResponse("stream exceeds its byte limit")
            }
            if let event = try parser.feed(line: line), try consume(event) {
                done = true
                break stream
            }
        }
        if !done, let event = try parser.finish() {
            done = try consume(event)
        }
        try Task.checkCancellation()
        guard done else { throw ChatClientError.streamTruncated }
        guard terminalCount == 1 || !unifiedSeen else {
            throw ChatClientError.invalidResponse(
                "the unified stream ended without a terminal event")
        }
        guard !collector.hasIncompleteArtifacts else {
            throw ChatClientError.invalidResponse(
                "an artifact stream ended before its final chunk")
        }
        guard !text.isEmpty || finishReason == "length" || !artifacts.isEmpty else {
            throw ChatClientError.invalidResponse("the model returned no text")
        }
        return ChatCompletion(text: text, finishReason: finishReason,
                              sessionID: sessionID, artifacts: artifacts)
    }

    private func readLimited(_ bytes: URLSession.AsyncBytes, limit: Int) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            if data.count >= limit { break }
        }
        return data
    }

    static func errorDetail(_ data: Data) -> String {
        var detail = ""
        if let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let error = root["error"] as? String {
                detail = error
            } else if let error = root["error"] as? [String: Any] {
                detail = [error["type"], error["code"], error["message"]]
                    .compactMap { $0 as? String }.filter { !$0.isEmpty }
                    .joined(separator: ": ")
            }
            if detail.isEmpty { detail = root["message"] as? String ?? "" }
            if detail.isEmpty, root["detail"] != nil {
                detail = String(decoding: data, as: UTF8.self)
            }
        } else {
            detail = String(decoding: data, as: UTF8.self)
        }
        detail = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        if detail.isEmpty { detail = "no error detail" }
        return String(detail.prefix(500))
    }
}
