// SPDX-License-Identifier: Apache-2.0
// Full-duplex realtime voice WebSocket client (Stage B playback_ack + barge-in).
// Implements the /v1/realtime wire protocol documented in docs/full-duplex.md.

import AVFoundation
import CoreFoundation
import Foundation

enum RealtimeSessionState: Equatable, Sendable {
    case idle
    case connecting
    case listening
    case userSpeaking
    case responding
    case error(String)
}

struct RealtimeSessionConfig: Equatable, Sendable {
    var voice: String = "vivian"
    var silenceMs: Int = 400
    var playbackAck: Bool = true
    var maxNewTokens: Int? = nil
}

struct RealtimeSessionSnapshot: Equatable, Sendable {
    let inputAudioFormat: String
    let serverVADSilenceMs: Int?
    let voice: String?
    let playbackAck: Bool
    let maxNewTokens: Int?
}

struct RealtimeAudioDelta: Equatable, Sendable {
    let responseID: String
    let segmentID: Int
    let pcm16: Data
    let sampleRate: Int
    let channels: Int
    let textCursor: Int?
    let textEndCursor: Int?

    var chunk: AudioStreamChunk {
        AudioStreamChunk(pcm16: pcm16, sampleRate: sampleRate, channels: channels)
    }
}

struct RealtimeCancellation: Equatable, Sendable {
    let responseID: String
    let segmentID: Int
    let reason: String
    let textCursor: Int?
    let acknowledgedText: String?
    let historyRecorded: Bool?
}

struct RealtimePlaybackAck: Equatable, Sendable {
    let responseID: String
    let textCursor: Int
}

enum RealtimeServerEvent: Equatable, Sendable {
    case sessionUpdated(RealtimeSessionSnapshot)
    case speechStarted(segmentID: Int, interruptedResponseID: String?)
    case speechStopped(segmentID: Int, audioMs: Int?)
    case transcriptionSegment(segmentID: Int, eventIndex: Int, text: String, isFinal: Bool)
    case transcriptionCompleted(segmentID: Int, text: String)
    case responseTextDelta(responseID: String, segmentID: Int, text: String, textCursor: Int?)
    case audioDelta(RealtimeAudioDelta)
    case audioDone(responseID: String, segmentID: Int, textCursor: Int?)
    case responseCompleted(responseID: String, segmentID: Int, textCursor: Int?)
    case responsePlaybackAck(responseID: String, segmentID: Int, textCursor: Int, isFinal: Bool)
    case responseCancelled(RealtimeCancellation)
    case responseFailed(responseID: String?, segmentID: Int, error: String)
    case error(code: String?, message: String)
    case unknown(type: String)
}

enum RealtimeClientError: LocalizedError, Equatable, Sendable {
    case invalidEvent(String)
    case invalidAudio(String)
    case serverError(String)
    case notConnected
    case microphoneDenied
    case audioCaptureFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidEvent(let detail):
            return "Invalid realtime event: \(detail)"
        case .invalidAudio(let detail):
            return "Invalid realtime audio: \(detail)"
        case .serverError(let detail):
            return "Realtime server error: \(detail)"
        case .notConnected:
            return "The realtime session is not connected."
        case .microphoneDenied:
            return "Microphone access is not authorized. Allow ummmlx in System Settings > Privacy & Security > Microphone."
        case .audioCaptureFailed(let detail):
            return "Audio capture failed: \(detail)"
        }
    }
}

/// Wire JSON encoder/decoder matching `src/serve/realtime_api.cpp`.
enum RealtimeWire {
    static func encodeSessionUpdate(_ config: RealtimeSessionConfig) throws -> String {
        let voice = config.voice.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedVoice = voice.isEmpty ? "vivian" : voice
        let clampedSilence = min(max(config.silenceMs, 100), 5000)

        var audioOutput: [String: Any] = ["voice": resolvedVoice]
        if config.playbackAck {
            audioOutput["playback_ack"] = true
        }
        var session: [String: Any] = [
            "input_audio_format": "pcm16",
            "turn_detection": [
                "type": "server_vad",
                "silence_ms": clampedSilence,
            ],
            "audio_output": audioOutput,
        ]
        if let maxTokens = config.maxNewTokens, (1...4096).contains(maxTokens) {
            session["max_new_tokens"] = maxTokens
        }
        return try serialize([
            "type": "session.update",
            "session": session,
        ])
    }

    static func encodeAudioAppend(_ pcm16: Data) throws -> String {
        guard !pcm16.isEmpty, pcm16.count % 2 == 0 else {
            throw RealtimeClientError.invalidAudio("PCM16 frame must be non-empty with whole samples")
        }
        return try serialize([
            "type": "input_audio_buffer.append",
            "audio": pcm16.base64EncodedString(),
        ])
    }

    static func encodePlaybackAck(responseID: String, textCursor: Int) throws -> String {
        guard !responseID.isEmpty else {
            throw RealtimeClientError.invalidEvent("playback_ack requires a response_id")
        }
        guard textCursor >= 0 else {
            throw RealtimeClientError.invalidEvent("playback_ack text_cursor must be non-negative")
        }
        return try serialize([
            "type": "output_audio_buffer.playback_ack",
            "response_id": responseID,
            "text_cursor": textCursor,
        ])
    }

    static func decodeServerEvent(_ text: String) throws -> RealtimeServerEvent {
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: Data(text.utf8))
        } catch {
            throw RealtimeClientError.invalidEvent("frame is not valid JSON")
        }
        guard let root = parsed as? [String: Any] else {
            throw RealtimeClientError.invalidEvent("frame is not a JSON object")
        }
        let type = try requireString(root["type"], "type")

        switch type {
        case "transcription_session.updated":
            guard let session = root["session"] as? [String: Any] else {
                throw RealtimeClientError.invalidEvent("transcription_session.updated requires session object")
            }
            let inputFormat = try optionalString(session["input_audio_format"], "session.input_audio_format") ?? "pcm16"
            var silenceMs: Int?
            if let turn = session["turn_detection"] as? [String: Any] {
                silenceMs = try optionalInt(turn["silence_ms"], "session.turn_detection.silence_ms")
            }
            var voice: String?
            var playbackAck = false
            if let audio = session["audio_output"] as? [String: Any] {
                voice = try optionalString(audio["voice"], "session.audio_output.voice")
                playbackAck = try optionalBool(audio["playback_ack"], "session.audio_output.playback_ack") ?? false
            }
            let maxTokens = try optionalInt(session["max_new_tokens"], "session.max_new_tokens")
            return .sessionUpdated(RealtimeSessionSnapshot(
                inputAudioFormat: inputFormat,
                serverVADSilenceMs: silenceMs,
                voice: voice,
                playbackAck: playbackAck,
                maxNewTokens: maxTokens
            ))

        case "input_audio_buffer.speech_started":
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let interrupted = try optionalString(root["interrupted_response_id"], "interrupted_response_id")
            return .speechStarted(segmentID: segmentID, interruptedResponseID: interrupted)

        case "input_audio_buffer.speech_stopped":
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let audioMs = try optionalInt(root["audio_ms"], "audio_ms")
            return .speechStopped(segmentID: segmentID, audioMs: audioMs)

        case "transcription.segment":
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let eventIndex = try optionalInt(root["event_index"], "event_index") ?? 0
            let segmentText = try optionalString(root["text"], "text") ?? ""
            let isFinal = try optionalBool(root["is_final"], "is_final") ?? false
            return .transcriptionSegment(
                segmentID: segmentID, eventIndex: eventIndex,
                text: segmentText, isFinal: isFinal
            )

        case "transcription.completed":
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let finalText = try optionalString(root["text"], "text") ?? ""
            return .transcriptionCompleted(segmentID: segmentID, text: finalText)

        case "response.text.delta":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let deltaText = try optionalString(root["text"], "text") ?? ""
            let textCursor = try optionalNonNegativeInt(root["text_cursor"], "text_cursor")
            return .responseTextDelta(
                responseID: responseID, segmentID: segmentID,
                text: deltaText, textCursor: textCursor
            )

        case "audio.delta":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            if let format = try optionalString(root["format"], "format"), format != "pcm16" {
                throw RealtimeClientError.invalidAudio("unsupported audio.delta format '\(format)'")
            }
            let sampleRate = try optionalInt(root["sample_rate"], "sample_rate") ?? 24_000
            let channels = try optionalInt(root["channels"], "channels") ?? 1
            guard sampleRate > 0, (1...8).contains(channels) else {
                throw RealtimeClientError.invalidAudio("invalid audio.delta sample_rate or channels")
            }
            let encoded = try requireString(root["audio"], "audio")
            guard let pcm = Data(base64Encoded: encoded, options: []) else {
                throw RealtimeClientError.invalidAudio("audio.delta is not valid base64")
            }
            let frameBytes = channels * 2
            guard !pcm.isEmpty, pcm.count % frameBytes == 0 else {
                throw RealtimeClientError.invalidAudio("audio.delta PCM16 is empty or not sample-aligned")
            }
            let textCursor = try optionalNonNegativeInt(root["text_cursor"], "text_cursor")
            let textEndCursor = try optionalNonNegativeInt(root["text_end_cursor"], "text_end_cursor")
            return .audioDelta(RealtimeAudioDelta(
                responseID: responseID,
                segmentID: segmentID,
                pcm16: pcm,
                sampleRate: sampleRate,
                channels: channels,
                textCursor: textCursor,
                textEndCursor: textEndCursor
            ))

        case "audio.done":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let textCursor = try optionalNonNegativeInt(root["text_cursor"], "text_cursor")
            return .audioDone(responseID: responseID, segmentID: segmentID, textCursor: textCursor)

        case "response.completed":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let textCursor = try optionalNonNegativeInt(root["text_cursor"], "text_cursor")
            return .responseCompleted(responseID: responseID, segmentID: segmentID, textCursor: textCursor)

        case "response.playback_ack":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let textCursor = try requireNonNegativeInt(root["text_cursor"], "text_cursor")
            let isFinal = try optionalBool(root["final"], "final") ?? false
            return .responsePlaybackAck(
                responseID: responseID, segmentID: segmentID,
                textCursor: textCursor, isFinal: isFinal
            )

        case "response.cancelled":
            let responseID = try requireString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let reason = try optionalString(root["reason"], "reason") ?? "cancelled"
            let textCursor = try optionalNonNegativeInt(root["text_cursor"], "text_cursor")
            let ackText = try optionalString(root["acknowledged_text"], "acknowledged_text")
            let historyRecorded = try optionalBool(root["history_recorded"], "history_recorded")
            return .responseCancelled(RealtimeCancellation(
                responseID: responseID,
                segmentID: segmentID,
                reason: reason,
                textCursor: textCursor,
                acknowledgedText: ackText,
                historyRecorded: historyRecorded
            ))

        case "response.failed":
            let responseID = try optionalString(root["response_id"], "response_id")
            let segmentID = try optionalInt(root["segment_id"], "segment_id") ?? 0
            let message = try optionalString(root["error"], "error") ?? "spoken response failed"
            return .responseFailed(responseID: responseID, segmentID: segmentID, error: message)

        case "error":
            if let obj = root["error"] as? [String: Any] {
                let code = try optionalString(obj["code"], "error.code")
                let message = try optionalString(obj["message"], "error.message") ?? "realtime server error"
                return .error(code: code, message: message)
            }
            if let message = root["error"] as? String {
                return .error(code: nil, message: message)
            }
            return .error(code: nil, message: "realtime server error")

        default:
            return .unknown(type: type)
        }
    }

    private static func serialize(_ object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private static func requireString(_ value: Any?, _ name: String) throws -> String {
        guard let text = value as? String else {
            throw RealtimeClientError.invalidEvent("\(name) must be a string")
        }
        return text
    }

    private static func optionalString(_ value: Any?, _ name: String) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        return try requireString(value, name)
    }

    private static func requireInt(_ value: Any?, _ name: String) throws -> Int {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number) else {
            throw RealtimeClientError.invalidEvent("\(name) must be an integer")
        }
        return number.intValue
    }

    private static func optionalInt(_ value: Any?, _ name: String) throws -> Int? {
        guard let value, !(value is NSNull) else { return nil }
        return try requireInt(value, name)
    }

    private static func requireNonNegativeInt(_ value: Any?, _ name: String) throws -> Int {
        let parsed = try requireInt(value, name)
        guard parsed >= 0 else {
            throw RealtimeClientError.invalidEvent("\(name) must be non-negative")
        }
        return parsed
    }

    private static func optionalNonNegativeInt(_ value: Any?, _ name: String) throws -> Int? {
        guard let value, !(value is NSNull) else { return nil }
        return try requireNonNegativeInt(value, name)
    }

    private static func optionalBool(_ value: Any?, _ name: String) throws -> Bool? {
        guard let value, !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw RealtimeClientError.invalidEvent("\(name) must be a boolean")
        }
        return number.boolValue
    }
}

/// Actions emitted by `RealtimeStateMachine` for the host runtime to execute.
enum RealtimeAction: Equatable, Sendable {
    case forward(RealtimeServerEvent)
    case beginAudioStream(responseID: String)
    case enqueueAudio(RealtimeAudioDelta)
    case finishAudioStream(responseID: String)
    case stopAudioImmediately(responseID: String)
    case sendPlaybackAck(RealtimePlaybackAck)
    case dropStale(responseID: String)
}

/// Pure, deterministic Stage B `playback_ack` + barge-in state machine.
///
/// Key invariants (from `docs/full-duplex.md`):
/// - `response.text.delta` cursors are informational only and never authorize a `playback_ack`.
/// - Only server-issued cursors from `audio.delta` / `audio.done` for the active `response_id`
///   can be acknowledged, strictly monotonically, and ONLY AFTER the corresponding PCM chunks
///   have been consumed by playback.
/// - `audio.done` / `response.completed` indicate generation finished, not playback finished;
///   the final cursor is acknowledged only once all queued PCM chunks finish playing.
/// - `response.cancelled` immediately stops playback, records the response ID as cancelled,
///   blocks any late playback callbacks from emitting acks, and drops all subsequent stale
///   frames matching that `response_id`.
struct RealtimeStateMachine: Equatable, Sendable {
    private(set) var sessionReady = false
    private(set) var playbackAckEnabled = false
    private(set) var activeResponseID: String?
    private(set) var audioStreamStarted = false
    private(set) var issuedCursors: Set<Int> = []
    private(set) var highestAckedCursor: Int?
    private(set) var pendingAudioChunks = 0
    private(set) var audioDoneReceived = false
    private(set) var audioDoneCursor: Int?
    private(set) var finalAckEmitted = false
    private(set) var cancelledResponseIDs: Set<String> = []
    private(set) var completedResponseIDs: Set<String> = []

    func isStaleResponse(_ responseID: String) -> Bool {
        cancelledResponseIDs.contains(responseID) || completedResponseIDs.contains(responseID)
    }

    mutating func receive(_ event: RealtimeServerEvent) -> [RealtimeAction] {
        switch event {
        case .sessionUpdated(let snapshot):
            sessionReady = true
            playbackAckEnabled = snapshot.playbackAck
            return [.forward(event)]

        case .speechStarted(_, let interruptedResponseID):
            var actions: [RealtimeAction] = []
            if let interrupted = interruptedResponseID,
               !interrupted.isEmpty,
               activeResponseID == interrupted,
               !cancelledResponseIDs.contains(interrupted) {
                actions.append(.stopAudioImmediately(responseID: interrupted))
            }
            actions.append(.forward(event))
            return actions

        case .speechStopped, .transcriptionSegment, .transcriptionCompleted, .error, .unknown:
            return [.forward(event)]

        case .responseTextDelta(let responseID, _, _, _):
            guard !isStaleResponse(responseID) else {
                return [.dropStale(responseID: responseID)]
            }
            activateResponseIfNeeded(responseID)
            // Deliberately do NOT insert `textCursor` into `issuedCursors`:
            // text emission alone never authorizes a playback acknowledgement.
            return [.forward(event)]

        case .audioDelta(let delta):
            guard !isStaleResponse(delta.responseID) else {
                return [.dropStale(responseID: delta.responseID)]
            }
            activateResponseIfNeeded(delta.responseID)
            if let cursor = delta.textCursor, cursor >= 0 {
                issuedCursors.insert(cursor)
            }
            pendingAudioChunks += 1
            var actions: [RealtimeAction] = []
            if !audioStreamStarted {
                audioStreamStarted = true
                actions.append(.beginAudioStream(responseID: delta.responseID))
            }
            actions.append(.enqueueAudio(delta))
            actions.append(.forward(event))
            return actions

        case .audioDone(let responseID, _, let textCursor):
            guard !isStaleResponse(responseID) else {
                return [.dropStale(responseID: responseID)]
            }
            activateResponseIfNeeded(responseID)
            audioDoneReceived = true
            if let cursor = textCursor, cursor >= 0 {
                issuedCursors.insert(cursor)
                audioDoneCursor = cursor
            }
            var actions: [RealtimeAction] = []
            if audioStreamStarted {
                actions.append(.finishAudioStream(responseID: responseID))
            }
            actions.append(.forward(event))
            actions.append(contentsOf: emitFinalAckIfDrained())
            return actions

        case .responseCompleted(let responseID, _, _):
            guard !isStaleResponse(responseID) else {
                return [.dropStale(responseID: responseID)]
            }
            if !playbackAckEnabled {
                completedResponseIDs.insert(responseID)
                if activeResponseID == responseID {
                    clearActiveResponse()
                }
            }
            return [.forward(event)]

        case .responsePlaybackAck(let responseID, _, _, let isFinal):
            guard !cancelledResponseIDs.contains(responseID) else {
                return [.dropStale(responseID: responseID)]
            }
            if isFinal {
                completedResponseIDs.insert(responseID)
                if activeResponseID == responseID {
                    clearActiveResponse()
                }
            }
            return [.forward(event)]

        case .responseCancelled(let cancellation):
            cancelledResponseIDs.insert(cancellation.responseID)
            if activeResponseID == cancellation.responseID {
                clearActiveResponse()
            }
            return [
                .stopAudioImmediately(responseID: cancellation.responseID),
                .forward(event),
            ]

        case .responseFailed(let responseID, _, _):
            if let id = responseID, isStaleResponse(id) {
                return [.dropStale(responseID: id)]
            }
            var actions: [RealtimeAction] = []
            if let id = responseID {
                completedResponseIDs.insert(id)
                if activeResponseID == id {
                    actions.append(.stopAudioImmediately(responseID: id))
                    clearActiveResponse()
                }
            } else if let active = activeResponseID {
                completedResponseIDs.insert(active)
                actions.append(.stopAudioImmediately(responseID: active))
                clearActiveResponse()
            }
            actions.append(.forward(event))
            return actions
        }
    }

    /// Called ONLY after a scheduled PCM chunk has been consumed by playback.
    mutating func audioChunkConsumed(responseID: String, textCursor: Int?) -> [RealtimeAction] {
        guard activeResponseID == responseID, !isStaleResponse(responseID) else {
            return []
        }
        pendingAudioChunks = max(0, pendingAudioChunks - 1)
        guard playbackAckEnabled else { return [] }

        var actions: [RealtimeAction] = []
        if let cursor = textCursor, issuedCursors.contains(cursor) {
            let advances: Bool
            if let highest = highestAckedCursor {
                advances = cursor > highest
            } else {
                advances = true
            }
            if advances {
                highestAckedCursor = cursor
                actions.append(.sendPlaybackAck(RealtimePlaybackAck(
                    responseID: responseID, textCursor: cursor
                )))
            }
        }
        actions.append(contentsOf: emitFinalAckIfDrained())
        return actions
    }

    private mutating func activateResponseIfNeeded(_ responseID: String) {
        guard activeResponseID != responseID else { return }
        activeResponseID = responseID
        audioStreamStarted = false
        issuedCursors = [0]
        highestAckedCursor = nil
        pendingAudioChunks = 0
        audioDoneReceived = false
        audioDoneCursor = nil
        finalAckEmitted = false
    }

    private mutating func clearActiveResponse() {
        activeResponseID = nil
        audioStreamStarted = false
        issuedCursors = []
        highestAckedCursor = nil
        pendingAudioChunks = 0
        audioDoneReceived = false
        audioDoneCursor = nil
        finalAckEmitted = false
    }

    private mutating func emitFinalAckIfDrained() -> [RealtimeAction] {
        guard playbackAckEnabled,
              let responseID = activeResponseID,
              audioDoneReceived,
              pendingAudioChunks == 0,
              !finalAckEmitted,
              let finalCursor = audioDoneCursor,
              issuedCursors.contains(finalCursor) else {
            return []
        }
        if let highest = highestAckedCursor, finalCursor < highest {
            return []
        }
        highestAckedCursor = finalCursor
        finalAckEmitted = true
        return [.sendPlaybackAck(RealtimePlaybackAck(
            responseID: responseID, textCursor: finalCursor
        ))]
    }
}

// MARK: - Transport & Audio Capture Abstractions

protocol RealtimeTransport: AnyObject, Sendable {
    func connect(url: URL) async throws
    func send(text: String) async throws
    func receive() -> AsyncThrowingStream<String, Error>
    func close() async
}

protocol RealtimeAudioCapturing: AnyObject, Sendable {
    func startCapture() -> AsyncThrowingStream<Data, Error>
    func stopCapture()
}

final class URLSessionRealtimeTransport: RealtimeTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var session: URLSession?
    private var task: URLSessionWebSocketTask?
    private var closed = false

    func connect(url: URL) async throws {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 0
        config.connectionProxyDictionary = [:]
        let urlSession = URLSession(configuration: config)
        let wsTask = urlSession.webSocketTask(with: url)
        wsTask.maximumMessageSize = 4 * 1024 * 1024

        lock.withLock {
            self.session = urlSession
            self.task = wsTask
            self.closed = false
        }

        wsTask.resume()
    }

    func send(text: String) async throws {
        let wsTask: URLSessionWebSocketTask? = lock.withLock { task }
        guard let wsTask else { throw RealtimeClientError.notConnected }
        try await wsTask.send(.string(text))
    }

    func receive() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let reader = Task { [weak self] in
                while !Task.isCancelled {
                    guard let self else { break }
                    let wsTask: URLSessionWebSocketTask? = self.lock.withLock {
                        self.closed ? nil : self.task
                    }
                    guard let wsTask else { break }
                    do {
                        let message = try await wsTask.receive()
                        switch message {
                        case .string(let text):
                            continuation.yield(text)
                        case .data(let data):
                            continuation.yield(String(decoding: data, as: UTF8.self))
                        @unknown default:
                            break
                        }
                    } catch {
                        let wasClosed = self.lock.withLock { self.closed }
                        if wasClosed || Task.isCancelled {
                            continuation.finish()
                        } else {
                            continuation.finish(throwing: error)
                        }
                        return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                reader.cancel()
            }
        }
    }

    func close() async {
        let (wsTask, urlSession): (URLSessionWebSocketTask?, URLSession?) = lock.withLock {
            closed = true
            let t = task
            let s = session
            task = nil
            session = nil
            return (t, s)
        }
        wsTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
    }
}

/// Captures microphone audio and resamples to 16 kHz mono PCM16 in 100 ms (3200-byte) chunks.
final class LiveRealtimeAudioCapturer: RealtimeAudioCapturing, @unchecked Sendable {
    private let lock = NSLock()
    private var engine: AVAudioEngine?
    private var continuation: AsyncThrowingStream<Data, Error>.Continuation?
    private var setupTask: Task<Void, Never>?
    private var generation = 0

    /// 100 ms at 16 kHz mono Int16 = 1600 samples = 3200 bytes.
    static let targetSampleRate: Double = 16_000
    static let chunkByteCount = 3_200

    private final class TapState: @unchecked Sendable {
        let converter: AVAudioConverter
        let targetFormat: AVAudioFormat
        var pending = Data()
        let onChunk: (Data) -> Void

        init(converter: AVAudioConverter, targetFormat: AVAudioFormat, onChunk: @escaping (Data) -> Void) {
            self.converter = converter
            self.targetFormat = targetFormat
            self.onChunk = onChunk
        }

        private final class OneShotInputBox: @unchecked Sendable {
            private var buffer: AVAudioPCMBuffer?
            init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
            func take() -> AVAudioPCMBuffer? {
                let value = buffer
                buffer = nil
                return value
            }
        }

        func process(_ buffer: AVAudioPCMBuffer) {
            let ratio = targetFormat.sampleRate / buffer.format.sampleRate
            let capacity = max(AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64, 512)
            guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
                return
            }
            var error: NSError?
            let inputBox = OneShotInputBox(buffer)
            let status = converter.convert(to: converted, error: &error) { _, outStatus in
                guard let next = inputBox.take() else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                outStatus.pointee = .haveData
                return next
            }
            guard status != .error, error == nil, converted.frameLength > 0,
                  let int16Data = converted.int16ChannelData else {
                return
            }
            let byteCount = Int(converted.frameLength) * 2
            pending.append(UnsafeBufferPointer(
                start: UnsafeRawPointer(int16Data[0]).assumingMemoryBound(to: UInt8.self),
                count: byteCount
            ))
            while pending.count >= LiveRealtimeAudioCapturer.chunkByteCount {
                let slice = Data(pending.prefix(LiveRealtimeAudioCapturer.chunkByteCount))
                pending.removeFirst(LiveRealtimeAudioCapturer.chunkByteCount)
                onChunk(slice)
            }
        }
    }

    func startCapture() -> AsyncThrowingStream<Data, Error> {
        stopCapture()
        let gen: Int = lock.withLock { generation }
        var capturedContinuation: AsyncThrowingStream<Data, Error>.Continuation?
        let stream = AsyncThrowingStream<Data, Error> { continuation in
            capturedContinuation = continuation
            continuation.onTermination = { [weak self] _ in
                self?.stopCapture()
            }
        }
        lock.withLock {
            self.continuation = capturedContinuation
        }
        let setup = Task { [weak self] in
            guard let self else { return }
            let granted = await AVAudioApplication.requestRecordPermission()
            guard self.isCurrent(gen) else { return }
            guard granted else {
                self.fail(RealtimeClientError.microphoneDenied, generation: gen)
                return
            }
            do {
                let audioEngine = AVAudioEngine()
                let input = audioEngine.inputNode
                let inputFormat = input.outputFormat(forBus: 0)
                guard inputFormat.channelCount > 0, inputFormat.sampleRate > 0 else {
                    throw RealtimeClientError.audioCaptureFailed("no audio input device available")
                }
                guard let targetFormat = AVAudioFormat(
                    commonFormat: .pcmFormatInt16,
                    sampleRate: Self.targetSampleRate,
                    channels: 1,
                    interleaved: true
                ), let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                    throw RealtimeClientError.audioCaptureFailed("unable to configure 16 kHz PCM16 converter")
                }
                let tapState = TapState(converter: converter, targetFormat: targetFormat) { [weak self] chunk in
                    self?.yield(chunk, generation: gen)
                }
                input.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { buffer, _ in
                    tapState.process(buffer)
                }
                audioEngine.prepare()
                do {
                    try audioEngine.start()
                } catch {
                    input.removeTap(onBus: 0)
                    throw RealtimeClientError.audioCaptureFailed(error.localizedDescription)
                }
                let adopted: Bool = self.lock.withLock {
                    guard self.generation == gen else { return false }
                    self.engine = audioEngine
                    return true
                }
                if !adopted {
                    audioEngine.stop()
                    input.removeTap(onBus: 0)
                }
            } catch {
                self.fail(error, generation: gen)
            }
        }
        lock.withLock { self.setupTask = setup }
        return stream
    }

    func stopCapture() {
        let (oldSetup, oldEngine, oldContinuation): (
            Task<Void, Never>?, AVAudioEngine?, AsyncThrowingStream<Data, Error>.Continuation?
        ) = lock.withLock {
            generation += 1
            let s = setupTask
            let e = engine
            let c = continuation
            setupTask = nil
            engine = nil
            continuation = nil
            return (s, e, c)
        }
        oldSetup?.cancel()
        if let oldEngine {
            oldEngine.stop()
            oldEngine.inputNode.removeTap(onBus: 0)
        }
        oldContinuation?.finish()
    }

    private func isCurrent(_ gen: Int) -> Bool {
        lock.withLock { generation == gen }
    }

    private func yield(_ data: Data, generation gen: Int) {
        let cont: AsyncThrowingStream<Data, Error>.Continuation? = lock.withLock {
            generation == gen ? continuation : nil
        }
        cont?.yield(data)
    }

    private func fail(_ error: Error, generation gen: Int) {
        let (oldEngine, oldContinuation): (
            AVAudioEngine?, AsyncThrowingStream<Data, Error>.Continuation?
        ) = lock.withLock {
            guard generation == gen else { return (nil, nil) }
            let e = engine
            let c = continuation
            engine = nil
            continuation = nil
            return (e, c)
        }
        if let oldEngine {
            oldEngine.stop()
            oldEngine.inputNode.removeTap(onBus: 0)
        }
        oldContinuation?.finish(throwing: error)
    }
}

// MARK: - RealtimeClient

@MainActor
final class RealtimeClient {
    private let transportFactory: @Sendable () -> RealtimeTransport
    private let audioCapturer: RealtimeAudioCapturing

    private var transport: RealtimeTransport?
    private var receiveTask: Task<Void, Never>?
    private var captureTask: Task<Void, Never>?
    private weak var audioStream: AudioStreamPlaying?
    private var sessionToken = 0
    private var captureStarted = false

    private(set) var stateMachine = RealtimeStateMachine()
    private(set) var droppedStaleEventCount = 0
    private(set) var isActive = false

    private var onEventHandler: ((RealtimeServerEvent) -> Void)?
    private var onErrorHandler: ((Error) -> Void)?
    private var onDisconnectHandler: (() -> Void)?

    init(
        transportFactory: @escaping @Sendable () -> RealtimeTransport = { URLSessionRealtimeTransport() },
        audioCapturer: RealtimeAudioCapturing = LiveRealtimeAudioCapturer()
    ) {
        self.transportFactory = transportFactory
        self.audioCapturer = audioCapturer
    }

    /// Converts an HTTP/HTTPS base URL to `ws://` / `wss://`
    /// `<base>/realtime?intent=transcription&model=<model>`, preserving the exact
    /// loopback/private/ULA/allowlist URL security policy from `ChatClient`.
    static func realtimeURL(
        base: String,
        model: String,
        allowedHTTPHosts: Set<String> = ChatClient.plainHTTPHostAllowlist()
    ) throws -> URL {
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModel.isEmpty else {
            throw ChatClientError.modelRequired
        }
        let text = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              let host = components.host?.lowercased(), !host.isEmpty,
              scheme == "http" || scheme == "https" || scheme == "ws" || scheme == "wss",
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else {
            throw ChatClientError.invalidEndpoint(
                "Use an HTTP(S) base URL without credentials, query, or fragment.")
        }
        let isPlaintext = scheme == "http" || scheme == "ws"
        if isPlaintext && !ChatClient.isPlainHTTPHost(host, allowed: allowedHTTPHosts) {
            throw ChatClientError.invalidEndpoint(
                "Plain WS/HTTP is allowed only on loopback, a private address, "
                + "or a host named in UMMMLX_HTTP_HOSTS. Use HTTPS/WSS for other hosts.")
        }
        var mutable = components
        mutable.scheme = isPlaintext ? "ws" : "wss"
        var path = mutable.path
        while path.count > 1 && path.last == "/" { path.removeLast() }
        if path.isEmpty || path == "/" { path = "/v1" }
        mutable.path = path + "/realtime"
        mutable.queryItems = [
            URLQueryItem(name: "intent", value: "transcription"),
            URLQueryItem(name: "model", value: trimmedModel),
        ]
        guard let url = mutable.url else {
            throw ChatClientError.invalidEndpoint("The base URL is invalid.")
        }
        return url
    }

    func start(
        base: String,
        model: String,
        config: RealtimeSessionConfig = RealtimeSessionConfig(),
        audioStream: AudioStreamPlaying,
        allowedHTTPHosts: Set<String> = ChatClient.plainHTTPHostAllowlist(),
        onEvent: @escaping (RealtimeServerEvent) -> Void,
        onError: @escaping (Error) -> Void,
        onDisconnect: @escaping () -> Void = {}
    ) throws {
        let url = try Self.realtimeURL(base: base, model: model, allowedHTTPHosts: allowedHTTPHosts)
        let sessionUpdateFrame = try RealtimeWire.encodeSessionUpdate(config)

        stop()
        sessionToken += 1
        let token = sessionToken

        let newTransport = transportFactory()
        self.transport = newTransport
        self.audioStream = audioStream
        self.stateMachine = RealtimeStateMachine()
        self.droppedStaleEventCount = 0
        self.captureStarted = false
        self.isActive = true
        self.onEventHandler = onEvent
        self.onErrorHandler = onError
        self.onDisconnectHandler = onDisconnect

        receiveTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await newTransport.connect(url: url)
                guard self.sessionToken == token else {
                    await newTransport.close()
                    return
                }
                let incoming = newTransport.receive()
                try await newTransport.send(text: sessionUpdateFrame)
                guard self.sessionToken == token else { return }

                for try await raw in incoming {
                    guard self.sessionToken == token else { return }
                    let event = try RealtimeWire.decodeServerEvent(raw)
                    self.handleServerEvent(event, token: token)
                }
                guard self.sessionToken == token else { return }
                self.teardown(token: token, error: nil)
            } catch {
                guard self.sessionToken == token else { return }
                if error is CancellationError {
                    self.teardown(token: token, error: nil)
                } else {
                    self.teardown(token: token, error: error)
                }
            }
        }
    }

    /// Sends one raw 16 kHz mono PCM16 chunk over the active realtime WebSocket.
    func sendAudioChunk(_ pcm16: Data) async throws {
        guard isActive, let transport else {
            throw RealtimeClientError.notConnected
        }
        let frame = try RealtimeWire.encodeAudioAppend(pcm16)
        try await transport.send(text: frame)
    }

    func stop() {
        sessionToken += 1
        let oldTransport = transport
        transport = nil
        receiveTask?.cancel()
        receiveTask = nil
        captureTask?.cancel()
        captureTask = nil
        if captureStarted {
            audioCapturer.stopCapture()
        }
        audioStream?.stop()
        audioStream = nil
        captureStarted = false
        isActive = false
        onEventHandler = nil
        onErrorHandler = nil
        onDisconnectHandler = nil
        if let oldTransport {
            Task { await oldTransport.close() }
        }
    }

    private func handleServerEvent(_ event: RealtimeServerEvent, token: Int) {
        guard sessionToken == token else { return }
        let actions = stateMachine.receive(event)
        execute(actions, token: token)
        if case .sessionUpdated = event, !captureStarted {
            startMicrophoneStreaming(token: token)
        }
    }

    private func handleAudioChunkConsumed(responseID: String, textCursor: Int?, token: Int) {
        guard sessionToken == token, isActive else { return }
        let actions = stateMachine.audioChunkConsumed(responseID: responseID, textCursor: textCursor)
        execute(actions, token: token)
    }

    private func execute(_ actions: [RealtimeAction], token: Int) {
        guard sessionToken == token else { return }
        for action in actions {
            switch action {
            case .beginAudioStream:
                audioStream?.beginStream()
            case .enqueueAudio(let delta):
                let responseID = delta.responseID
                let cursor = delta.textCursor
                audioStream?.enqueue(delta.chunk) { [weak self] in
                    self?.handleAudioChunkConsumed(
                        responseID: responseID, textCursor: cursor, token: token
                    )
                }
            case .finishAudioStream:
                audioStream?.finishStream()
            case .stopAudioImmediately:
                audioStream?.stop()
            case .sendPlaybackAck(let ack):
                sendPlaybackAck(ack, token: token)
            case .forward(let event):
                onEventHandler?(event)
            case .dropStale:
                droppedStaleEventCount += 1
            }
        }
    }

    private func sendPlaybackAck(_ ack: RealtimePlaybackAck, token: Int) {
        guard sessionToken == token, let transport else { return }
        guard let frame = try? RealtimeWire.encodePlaybackAck(
            responseID: ack.responseID, textCursor: ack.textCursor
        ) else {
            return
        }
        Task { @MainActor [weak self] in
            guard let self, self.sessionToken == token else { return }
            do {
                try await transport.send(text: frame)
            } catch {
                guard self.sessionToken == token else { return }
                self.teardown(token: token, error: error)
            }
        }
    }

    private func startMicrophoneStreaming(token: Int) {
        guard sessionToken == token, !captureStarted else { return }
        captureStarted = true
        let stream = audioCapturer.startCapture()
        captureTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                for try await pcm16 in stream {
                    guard self.sessionToken == token, let transport = self.transport else { return }
                    let frame = try RealtimeWire.encodeAudioAppend(pcm16)
                    try await transport.send(text: frame)
                }
            } catch {
                guard self.sessionToken == token else { return }
                if !(error is CancellationError) {
                    self.teardown(token: token, error: error)
                }
            }
        }
    }

    private func teardown(token: Int, error: Error?) {
        guard sessionToken == token else { return }
        let errorHandler = onErrorHandler
        let disconnectHandler = onDisconnectHandler
        stop()
        if let error {
            errorHandler?(error)
        } else {
            disconnectHandler?()
        }
    }
}
