// SPDX-License-Identifier: Apache-2.0
// Local speech input. Apple's on-device model only: no audio, transcript
// request or partial result leaves this Mac or reaches an umm server.

import AVFoundation
import Foundation
import Speech

enum SpeechUpdate: Equatable, Sendable {
    case partial(String)
    case final(String)
}

enum SpeechError: LocalizedError, Equatable {
    case authorizationDenied
    case microphoneDenied
    case recognizerUnavailable
    case onDeviceUnavailable
    case audioEngine(String)
    case interrupted
    case emptyTranscript

    var errorDescription: String? {
        switch self {
        case .authorizationDenied:
            return "Speech recognition is not authorized. Allow ummmlx in System Settings > Privacy & Security > Speech Recognition."
        case .microphoneDenied:
            return "Microphone access is not authorized. Allow ummmlx in System Settings > Privacy & Security > Microphone."
        case .recognizerUnavailable:
            return "The on-device recognizer is not available for this language."
        case .onDeviceUnavailable:
            return "This Mac has no on-device speech model for the selected language."
        case .audioEngine(let detail):
            return "The audio engine failed: \(detail)"
        case .interrupted:
            return "The speech recognizer stopped before it finished."
        case .emptyTranscript:
            return "No speech was recognized."
        }
    }
}

/// Both recognition paths. `requiresOnDeviceRecognition` is set on every
/// request the Apple implementation creates.
protocol SpeechTranscribing: AnyObject {
    func transcribeMicrophone() -> AsyncThrowingStream<SpeechUpdate, Error>
    func transcribeFile(at url: URL) -> AsyncThrowingStream<SpeechUpdate, Error>
    func cancel()
}

/// Thread-safe holder for a recognition request captured by an audio tap.
private final class RequestBox: @unchecked Sendable {
    let request: SFSpeechAudioBufferRecognitionRequest
    init(_ request: SFSpeechAudioBufferRecognitionRequest) { self.request = request }
}

final class AppleSpeechTranscriber: NSObject, SpeechTranscribing, @unchecked Sendable {
    private let locale: Locale
    private let lock = NSLock()
    private var continuation: AsyncThrowingStream<SpeechUpdate, Error>.Continuation?
    private var engine: AVAudioEngine?
    private var task: SFSpeechRecognitionTask?
    private var setup: Task<Void, Never>?
    /// Bumped by every cancel and every new stream. A setup that resumes after
    /// a cancel sees a stale value and tears itself down instead of installing
    /// an engine and tap that no reference can stop.
    private var generation = 0

    init(locale: Locale = Locale(identifier: "en-US")) {
        self.locale = locale
        super.init()
    }

    func transcribeMicrophone() -> AsyncThrowingStream<SpeechUpdate, Error> {
        let (stream, generation) = makeStream()
        let setup = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.prepare(requireMicrophone: true)
                try self.requireCurrent(generation)
                let recognizer = try self.makeRecognizer()
                let request = SFSpeechAudioBufferRecognitionRequest()
                request.requiresOnDeviceRecognition = true
                request.shouldReportPartialResults = true
                let engine = AVAudioEngine()
                let input = engine.inputNode
                let format = input.outputFormat(forBus: 0)
                guard format.channelCount > 0 else {
                    throw SpeechError.audioEngine("no audio input device")
                }
                let box = RequestBox(request)
                input.installTap(onBus: 0, bufferSize: 4096, format: format) {
                    buffer, _ in
                    box.request.append(buffer)
                }
                engine.prepare()
                do {
                    try engine.start()
                } catch {
                    input.removeTap(onBus: 0)
                    throw SpeechError.audioEngine(error.localizedDescription)
                }
                guard self.adopt(engine: engine, generation: generation) else {
                    engine.stop()
                    input.removeTap(onBus: 0)
                    return
                }
                self.start(recognizer: recognizer, request: request,
                           generation: generation)
            } catch {
                self.fail(error, generation: generation)
            }
        }
        lock.lock()
        self.setup = setup
        lock.unlock()
        return stream
    }

    func transcribeFile(at url: URL) -> AsyncThrowingStream<SpeechUpdate, Error> {
        let (stream, generation) = makeStream()
        let setup = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.prepare(requireMicrophone: false)
                try self.requireCurrent(generation)
                let recognizer = try self.makeRecognizer()
                let request = SFSpeechURLRecognitionRequest(url: url)
                request.requiresOnDeviceRecognition = true
                request.shouldReportPartialResults = true
                self.start(recognizer: recognizer, request: request,
                           generation: generation)
            } catch {
                self.fail(error, generation: generation)
            }
        }
        lock.lock()
        self.setup = setup
        lock.unlock()
        return stream
    }

    func cancel() {
        lock.lock()
        generation += 1
        let continuation = self.continuation
        self.continuation = nil
        let engine = self.engine
        self.engine = nil
        let task = self.task
        self.task = nil
        let setup = self.setup
        self.setup = nil
        lock.unlock()
        setup?.cancel()
        task?.cancel()
        if let engine {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
        continuation?.finish()
    }

    private func makeStream() -> (AsyncThrowingStream<SpeechUpdate, Error>, Int) {
        cancel()
        lock.lock()
        let value = generation
        lock.unlock()
        let stream = AsyncThrowingStream<SpeechUpdate, Error> { continuation in
            self.lock.lock()
            self.continuation = continuation
            self.lock.unlock()
            continuation.onTermination = { [weak self] _ in
                self?.cancel()
            }
        }
        return (stream, value)
    }

    private func isCurrent(_ value: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return generation == value
    }

    private func requireCurrent(_ value: Int) throws {
        guard isCurrent(value) else { throw CancellationError() }
    }

    private func prepare(requireMicrophone: Bool) async throws {
        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard status == .authorized else { throw SpeechError.authorizationDenied }
        if requireMicrophone {
            let granted = await AVAudioApplication.requestRecordPermission()
            guard granted else { throw SpeechError.microphoneDenied }
        }
    }

    private func makeRecognizer() throws -> SFSpeechRecognizer {
        guard let recognizer = SFSpeechRecognizer(locale: locale),
              recognizer.isAvailable else {
            throw SpeechError.recognizerUnavailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw SpeechError.onDeviceUnavailable
        }
        return recognizer
    }

    /// Stores the engine only when this setup is still the current one.
    private func adopt(engine: AVAudioEngine, generation value: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard generation == value else { return false }
        self.engine = engine
        return true
    }

    private func start(recognizer: SFSpeechRecognizer,
                       request: SFSpeechRecognitionRequest,
                       generation value: Int) {
        let task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let text = result.bestTranscription.formattedString
                if result.isFinal {
                    self.settle(with: text, generation: value)
                } else if !text.isEmpty {
                    self.emit(.partial(text), generation: value)
                }
            }
            if let error {
                // A stale generation means cancel() ran; it already finished
                // the stream and stopped the engine.
                guard self.isCurrent(value) else { return }
                let ns = error as NSError
                // 216: the recognition task was cancelled. When this side
                // cancelled, the guard above returns first.
                self.fail(ns.code == 216 ? SpeechError.interrupted : error,
                          generation: value)
            }
        }
        lock.lock()
        let current = generation == value
        if current { self.task = task }
        lock.unlock()
        if !current { task.cancel() }
    }

    private func emit(_ update: SpeechUpdate, generation value: Int) {
        lock.lock()
        let continuation = generation == value ? self.continuation : nil
        lock.unlock()
        continuation?.yield(update)
    }

    private func settle(with text: String, generation value: Int) {
        lock.lock()
        guard generation == value else {
            lock.unlock()
            return
        }
        let continuation = self.continuation
        self.continuation = nil
        let engine = self.engine
        self.engine = nil
        let task = self.task
        self.task = nil
        lock.unlock()
        continuation?.yield(.final(text))
        continuation?.finish()
        task?.cancel()
        if let engine {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
    }

    private func fail(_ error: Error, generation value: Int) {
        lock.lock()
        guard generation == value else {
            lock.unlock()
            return
        }
        let continuation = self.continuation
        self.continuation = nil
        let engine = self.engine
        self.engine = nil
        let task = self.task
        self.task = nil
        lock.unlock()
        task?.cancel()
        if let engine {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
        continuation?.finish(throwing: error)
    }
}

@MainActor
final class SpeechController: ObservableObject {
    enum Phase: Equatable {
        case idle
        case listening
        case transcribing
        case finished
        case failed
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var partial = ""
    @Published private(set) var finalText = ""
    @Published private(set) var error: String?

    var onPartial: ((String) -> Void)?
    var onFinal: ((String) -> Void)?
    var onFailure: ((String) -> Void)?
    /// A run ended without a final result (silence, or the system ending the
    /// task). The UI releases the microphone; this is not an error.
    var onIdle: (() -> Void)?
    /// True when the active recognizer is Apple's on-device model.
    private(set) var onDevice = true

    private let transcriber: SpeechTranscribing
    private var task: Task<Void, Never>?
    /// Identifies the live runner. A cancelled runner that finishes later must
    /// not clear the task that replaced it.
    private var runID = 0

    var isRunning: Bool { task != nil }

    init(transcriber: SpeechTranscribing = AppleSpeechTranscriber()) {
        self.transcriber = transcriber
    }

    func startMicrophone() {
        run(stream: transcriber.transcribeMicrophone(), listening: true)
    }

    func startFile(_ url: URL) {
        run(stream: transcriber.transcribeFile(at: url), listening: false)
    }

    /// Blocking file recognition for automation. Returns the final transcript.
    func transcribeFile(_ url: URL) async throws -> String {
        cancel()
        partial = ""
        finalText = ""
        error = nil
        phase = .transcribing
        do {
            var final = ""
            for try await update in transcriber.transcribeFile(at: url) {
                switch update {
                case .partial(let text):
                    partial = text
                    onPartial?(text)
                case .final(let text):
                    final = text
                    partial = text
                }
            }
            guard !final.isEmpty else { throw SpeechError.emptyTranscript }
            finalText = final
            phase = .finished
            onFinal?(final)
            return final
        } catch {
            self.error = error.localizedDescription
            phase = .failed
            throw error
        }
    }

    func cancel() {
        transcriber.cancel()
        runID += 1
        task?.cancel()
        task = nil
        if phase == .listening || phase == .transcribing {
            phase = .idle
        }
        partial = ""
    }

    private func run(stream: AsyncThrowingStream<SpeechUpdate, Error>, listening: Bool) {
        guard task == nil else { return }
        partial = ""
        finalText = ""
        error = nil
        phase = listening ? .listening : .transcribing
        runID += 1
        let id = runID
        task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                for try await update in stream {
                    // A cancelled run must not write into the session that
                    // replaced it.
                    guard self.runID == id else { return }
                    switch update {
                    case .partial(let text):
                        self.partial = text
                        self.onPartial?(text)
                    case .final(let text):
                        self.finalText = text
                        self.partial = text
                        self.phase = .finished
                        self.onFinal?(text)
                    }
                }
                if self.runID == id,
                   self.phase == .listening || self.phase == .transcribing {
                    self.phase = .finished
                    self.onIdle?()
                }
            } catch {
                if self.runID == id, !(error is CancellationError) {
                    self.error = error.localizedDescription
                    self.phase = .failed
                    self.onFailure?(error.localizedDescription)
                }
            }
            if self.runID == id { self.task = nil }
        }
    }
}
