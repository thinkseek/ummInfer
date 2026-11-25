// SPDX-License-Identifier: Apache-2.0
// Incremental PCM16 audio playback for streamed TTS responses. Schedules
// verified chunks onto AVAudioPlayerNode as they arrive and drains the tail
// naturally after the SSE stream finishes unless explicitly stopped.

import AVFoundation
import Foundation

@MainActor
protocol AudioStreamPlaying: AnyObject {
    var isPlaying: Bool { get }
    var onStateChange: ((Bool) -> Void)? { get set }
    func beginStream()
    func enqueue(_ chunk: AudioStreamChunk)
    func enqueue(_ chunk: AudioStreamChunk, onPlayed: (@MainActor @Sendable () -> Void)?)
    func finishStream()
    func stop()
}

extension AudioStreamPlaying {
    func enqueue(_ chunk: AudioStreamChunk, onPlayed: (@MainActor @Sendable () -> Void)?) {
        enqueue(chunk)
        onPlayed?()
    }
}

@MainActor
final class NoOpAudioStreamPlayer: AudioStreamPlaying {
    let isPlaying = false
    var onStateChange: ((Bool) -> Void)?
    func beginStream() {}
    func enqueue(_ chunk: AudioStreamChunk) {}
    func finishStream() {}
    func stop() {}
}

/// Jitter-buffer sizing for one streamed-PCM turn. The two extremes it must
/// cover: a burst delivery (all chunks at once, so latency stays low) and a
/// link slower than real time (gaps above 2 s, so the cushion grows to the cap).
/// One value per stream; a new stream does not inherit the previous worst gap.
struct JitterPolicy {
    private(set) var targetSeconds: Double
    private let minimumSeconds: Double
    private let maximumSeconds: Double
    private var recentGaps: [Double] = []
    private var cleanChunks = 0

    init(minimumSeconds: Double = 0.25, maximumSeconds: Double = 3.0,
         initialSeconds: Double = 0.3) {
        self.minimumSeconds = minimumSeconds
        self.maximumSeconds = maximumSeconds
        self.targetSeconds = initialSeconds
    }

    /// `gap` is the wall-clock gap since the previous chunk, nil for the first
    /// chunk. `starved` means the node had run dry, so the current gap needs
    /// its own cushion on top of the worst gap seen.
    mutating func noteArrival(gap: Double?, starved: Bool) {
        if let gap, gap > 0.001 {
            recentGaps.append(gap)
            if recentGaps.count > 8 { recentGaps.removeFirst() }
            if starved {
                targetSeconds = min(maximumSeconds,
                                    max(targetSeconds, gap * 1.5 + 0.2))
            }
        }
        if let worst = recentGaps.max() {
            targetSeconds = min(maximumSeconds,
                                max(targetSeconds, worst * 1.5 + 0.1))
        }
    }

    mutating func notePlayedChunk() {
        cleanChunks += 1
        if cleanChunks >= 20 {
            cleanChunks = 0
            targetSeconds = max(minimumSeconds, targetSeconds * 0.85)
        }
    }
}

@MainActor
final class LiveAudioStreamPlayer: AudioStreamPlaying {
    var onStateChange: ((Bool) -> Void)?
    private(set) var isPlaying = false {
        didSet {
            if oldValue != isPlaying {
                onStateChange?(isPlaying)
            }
        }
    }

    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var activeRate = 0
    private var activeChannels = 0
    private var pendingBuffers = 0
    private var streamEnded = false
    private var engineFailed = false
    private var generation = 0
    /// Chunks arrive per sentence, and the first one can be 0.16 s long while
    /// the next takes 0.3 s. Starting playback on the first chunk therefore
    /// runs the node dry and the audio stutters. Wait for a small jitter
    /// buffer before the first start, and once playing keep scheduling.
    private var bufferedFrames = 0
    private var playbackStarted = false
    /// Adaptive jitter buffer. A slow link or a slow synthesiser shows up as
    /// long gaps between chunks, so the target is derived from the gaps this
    /// stream actually produced instead of a fixed constant: start low for
    /// latency, grow when the node is found starved, shrink again after a run
    /// of clean chunks.
    private var jitter = JitterPolicy()
    private var lastArrival: Double = 0
    private var underruns = 0
    /// When the producer is slower than real time the cushion can never grow,
    /// so waiting for a target only delays the first gap. In that case hold
    /// every chunk and play the answer once it is complete.
    private var firstArrival: Double = 0
    private var producedSeconds: Double = 0
    private var chunkCount = 0
    private var bufferToEnd = false

    private static let debugAudio =
        ProcessInfo.processInfo.environment["UMM_AUDIO_DEBUG"] == "1"
    private static func log(_ what: String) {
        guard debugAudio else { return }
        FileHandle.standardError.write(
            Data("AUDIO \(Date().timeIntervalSince1970) \(what)\n".utf8))
    }

    /// Read-only view of the active stream's cushion, used by the fault gate
    /// and the regression tests. It changes no playback behavior.
    var jitterTargetSeconds: Double { jitter.targetSeconds }
    /// True when the output device could not be started.
    var outputDeviceUnavailable: Bool { engineFailed }

    static func makePCMBuffer(_ chunk: AudioStreamChunk) -> AVAudioPCMBuffer? {
        guard chunk.sampleRate > 0,
              chunk.channels >= 1, chunk.channels <= 8 else {
            return nil
        }
        let frameBytes = chunk.channels * 2
        guard !chunk.pcm16.isEmpty, chunk.pcm16.count % frameBytes == 0 else {
            return nil
        }
        let frameCount = chunk.pcm16.count / frameBytes
        guard let format = AVAudioFormat(
                  standardFormatWithSampleRate: Double(chunk.sampleRate),
                  channels: AVAudioChannelCount(chunk.channels)),
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(frameCount)),
              let channelData = buffer.floatChannelData else {
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        chunk.pcm16.withUnsafeBytes { raw in
            for frame in 0..<frameCount {
                for ch in 0..<chunk.channels {
                    let offset = (frame * chunk.channels + ch) * 2
                    let lo = UInt16(raw[offset])
                    let hi = UInt16(raw[offset + 1]) << 8
                    let sample = Int16(bitPattern: lo | hi)
                    channelData[ch][frame] = max(-1.0, Float(sample) / 32768.0)
                }
            }
        }
        return buffer
    }

    func beginStream() {
        stop()
        engineFailed = false
        streamEnded = false
        bufferedFrames = 0
        playbackStarted = false
        firstArrival = 0
        producedSeconds = 0
        chunkCount = 0
        bufferToEnd = false
        // A stream that follows a slow one starts at the minimum cushion
        // instead of inheriting the previous stream's worst gap.
        jitter = JitterPolicy()
        lastArrival = 0
        underruns = 0
    }

    func enqueue(_ chunk: AudioStreamChunk) {
        enqueue(chunk, onPlayed: nil)
    }

    func enqueue(_ chunk: AudioStreamChunk, onPlayed: (@MainActor @Sendable () -> Void)?) {
        guard !engineFailed, let buffer = Self.makePCMBuffer(chunk) else { return }
        if engine == nil || activeRate != chunk.sampleRate || activeChannels != chunk.channels {
            teardownEngine()
            guard configureEngine(format: buffer.format,
                                  sampleRate: chunk.sampleRate,
                                  channels: chunk.channels) else {
                engineFailed = true
                return
            }
        }
        guard let engine, let node = playerNode else { return }
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                engineFailed = true
                stop()
                return
            }
        }
        let currentGen = generation
        let frames = Int(buffer.frameLength)
        let sampleRate = Double(chunk.sampleRate)
        let now = Date().timeIntervalSince1970
        // The node ran dry before this chunk arrived: playback stopped and
        // only resumes once the cushion is refilled.
        let gap = lastArrival > 0 ? now - lastArrival : nil
        let starved = playbackStarted && pendingBuffers == 0 && !streamEnded
        jitter.noteArrival(gap: gap, starved: starved)
        if starved {
            underruns += 1
            playbackStarted = false
            isPlaying = false
            Self.log("underrun gap_s=\(gap ?? 0) target_s=\(jitter.targetSeconds)")
        }
        lastArrival = now
        chunkCount += 1
        producedSeconds += Double(frames) / sampleRate
        if firstArrival == 0 { firstArrival = now }
        let elapsed = now - firstArrival
        if !bufferToEnd, chunkCount >= 4, elapsed > 0.5,
           producedSeconds / elapsed < 0.95 {
            bufferToEnd = true
            Self.log("slow_producer ratio=\(producedSeconds / elapsed)")
        }
        pendingBuffers += 1
        bufferedFrames += frames
        let needed = Int(jitter.targetSeconds * sampleRate)
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.didFinishBuffer(generation: currentGen, frames: frames, onPlayed: onPlayed)
            }
        }
        if !playbackStarted && !bufferToEnd && bufferedFrames >= needed {
            playbackStarted = true
            isPlaying = true
            Self.log("start target_s=\(jitter.targetSeconds) buffered_s=\(Double(bufferedFrames) / sampleRate)")
        }
        if playbackStarted && !node.isPlaying {
            node.play()
        }
        if playbackStarted && underruns == 0 {
            jitter.notePlayedChunk()
        }
        Self.log("enqueue frames=\(frames) buffered_ms=\(Int(Double(bufferedFrames) / sampleRate * 1000)) target_ms=\(Int(jitter.targetSeconds * 1000))")
    }

    func finishStream() {
        Self.log("finish_stream pending=\(pendingBuffers)")
        streamEnded = true
        if pendingBuffers == 0 {
            teardownEngine()
            isPlaying = false
        } else if !playbackStarted {
            // A short answer never reached the jitter target; play what we have.
            playbackStarted = true
            isPlaying = true
            playerNode?.play()
        }
    }

    func stop() {
        generation += 1
        pendingBuffers = 0
        bufferedFrames = 0
        playbackStarted = false
        streamEnded = true
        teardownEngine()
        isPlaying = false
    }

    private func configureEngine(format: AVAudioFormat,
                                 sampleRate: Int,
                                 channels: Int) -> Bool {
        let newEngine = AVAudioEngine()
        guard newEngine.outputNode.outputFormat(forBus: 0).channelCount > 0 else {
            return false
        }
        let node = AVAudioPlayerNode()
        newEngine.attach(node)
        newEngine.connect(node, to: newEngine.mainMixerNode, format: format)
        newEngine.prepare()
        do {
            try newEngine.start()
        } catch {
            return false
        }
        engine = newEngine
        playerNode = node
        activeRate = sampleRate
        activeChannels = channels
        return true
    }

    private func didFinishBuffer(generation value: Int, frames: Int,
                                 onPlayed: (@MainActor @Sendable () -> Void)?) {
        guard generation == value else { return }
        pendingBuffers = max(0, pendingBuffers - 1)
        bufferedFrames = max(0, bufferedFrames - frames)
        onPlayed?()
        if streamEnded && pendingBuffers == 0 {
            teardownEngine()
            isPlaying = false
        }
    }

    private func teardownEngine() {
        Self.log("teardown pending=\(pendingBuffers) ended=\(streamEnded)")
        playerNode?.stop()
        engine?.stop()
        playerNode = nil
        engine = nil
        activeRate = 0
        activeChannels = 0
    }
}
