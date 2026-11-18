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
    func finishStream()
    func stop()
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
    }

    func enqueue(_ chunk: AudioStreamChunk) {
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
        pendingBuffers += 1
        isPlaying = true
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.didFinishBuffer(generation: currentGen)
            }
        }
        if !node.isPlaying {
            node.play()
        }
    }

    func finishStream() {
        streamEnded = true
        if pendingBuffers == 0 {
            teardownEngine()
            isPlaying = false
        }
    }

    func stop() {
        generation += 1
        pendingBuffers = 0
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

    private func didFinishBuffer(generation value: Int) {
        guard generation == value else { return }
        pendingBuffers = max(0, pendingBuffers - 1)
        if streamEnded && pendingBuffers == 0 {
            teardownEngine()
            isPlaying = false
        }
    }

    private func teardownEngine() {
        playerNode?.stop()
        engine?.stop()
        playerNode = nil
        engine = nil
        activeRate = 0
        activeChannels = 0
    }
}
