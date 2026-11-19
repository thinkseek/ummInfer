// SPDX-License-Identifier: Apache-2.0
// Crosses the real player path (beginStream -> enqueue) for the two delivery
// extremes and for the hand-off between two streams.

import Foundation
import XCTest
@testable import ummmlx

@MainActor
final class JitterPlaybackTests: XCTestCase {
    /// 0.1 s of mono PCM16 at 24 kHz.
    private let chunk = AudioStreamChunk(
        pcm16: Data(repeating: 0, count: 4_800), sampleRate: 24_000, channels: 1)

    private func waitForEngine(_ player: LiveAudioStreamPlayer) throws {
        player.beginStream()
        player.enqueue(chunk)
        if player.outputDeviceUnavailable {
            throw XCTSkip("no audio output device in this environment")
        }
        XCTAssertEqual(player.jitterTargetSeconds, 0.3, accuracy: 1e-9,
                       "the first chunk must not move the target")
    }

    /// Burst delivery: chunks arrive back to back, so latency stays at the
    /// 0.3 s default.
    func testBurstDeliveryKeepsTheDefaultCushion() throws {
        let player = LiveAudioStreamPlayer()
        defer { player.stop() }
        try waitForEngine(player)
        for _ in 0..<4 { player.enqueue(chunk) }
        XCTAssertEqual(player.jitterTargetSeconds, 0.3, accuracy: 1e-9)
    }

    /// A link slower than 2 s between chunks must push the target to the cap.
    func testSlowDeliveryGrowsTheCushionToTheCap() throws {
        let player = LiveAudioStreamPlayer()
        defer { player.stop() }
        try waitForEngine(player)
        Thread.sleep(forTimeInterval: 2.5)
        player.enqueue(chunk)
        XCTAssertGreaterThan(player.jitterTargetSeconds, 2.9,
                             "a 2.5 s gap must grow the cushion toward 3 s")
        XCTAssertEqual(player.jitterTargetSeconds, 3.0, accuracy: 1e-9)
    }

    /// A stream that follows a slow one starts at the default. Before the fix
    /// the player kept its target across beginStream.
    func testSecondStreamDoesNotInheritTheFirstStreamsCushion() throws {
        let player = LiveAudioStreamPlayer()
        defer { player.stop() }
        try waitForEngine(player)
        Thread.sleep(forTimeInterval: 0.25)
        player.enqueue(chunk)
        XCTAssertGreaterThan(player.jitterTargetSeconds, 0.3,
                             "the first stream must have grown its target")

        player.beginStream()
        XCTAssertEqual(player.jitterTargetSeconds, 0.3, accuracy: 1e-9,
                       "a new stream must start at the default cushion")
    }
}
