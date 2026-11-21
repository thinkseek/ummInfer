// SPDX-License-Identifier: Apache-2.0
// Deterministic checks for the streamed-audio jitter target. The two extremes
// are a burst delivery and a link slower than real time.

import XCTest
@testable import ummmlx

final class JitterPolicyTests: XCTestCase {
    func testFreshStreamStartsAtTheLowTarget() {
        XCTAssertEqual(JitterPolicy().targetSeconds, 0.3, accuracy: 1e-9)
    }

    func testBurstDeliveryLeavesTheTargetAtTheDefault() {
        var policy = JitterPolicy()
        for _ in 0..<50 {
            policy.noteArrival(gap: 0.0005, starved: false)
        }
        policy.notePlayedChunk()
        XCTAssertEqual(policy.targetSeconds, 0.3, accuracy: 1e-9)
    }

    func testSlowLinkGrowsTheTargetAndCapsIt() {
        var policy = JitterPolicy()
        // A 2.5 s gap leaves the node dry; the next cushion must cover it.
        policy.noteArrival(gap: 2.5, starved: true)
        XCTAssertEqual(policy.targetSeconds, 3.0, accuracy: 1e-9)
        policy.noteArrival(gap: 10, starved: true)
        XCTAssertEqual(policy.targetSeconds, 3.0, accuracy: 1e-9)
    }

    func testCleanPlaybackShrinksTheTargetToTheFloor() {
        var policy = JitterPolicy()
        policy.noteArrival(gap: 2.5, starved: true)
        for _ in 0..<20 { policy.notePlayedChunk() }
        XCTAssertEqual(policy.targetSeconds, 2.55, accuracy: 1e-9)
        for _ in 0..<600 { policy.notePlayedChunk() }
        XCTAssertEqual(policy.targetSeconds, 0.25, accuracy: 1e-9)
    }

    /// The player allocates one policy per stream (`beginStream`). A stream
    /// that follows a slow one must not inherit its three-second cushion.
    func testEachStreamGetsAFreshTarget() {
        var slow = JitterPolicy()
        slow.noteArrival(gap: 2.5, starved: true)
        XCTAssertEqual(slow.targetSeconds, 3.0, accuracy: 1e-9)
        XCTAssertEqual(JitterPolicy().targetSeconds, 0.3, accuracy: 1e-9)
    }
}
