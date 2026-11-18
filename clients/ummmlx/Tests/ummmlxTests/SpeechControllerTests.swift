// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import ummmlx

/// Scripted recognizer. No microphone, no system permission, no model.
private final class FakeTranscriber: SpeechTranscribing, @unchecked Sendable {
    var updates: [SpeechUpdate] = []
    var failure: Error?
    var holdOpen = false
    private(set) var cancelCount = 0
    private var continuation: AsyncThrowingStream<SpeechUpdate, Error>.Continuation?

    func transcribeMicrophone() -> AsyncThrowingStream<SpeechUpdate, Error> {
        makeStream()
    }

    func transcribeFile(at url: URL) -> AsyncThrowingStream<SpeechUpdate, Error> {
        makeStream()
    }

    func cancel() {
        cancelCount += 1
        continuation?.finish()
        continuation = nil
    }

    private func makeStream() -> AsyncThrowingStream<SpeechUpdate, Error> {
        AsyncThrowingStream { continuation in
            self.continuation = continuation
            for update in updates { continuation.yield(update) }
            if let failure {
                continuation.finish(throwing: failure)
                self.continuation = nil
            } else if !holdOpen {
                continuation.finish()
                self.continuation = nil
            }
        }
    }
}

@MainActor
final class SpeechControllerTests: XCTestCase {
    func testFileRecognitionPublishesPartialsThenFinal() async throws {
        let fake = FakeTranscriber()
        fake.updates = [.partial("Hel"), .partial("Hello"), .final("Hello world")]
        let controller = SpeechController(transcriber: fake)
        var finals: [String] = []
        controller.onFinal = { finals.append($0) }

        let text = try await controller.transcribeFile(URL(fileURLWithPath: "/tmp/x.wav"))

        XCTAssertEqual(text, "Hello world")
        XCTAssertEqual(controller.finalText, "Hello world")
        XCTAssertEqual(controller.partial, "Hello world")
        XCTAssertEqual(controller.phase, .finished)
        XCTAssertEqual(finals, ["Hello world"])
        XCTAssertNil(controller.error)
    }

    func testPartialUpdatesReachTheCallback() async throws {
        let fake = FakeTranscriber()
        fake.updates = [.partial("one"), .partial("one two"), .final("one two")]
        let controller = SpeechController(transcriber: fake)
        var partials: [String] = []
        controller.onPartial = { partials.append($0) }

        _ = try await controller.transcribeFile(URL(fileURLWithPath: "/tmp/x.wav"))

        XCTAssertEqual(partials, ["one", "one two"])
    }

    func testRecognitionFailureSetsFailedPhase() async {
        let fake = FakeTranscriber()
        fake.failure = SpeechError.onDeviceUnavailable
        let controller = SpeechController(transcriber: fake)

        do {
            _ = try await controller.transcribeFile(URL(fileURLWithPath: "/tmp/x.wav"))
            XCTFail("A recognition failure must throw")
        } catch {
            XCTAssertEqual(controller.phase, .failed)
            XCTAssertEqual(controller.error, SpeechError.onDeviceUnavailable.errorDescription)
        }
    }

    func testEmptyTranscriptIsAFailure() async {
        let fake = FakeTranscriber()
        fake.updates = [.partial("noise")]
        let controller = SpeechController(transcriber: fake)

        do {
            _ = try await controller.transcribeFile(URL(fileURLWithPath: "/tmp/x.wav"))
            XCTFail("An empty final transcript must throw")
        } catch {
            XCTAssertEqual(error as? SpeechError, .emptyTranscript)
            XCTAssertEqual(controller.phase, .failed)
        }
    }

    func testMicrophoneFinalFillsTheText() async {
        let fake = FakeTranscriber()
        fake.updates = [.partial("mo"), .final("more")]
        let controller = SpeechController(transcriber: fake)
        let delivered = expectation(description: "final transcript")
        controller.onFinal = { text in
            XCTAssertEqual(text, "more")
            delivered.fulfill()
        }

        controller.startMicrophone()
        XCTAssertEqual(controller.phase, .listening)
        await fulfillment(of: [delivered], timeout: 2)
        XCTAssertEqual(controller.finalText, "more")
        XCTAssertEqual(controller.phase, .finished)
    }

    func testCancelStopsTheRecognizerAndReturnsToIdle() {
        let fake = FakeTranscriber()
        fake.holdOpen = true
        let controller = SpeechController(transcriber: fake)

        controller.startMicrophone()
        XCTAssertEqual(controller.phase, .listening)
        controller.cancel()

        XCTAssertEqual(fake.cancelCount, 1)
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertEqual(controller.partial, "")
    }

    func testCancelledRunnerDoesNotClearTheNewSession() async {
        let fake = FakeTranscriber()
        fake.holdOpen = true
        let controller = SpeechController(transcriber: fake)

        controller.startMicrophone()
        controller.cancel()
        controller.startMicrophone()
        // Let the cancelled runner finish its for-await loop.
        for _ in 0..<50 { await Task.yield() }

        XCTAssertTrue(controller.isRunning,
                      "the old runner must not clear the new task")
        XCTAssertEqual(controller.phase, .listening)
    }
}
