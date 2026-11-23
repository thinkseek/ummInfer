// SPDX-License-Identifier: Apache-2.0
// Theme.swift is the single source of the client palette. These tests pin the
// light/dark tokens and the hex constructor so a token drift is caught here
// instead of in a screenshot.

import AppKit
import SwiftUI
import XCTest
@testable import ummmlx

final class ThemeTests: XCTestCase {
    /// Resolve a SwiftUI Color to sRGB components so a hex token can be checked.
    private func rgb(_ color: Color) throws -> (Double, Double, Double) {
        let converted = try XCTUnwrap(NSColor(color).usingColorSpace(.sRGB))
        return (Double(converted.redComponent),
                Double(converted.greenComponent),
                Double(converted.blueComponent))
    }

    private func assertHex(_ color: Color, _ hex: UInt32, _ label: String,
                           file: StaticString = #filePath, line: UInt = #line) throws {
        let (red, green, blue) = try rgb(color)
        let expected = (Double((hex >> 16) & 0xFF) / 255,
                        Double((hex >> 8) & 0xFF) / 255,
                        Double(hex & 0xFF) / 255)
        XCTAssertEqual(red, expected.0, accuracy: 0.002, "\(label) red", file: file, line: line)
        XCTAssertEqual(green, expected.1, accuracy: 0.002, "\(label) green", file: file, line: line)
        XCTAssertEqual(blue, expected.2, accuracy: 0.002, "\(label) blue", file: file, line: line)
    }

    func testLightPaletteTokens() throws {
        let theme = Theme(dark: false)
        try assertHex(theme.background, 0xFFFFFF, "light background")
        try assertHex(theme.surface, 0xF7F7F7, "light surface")
        try assertHex(theme.hover, 0xF2F2F2, "light hover")
        try assertHex(theme.line, 0xEBEBEB, "light line")
        try assertHex(theme.text, 0x111111, "light text")
        try assertHex(theme.secondary, 0x6E6E6E, "light secondary")
        try assertHex(theme.tertiary, 0xA3A3A3, "light tertiary")
        try assertHex(theme.ink, 0x111111, "light ink")
        try assertHex(theme.onInk, 0xFFFFFF, "light onInk")
        try assertHex(theme.accent, 0x0D9384, "light accent")
        try assertHex(theme.danger, 0xEB5230, "light danger")
    }

    func testDarkPaletteTokens() throws {
        let theme = Theme(dark: true)
        try assertHex(theme.background, 0x151515, "dark background")
        try assertHex(theme.surface, 0x1C1C1C, "dark surface")
        try assertHex(theme.hover, 0x232323, "dark hover")
        try assertHex(theme.line, 0x2B2B2B, "dark line")
        try assertHex(theme.text, 0xF2F2F2, "dark text")
        try assertHex(theme.secondary, 0xA3A3A3, "dark secondary")
        try assertHex(theme.tertiary, 0x6B6B6B, "dark tertiary")
        try assertHex(theme.ink, 0xF2F2F2, "dark ink")
        try assertHex(theme.onInk, 0x151515, "dark onInk")
        try assertHex(theme.accent, 0x2BB3A3, "dark accent")
        try assertHex(theme.danger, 0xFF6B47, "dark danger")
    }

    func testHexConstructorReachesTheChannelBits() throws {
        // The constructor must read each byte, not collapse to grayscale.
        let theme = Theme(dark: false)
        let (red, green, blue) = try rgb(theme.accent)
        XCTAssertGreaterThan(blue, red, "accent must keep its blue channel")
        XCTAssertGreaterThan(green, blue, "accent must keep its green channel")
    }

    private func assertDiffers(_ first: Color, _ second: Color, _ label: String,
                               file: StaticString = #filePath, line: UInt = #line) throws {
        let (red, green, blue) = try rgb(first)
        let (otherRed, otherGreen, otherBlue) = try rgb(second)
        XCTAssertTrue(red != otherRed || green != otherGreen || blue != otherBlue,
                      label, file: file, line: line)
    }

    func testLightAndDarkDiffer() throws {
        let light = Theme(dark: false)
        let dark = Theme(dark: true)
        try assertDiffers(light.background, dark.background, "background must change with the mode")
        try assertDiffers(light.text, dark.text, "text must change with the mode")
        try assertDiffers(light.accent, dark.accent, "accent must change with the mode")
    }

    /// The web client mirrors these tokens; a value change here must stay in
    /// sync with the `:root` block of web/index.html.
    func testMetricsAreStable() {
        let theme = Theme(dark: false)
        XCTAssertEqual(theme.controlRadius, 8)
        XCTAssertEqual(theme.cardRadius, 12)
        XCTAssertEqual(theme.titleSize, 20)
        XCTAssertEqual(theme.sectionSize, 15)
        XCTAssertEqual(theme.bodySize, 13)
        XCTAssertEqual(theme.captionSize, 12)
    }
}
