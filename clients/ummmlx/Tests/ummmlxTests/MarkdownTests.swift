// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import XCTest
@testable import ummmlx

final class MarkdownTests: XCTestCase {
    func testParsesHeadingsParagraphsListsQuotesRulesCodeAndTables() {
        let md = """
        # Title One
        ## Section Two
        ### Subsection Three

        Intro with **bold**, *italic*, and `inline()`.

        - First item
        - Second item

        1. Step one
        2. Step two

        > Quoted line one
        > Quoted line two

        ---

        | Name | Value |
        | --- | --- |
        | FPS | 16 |
        | Size | 704x480 |

        ```swift
        let x = 1
        ```
        """

        let blocks = MarkdownParser.parse(md)
        XCTAssertEqual(blocks, [
            .heading(level: 1, text: "Title One"),
            .heading(level: 2, text: "Section Two"),
            .heading(level: 3, text: "Subsection Three"),
            .paragraph("Intro with **bold**, *italic*, and `inline()`."),
            .list(ordered: false, items: ["First item", "Second item"]),
            .list(ordered: true, items: ["Step one", "Step two"]),
            .blockquote(["Quoted line one", "Quoted line two"]),
            .divider,
            .table(headers: ["Name", "Value"], rows: [["FPS", "16"], ["Size", "704x480"]]),
            .codeBlock(language: "swift", code: "let x = 1"),
        ])
    }

    func testStreamingUnclosedCodeFenceDoesNotCrashOrDropContent() {
        let partial = "Here is code:\n```python\ndef hello():\n    return 42"
        let blocks = MarkdownParser.parse(partial)
        XCTAssertEqual(blocks, [
            .paragraph("Here is code:"),
            .codeBlock(language: "python", code: "def hello():\n    return 42"),
        ])
    }

    func testHeadingScaleIsBalancedAndNotOversized() {
        let theme = Theme(dark: false)
        let h1 = MarkdownTypography.headingSize(for: 1, bodySize: theme.bodySize)
        let h2 = MarkdownTypography.headingSize(for: 2, bodySize: theme.bodySize)
        let h3 = MarkdownTypography.headingSize(for: 3, bodySize: theme.bodySize)
        let h4 = MarkdownTypography.headingSize(for: 4, bodySize: theme.bodySize)
        let code = MarkdownTypography.codeSize(bodySize: theme.bodySize)

        XCTAssertGreaterThan(h1, h2)
        XCTAssertGreaterThan(h2, h3)
        XCTAssertGreaterThan(h3, h4)
        XCTAssertEqual(h4, theme.bodySize)
        XCTAssertLessThanOrEqual(h1, 16.0, "H1 should stay compact and balanced with 13pt body")
        XCTAssertLessThan(code, theme.bodySize)
    }

    func testInlineAttributedFallsBackGracefullyOnMalformedMarkdown() {
        let theme = Theme(dark: false)
        let rendered = MarkdownTypography.inlineAttributed("Unclosed **bold and `code", theme: theme, size: theme.bodySize)
        XCTAssertFalse(String(rendered.characters).isEmpty)
    }
}
