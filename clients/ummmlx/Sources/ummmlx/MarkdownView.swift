// SPDX-License-Identifier: Apache-2.0

import AppKit
import SwiftUI

enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case codeBlock(language: String?, code: String)
    case blockquote([String])
    case list(ordered: Bool, items: [String])
    case table(headers: [String], rows: [[String]])
    case divider
}

enum MarkdownParser {
    /// Parses a Markdown fragment (including incomplete streaming text such as
    /// an unclosed fenced code block) into balanced display blocks.
    static func parse(_ source: String) -> [MarkdownBlock] {
        let normalized = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var blocks: [MarkdownBlock] = []
        var i = 0

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                i += 1
                continue
            }

            // Fenced code block (``` or ~~~), resilient to unclosed fences mid-stream.
            if let fence = parseFenceOpen(trimmed) {
                i += 1
                var codeLines: [String] = []
                while i < lines.count {
                    if isFenceClose(lines[i].trimmingCharacters(in: .whitespaces), marker: fence.marker, length: fence.length) {
                        i += 1
                        break
                    }
                    codeLines.append(lines[i])
                    i += 1
                }
                blocks.append(.codeBlock(language: fence.language, code: codeLines.joined(separator: "\n")))
                continue
            }

            // Thematic break: ---, ***, ___
            if isHorizontalRule(trimmed) {
                blocks.append(.divider)
                i += 1
                continue
            }

            // ATX heading: # .. ######
            if let heading = parseHeading(trimmed) {
                blocks.append(.heading(level: heading.level, text: heading.text))
                i += 1
                continue
            }

            // GFM pipe table: header row + separator row
            if i + 1 < lines.count,
               let headers = parseTableRow(trimmed),
               isTableSeparator(lines[i + 1].trimmingCharacters(in: .whitespaces), columnCount: headers.count) {
                i += 2
                var rows: [[String]] = []
                while i < lines.count {
                    let rowTrimmed = lines[i].trimmingCharacters(in: .whitespaces)
                    if rowTrimmed.isEmpty { break }
                    guard let cells = parseTableRow(rowTrimmed) else { break }
                    var padded = cells
                    if padded.count < headers.count {
                        padded.append(contentsOf: Array(repeating: "", count: headers.count - padded.count))
                    } else if padded.count > headers.count {
                        padded = Array(padded.prefix(headers.count))
                    }
                    rows.append(padded)
                    i += 1
                }
                blocks.append(.table(headers: headers, rows: rows))
                continue
            }

            // Blockquote
            if trimmed.hasPrefix(">") {
                var quoteLines: [String] = []
                while i < lines.count {
                    let qTrim = lines[i].trimmingCharacters(in: .whitespaces)
                    guard qTrim.hasPrefix(">") else { break }
                    var content = String(qTrim.dropFirst())
                    if content.hasPrefix(" ") { content.removeFirst() }
                    quoteLines.append(content)
                    i += 1
                }
                blocks.append(.blockquote(quoteLines))
                continue
            }

            // Unordered or ordered list
            if let firstItem = parseListItem(line) {
                let ordered = firstItem.ordered
                var items: [String] = [firstItem.text]
                i += 1
                while i < lines.count {
                    let cur = lines[i]
                    let curTrim = cur.trimmingCharacters(in: .whitespaces)
                    if curTrim.isEmpty { break }
                    if let nextItem = parseListItem(cur), nextItem.ordered == ordered {
                        items.append(nextItem.text)
                        i += 1
                    } else if cur.hasPrefix("  ") || cur.hasPrefix("\t") {
                        items[items.count - 1] += "\n" + curTrim
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.list(ordered: ordered, items: items))
                continue
            }

            // Paragraph: collect contiguous plain lines until another block starts.
            var paraLines: [String] = [line]
            i += 1
            while i < lines.count {
                let next = lines[i]
                let nextTrim = next.trimmingCharacters(in: .whitespaces)
                if nextTrim.isEmpty
                    || parseFenceOpen(nextTrim) != nil
                    || isHorizontalRule(nextTrim)
                    || parseHeading(nextTrim) != nil
                    || nextTrim.hasPrefix(">")
                    || parseListItem(next) != nil
                    || (i + 1 < lines.count
                        && parseTableRow(nextTrim).map({ isTableSeparator(lines[i + 1].trimmingCharacters(in: .whitespaces), columnCount: $0.count) }) == true) {
                    break
                }
                paraLines.append(next)
                i += 1
            }
            blocks.append(.paragraph(paraLines.joined(separator: "\n")))
        }

        return blocks
    }

    private struct FenceOpen {
        let marker: Character
        let length: Int
        let language: String?
    }

    private static func parseFenceOpen(_ trimmed: String) -> FenceOpen? {
        guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
        var count = 0
        for ch in trimmed {
            if ch == first { count += 1 } else { break }
        }
        guard count >= 3 else { return nil }
        let rest = String(trimmed.dropFirst(count)).trimmingCharacters(in: .whitespaces)
        if first == "`" && rest.contains("`") { return nil }
        return FenceOpen(marker: first, length: count, language: rest.isEmpty ? nil : rest)
    }

    private static func isFenceClose(_ trimmed: String, marker: Character, length: Int) -> Bool {
        guard !trimmed.isEmpty, trimmed.allSatisfy({ $0 == marker }) else { return false }
        return trimmed.count >= length
    }

    private static func isHorizontalRule(_ trimmed: String) -> Bool {
        let compact = trimmed.filter { !$0.isWhitespace }
        guard compact.count >= 3, let first = compact.first,
              first == "-" || first == "*" || first == "_" else { return false }
        return compact.allSatisfy { $0 == first }
    }

    private static func parseHeading(_ trimmed: String) -> (level: Int, text: String)? {
        var level = 0
        for ch in trimmed {
            if ch == "#" { level += 1 } else { break }
        }
        guard (1...6).contains(level) else { return nil }
        let rest = trimmed.dropFirst(level)
        guard rest.first == " " || rest.first == "\t" else { return nil }
        var title = rest.trimmingCharacters(in: .whitespaces)
        while title.hasSuffix("#") {
            let without = title.dropLast()
            if without.isEmpty || without.last == " " || without.last == "\t" {
                title = without.trimmingCharacters(in: .whitespaces)
            } else {
                break
            }
        }
        guard !title.isEmpty else { return nil }
        return (level, title)
    }

    private static func parseListItem(_ line: String) -> (ordered: Bool, text: String)? {
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        guard trimmed.count >= 2 else { return nil }
        let first = trimmed.first!
        if (first == "-" || first == "*" || first == "+") {
            let after = trimmed.dropFirst()
            if after.first == " " || after.first == "\t" {
                let text = after.drop(while: { $0 == " " || $0 == "\t" })
                if !text.isEmpty { return (false, String(text)) }
            }
        }
        var digits = 0
        var idx = trimmed.startIndex
        while idx < trimmed.endIndex, trimmed[idx].isNumber, digits < 4 {
            digits += 1
            idx = trimmed.index(after: idx)
        }
        if digits > 0, idx < trimmed.endIndex,
           (trimmed[idx] == "." || trimmed[idx] == ")") {
            let afterPunct = trimmed.index(after: idx)
            if afterPunct < trimmed.endIndex,
               (trimmed[afterPunct] == " " || trimmed[afterPunct] == "\t") {
                let text = trimmed[afterPunct...].drop(while: { $0 == " " || $0 == "\t" })
                if !text.isEmpty { return (true, String(text)) }
            }
        }
        return nil
    }

    private static func parseTableRow(_ trimmed: String) -> [String]? {
        guard trimmed.contains("|") else { return nil }
        var inner = trimmed
        if inner.hasPrefix("|") { inner.removeFirst() }
        if inner.hasSuffix("|") { inner.removeLast() }
        let parts = inner.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count >= 2 else { return nil }
        return parts
    }

    private static func isTableSeparator(_ trimmed: String, columnCount: Int) -> Bool {
        guard let cells = parseTableRow(trimmed), cells.count == columnCount else { return false }
        return cells.allSatisfy { cell in
            let s = cell.trimmingCharacters(in: .whitespaces)
            guard !s.isEmpty else { return false }
            let core = s.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            return core.count >= 3 && core.allSatisfy { $0 == "-" }
        }
    }
}

enum MarkdownTypography {
    /// Restrained, balanced heading scale relative to the 13 pt chat body so
    /// Markdown titles give clear hierarchy without giant oversized fonts.
    static func headingSize(for level: Int, bodySize: CGFloat) -> CGFloat {
        switch level {
        case 1: return bodySize + 2.5   // 15.5 pt
        case 2: return bodySize + 1.5   // 14.5 pt
        case 3: return bodySize + 0.5   // 13.5 pt
        default: return bodySize        // 13.0 pt
        }
    }

    static func headingWeight(for level: Int) -> Font.Weight {
        level <= 3 ? .semibold : .medium
    }

    static func codeSize(bodySize: CGFloat) -> CGFloat {
        max(11, bodySize - 1)
    }

    static func inlineAttributed(_ text: String, theme: Theme, size: CGFloat,
                                 weight: Font.Weight = .regular) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        var result: AttributedString
        if let parsed = try? AttributedString(markdown: text, options: options) {
            result = parsed
        } else {
            result = AttributedString(text)
        }

        let codeFont = Font.system(size: codeSize(bodySize: size), design: .monospaced)
        for run in result.runs {
            let range = run.range
            if let intent = run.inlinePresentationIntent, intent.contains(.code) {
                result[range].font = codeFont
                result[range].backgroundColor = theme.hover
            } else if run.link != nil {
                result[range].foregroundColor = theme.accent
                result[range].underlineStyle = .single
            } else if weight != .regular {
                // Keep heading weight while preserving bold/italic runs if present.
                let isItalic = run.inlinePresentationIntent?.contains(.emphasized) == true
                if isItalic {
                    result[range].font = .system(size: size, weight: weight).italic()
                } else {
                    result[range].font = .system(size: size, weight: weight)
                }
            }
        }
        return result
    }
}

struct MarkdownTextView: View {
    let text: String
    let theme: Theme

    var body: some View {
        let blocks = MarkdownParser.parse(text)
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                blockView(block, isFirst: index == 0)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock, isFirst: Bool) -> some View {
        switch block {
        case .heading(let level, let title):
            let size = MarkdownTypography.headingSize(for: level, bodySize: theme.bodySize)
            let weight = MarkdownTypography.headingWeight(for: level)
            VStack(alignment: .leading, spacing: 4) {
                Text(MarkdownTypography.inlineAttributed(title, theme: theme, size: size, weight: weight))
                    .font(.system(size: size, weight: weight))
                    .foregroundStyle(level >= 5 ? theme.secondary : theme.text)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
                if level == 1 {
                    Divider().overlay(theme.line)
                }
            }
            .padding(.top, isFirst ? 0 : (level <= 2 ? 4 : 2))

        case .paragraph(let body):
            Text(MarkdownTypography.inlineAttributed(body, theme: theme, size: theme.bodySize))
                .font(.system(size: theme.bodySize))
                .foregroundStyle(theme.text)
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)

        case .codeBlock(let language, let code):
            MarkdownCodeBlockView(language: language, code: code, theme: theme)

        case .blockquote(let lines):
            HStack(alignment: .top, spacing: 9) {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(theme.accent.opacity(0.65))
                    .frame(width: 2.5)
                Text(MarkdownTypography.inlineAttributed(lines.joined(separator: "\n"), theme: theme, size: theme.bodySize))
                    .font(.system(size: theme.bodySize))
                    .foregroundStyle(theme.secondary)
                    .lineSpacing(3.5)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 2)

        case .list(let ordered, let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Text(ordered ? "\(idx + 1)." : "•")
                            .font(.system(size: theme.bodySize, weight: .medium))
                            .foregroundStyle(theme.secondary)
                            .frame(minWidth: ordered ? 18 : 10, alignment: .trailing)
                        Text(MarkdownTypography.inlineAttributed(item, theme: theme, size: theme.bodySize))
                            .font(.system(size: theme.bodySize))
                            .foregroundStyle(theme.text)
                            .lineSpacing(3.5)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 2)

        case .table(let headers, let rows):
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    tableRow(headers, isHeader: true)
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        Divider().overlay(theme.line)
                        tableRow(row, isHeader: false)
                    }
                }
                .background(theme.background)
                .clipShape(RoundedRectangle(cornerRadius: theme.controlRadius))
                .overlay(RoundedRectangle(cornerRadius: theme.controlRadius).stroke(theme.line))
            }

        case .divider:
            Divider().overlay(theme.line).padding(.vertical, 2)
        }
    }

    private func tableRow(_ cells: [String], isHeader: Bool) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(cells.enumerated()), id: \.offset) { idx, cell in
                if idx > 0 {
                    Divider().overlay(theme.line)
                }
                Text(MarkdownTypography.inlineAttributed(
                    cell, theme: theme, size: theme.captionSize,
                    weight: isHeader ? .semibold : .regular))
                    .font(.system(size: theme.captionSize, weight: isHeader ? .semibold : .regular))
                    .foregroundStyle(isHeader ? theme.text : theme.text)
                    .frame(minWidth: 84, maxWidth: 220, alignment: .leading)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
            }
        }
        .background(isHeader ? theme.hover : Color.clear)
    }
}

private struct MarkdownCodeBlockView: View {
    let language: String?
    let code: String
    let theme: Theme
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text((language?.isEmpty == false ? language! : "code").lowercased())
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(theme.secondary)
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 1_200_000_000)
                        copied = false
                    }
                }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(copied ? theme.accent : theme.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(theme.hover)

            Divider().overlay(theme.line)

            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(.system(size: MarkdownTypography.codeSize(bodySize: theme.bodySize), design: .monospaced))
                    .foregroundStyle(theme.text)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(theme.background)
        .clipShape(RoundedRectangle(cornerRadius: theme.controlRadius))
        .overlay(RoundedRectangle(cornerRadius: theme.controlRadius).stroke(theme.line))
    }
}
