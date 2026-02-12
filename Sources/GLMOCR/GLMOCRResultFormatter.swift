import Foundation

public struct GLMOCRFormattedRegion: Codable, Sendable, Equatable {
    public var index: Int
    public var label: String
    public var bbox2D: [Int]?
    public var content: String?
    public var nativeLabel: String

    public init(index: Int, label: String, bbox2D: [Int]?, content: String?, nativeLabel: String) {
        self.index = index
        self.label = label
        self.bbox2D = bbox2D
        self.content = content
        self.nativeLabel = nativeLabel
    }

    enum CodingKeys: String, CodingKey {
        case index
        case label
        case bbox2D = "bbox_2d"
        case content
        case nativeLabel = "native_label"
    }
}

public struct GLMOCRParseResult: Codable, Sendable, Equatable {
    public var jsonResult: [[GLMOCRFormattedRegion]]
    public var markdownResult: String

    public init(jsonResult: [[GLMOCRFormattedRegion]], markdownResult: String) {
        self.jsonResult = jsonResult
        self.markdownResult = markdownResult
    }
}

public struct GLMOCRResultFormatter: Sendable {
    public init() {}

    public func format(pages: [[GLMOCRFormattedRegion]]) -> GLMOCRParseResult {
        var formattedPages: [[GLMOCRFormattedRegion]] = []
        formattedPages.reserveCapacity(pages.count)

        for pageRegions in pages {
            var regions = pageRegions.sorted { $0.index < $1.index }

            regions = regions.compactMap { region in
                let mappedLabel = mapLabel(region.nativeLabel)
                let formattedContent = formatContent(
                    region.content,
                    mappedLabel: mappedLabel,
                    nativeLabel: region.nativeLabel
                )
                if let formattedContent, formattedContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return nil
                }
                return GLMOCRFormattedRegion(
                    index: region.index,
                    label: mappedLabel,
                    bbox2D: region.bbox2D,
                    content: formattedContent,
                    nativeLabel: region.nativeLabel
                )
            }

            regions = mergeFormulaNumbers(regions)
            regions = mergeTextBlocks(regions)
            regions = formatBulletPoints(regions)
            regions = reindex(regions)
            formattedPages.append(regions)
        }

        let markdown = buildMarkdown(pages: formattedPages)
        return GLMOCRParseResult(jsonResult: formattedPages, markdownResult: markdown)
    }

    private func reindex(_ regions: [GLMOCRFormattedRegion]) -> [GLMOCRFormattedRegion] {
        regions.enumerated().map { idx, region in
            var copy = region
            copy.index = idx
            return copy
        }
    }

    private func buildMarkdown(pages: [[GLMOCRFormattedRegion]]) -> String {
        var pageStrings: [String] = []
        pageStrings.reserveCapacity(pages.count)

        for (pageIdx, regions) in pages.enumerated() {
            var blocks: [String] = []
            blocks.reserveCapacity(regions.count)

            for region in regions {
                switch region.label {
                case "image":
                    let bbox = region.bbox2D ?? []
                    blocks.append("![](page=\(pageIdx),bbox=\(bbox))")
                default:
                    if let content = region.content, !content.isEmpty {
                        blocks.append(content)
                    }
                }
            }

            pageStrings.append(blocks.joined(separator: "\n\n"))
        }

        return pageStrings.joined(separator: "\n\n")
    }

    private func mergeFormulaNumbers(_ regions: [GLMOCRFormattedRegion]) -> [GLMOCRFormattedRegion] {
        guard !regions.isEmpty else { return regions }

        var out: [GLMOCRFormattedRegion] = []
        out.reserveCapacity(regions.count)

        var i = 0
        while i < regions.count {
            let region = regions[i]

            if region.nativeLabel == "formula_number" {
                if i + 1 < regions.count, regions[i + 1].label == "formula" {
                    let number = cleanFormulaNumber(regions[i].content ?? "")
                    var merged = regions[i + 1]
                    if let content = merged.content, content.hasSuffix("\n$$") {
                        merged.content = content.dropLast(3) + " \\tag{\(number)}\n$$"
                    }
                    out.append(merged)
                    i += 2
                    continue
                }
                i += 1
                continue
            }

            if region.label == "formula" {
                if i + 1 < regions.count, regions[i + 1].nativeLabel == "formula_number" {
                    let number = cleanFormulaNumber(regions[i + 1].content ?? "")
                    var merged = region
                    if let content = merged.content, content.hasSuffix("\n$$") {
                        merged.content = content.dropLast(3) + " \\tag{\(number)}\n$$"
                    }
                    out.append(merged)
                    i += 2
                    continue
                }
            }

            out.append(region)
            i += 1
        }

        return out
    }

    private func cleanFormulaNumber(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("("), s.hasSuffix(")") {
            s = String(s.dropFirst().dropLast())
        } else if s.hasPrefix("（"), s.hasSuffix("）") {
            s = String(s.dropFirst().dropLast())
        }
        return s
    }

    private func mapLabel(_ nativeLabel: String) -> String {
        if Self.imageLabels.contains(nativeLabel) { return "image" }
        if Self.textLabels.contains(nativeLabel) { return "text" }
        if Self.tableLabels.contains(nativeLabel) { return "table" }
        if Self.formulaLabels.contains(nativeLabel) { return "formula" }
        return nativeLabel
    }

    private func formatContent(_ content: String?, mappedLabel: String, nativeLabel: String) -> String? {
        guard var content else { return content }
        content = cleanContent(content)

        if mappedLabel == "table" {
            content = normalizeTableHTML(content)
        }

        if nativeLabel == "algorithm" {
            content = normalizeAlgorithmCodeBlock(content)
        }

        if nativeLabel == "doc_title" {
            content = content.replacingOccurrences(
                of: #"^#+\s*"#,
                with: "",
                options: [.regularExpression]
            )
            content = "# " + content
        } else if nativeLabel == "paragraph_title" {
            if content.hasPrefix("- ") || content.hasPrefix("* ") {
                content = String(content.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            content = content.replacingOccurrences(
                of: #"^#+\s*"#,
                with: "",
                options: [.regularExpression]
            )
            content = "## " + content.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if mappedLabel == "formula" {
            content = wrapFormula(content)
        }

        if mappedLabel == "text" {
            content = normalizeText(content)
        }

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func cleanContent(_ input: String) -> String {
        var content = input
        while content.hasPrefix("\\t") { content = String(content.dropFirst(2)) }
        while content.hasSuffix("\\t") { content = String(content.dropLast(2)) }
        content = content.trimmingCharacters(in: .whitespacesAndNewlines)

        
        content = content.replacingOccurrences(of: #"(\.)\1{2,}"#, with: "$1$1$1", options: [.regularExpression])
        content = content.replacingOccurrences(of: #"(·)\1{2,}"#, with: "$1$1$1", options: [.regularExpression])
        content = content.replacingOccurrences(of: #"(_)\1{2,}"#, with: "$1$1$1", options: [.regularExpression])
        content = content.replacingOccurrences(of: #"(\\_)\1{2,}"#, with: "$1$1$1", options: [.regularExpression])

        
        if content.count >= 2048 {
            content = cleanRepeatedContent(content)
        }

        return content
    }

    private func normalizeTableHTML(_ input: String) -> String {
        var content = input
        content = content.replacingOccurrences(
            of: #"<table\s+class=(["'])table table-bordered\1\s*>"#,
            with: "<table>",
            options: [.regularExpression]
        )
        return content
    }

    private func wrapFormula(_ input: String) -> String {
        var content = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if content.hasPrefix("$$"), content.hasSuffix("$$") {
            content = String(content.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespacesAndNewlines)
        } else if content.hasPrefix("\\["), content.hasSuffix("\\]") {
            content = String(content.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespacesAndNewlines)
        } else if content.hasPrefix("\\("), content.hasSuffix("\\)") {
            content = String(content.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return "$$\n" + content + "\n$$"
    }

    private func normalizeText(_ input: String) -> String {
        var content = input

        content = stripSpuriousEmphasis(content)
        content = removeCJKInterCharacterSpaces(content)

        if content.hasPrefix("·") || content.hasPrefix("•") {
            content = "- " + String(content.dropFirst()).trimmingCharacters(in: .whitespacesAndNewlines)
        } else if content.hasPrefix("* ") {
            content = "- " + String(content.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let m = content.firstMatch(of: /^\(?\p{Z}*([（(])(\d+|[A-Za-z])([）)])(.*)$/) {
            let open = String(m.1)
            let symbol = String(m.2)
            let close = String(m.3)
            let rest = String(m.4).trimmingCharacters(in: .whitespacesAndNewlines)
            content = "\(open)\(symbol)\(close) \(rest)"
        } else if let m = content.firstMatch(of: /^(\d+|[A-Za-z])([.)）])(.*)$/) {
            let symbol = String(m.1)
            var sep = String(m.2)
            let rest = String(m.3).trimmingCharacters(in: .whitespacesAndNewlines)
            if sep == "）" { sep = ")" }
            content = "\(symbol)\(sep) \(rest)"
        }

        content = replaceSingleNewlines(content)
        return content
    }

    private func removeCJKInterCharacterSpaces(_ input: String) -> String {
        input.replacingOccurrences(
            of: #"(?<=[\p{Han}\p{Hiragana}\p{Katakana}])[ \t]+(?=[\p{Han}\p{Hiragana}\p{Katakana}])"#,
            with: "",
            options: [.regularExpression]
        )
    }

    private func stripSpuriousEmphasis(_ input: String) -> String {
        input.replacingOccurrences(
            of: #"\*([A-Za-z0-9._/\-]+)\*"#,
            with: "$1",
            options: [.regularExpression]
        )
    }

    private func normalizeAlgorithmCodeBlock(_ input: String) -> String {
        guard input.hasPrefix("```html") else { return input }
        let lower = input.lowercased()
        if lower.contains("<html") || lower.contains("<body") || lower.contains("<div") {
            return input
        }
        if lower.contains("</") || lower.contains("/>") || lower.contains("<weblogic") || lower.contains("<local-") {
            return input.replacingOccurrences(of: "```html", with: "```xml", options: [.anchored])
        }
        return input
    }

    private func replaceSingleNewlines(_ input: String) -> String {
        var out = ""
        out.reserveCapacity(input.count + 8)
        let chars = Array(input)
        for i in chars.indices {
            let ch = chars[i]
            if ch != "\n" {
                out.append(ch)
                continue
            }
            let prevIsNewline = i > chars.startIndex ? chars[chars.index(before: i)] == "\n" : false
            let nextIsNewline = chars.index(after: i) < chars.endIndex ? chars[chars.index(after: i)] == "\n" : false
            if !prevIsNewline && !nextIsNewline {
                out.append("\n")
                out.append("\n")
            } else {
                out.append("\n")
            }
        }
        return out
    }

    private func mergeTextBlocks(_ regions: [GLMOCRFormattedRegion]) -> [GLMOCRFormattedRegion] {
        guard !regions.isEmpty else { return regions }

        var merged: [GLMOCRFormattedRegion] = []
        merged.reserveCapacity(regions.count)
        var skipIndices = Set<Int>()

        for i in regions.indices {
            if skipIndices.contains(i) { continue }

            let block = regions[i]
            guard block.label == "text", let content = block.content else {
                merged.append(block)
                continue
            }

            let trimmedTrailing = rstrip(content)
            guard !trimmedTrailing.isEmpty, trimmedTrailing.hasSuffix("-") else {
                merged.append(block)
                continue
            }

            var didMerge = false
            for j in (i + 1)..<regions.count {
                let next = regions[j]
                guard next.label == "text", let nextContent = next.content else { continue }

                let nextTrimmedLeading = lstrip(nextContent)
                guard let first = nextTrimmedLeading.first, first.isLowercase else { break }

                let beforeWords = String(trimmedTrailing.dropLast()).split(whereSeparator: \.isWhitespace)
                let afterWords = nextTrimmedLeading.split(whereSeparator: \.isWhitespace)
                guard let before = beforeWords.last, let after = afterWords.first else { break }
                guard isAlphabeticWord(before), isAlphabeticWord(after) else { break }

                var mergedBlock = block
                mergedBlock.content = String(trimmedTrailing.dropLast()) + nextTrimmedLeading
                merged.append(mergedBlock)
                skipIndices.insert(j)
                didMerge = true
                break
            }

            if !didMerge {
                merged.append(block)
            }
        }

        return merged
    }

    private func formatBulletPoints(
        _ regions: [GLMOCRFormattedRegion],
        leftAlignThreshold: Double = 10.0
    ) -> [GLMOCRFormattedRegion] {
        guard regions.count >= 3 else { return regions }
        var formatted = regions

        for i in 1..<(formatted.count - 1) {
            let prev = formatted[i - 1]
            var current = formatted[i]
            let next = formatted[i + 1]

            guard current.nativeLabel == "text",
                  prev.nativeLabel == "text",
                  next.nativeLabel == "text",
                  let currentContent = current.content,
                  let prevContent = prev.content,
                  let nextContent = next.content,
                  !currentContent.hasPrefix("- "),
                  prevContent.hasPrefix("- "),
                  nextContent.hasPrefix("- "),
                  let currentBBox = current.bbox2D, currentBBox.count >= 1,
                  let prevBBox = prev.bbox2D, prevBBox.count >= 1,
                  let nextBBox = next.bbox2D, nextBBox.count >= 1
            else {
                continue
            }

            let currentLeft = Double(currentBBox[0])
            let prevLeft = Double(prevBBox[0])
            let nextLeft = Double(nextBBox[0])

            if abs(currentLeft - prevLeft) <= leftAlignThreshold,
               abs(currentLeft - nextLeft) <= leftAlignThreshold
            {
                current.content = "- " + currentContent
                formatted[i] = current
            }
        }

        return formatted
    }

    private func cleanRepeatedContent(
        _ content: String,
        minLen: Int = 10,
        minRepeats: Int = 10,
        lineThreshold: Int = 10
    ) -> String {
        let stripped = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else { return content }

        if stripped.count > minLen * minRepeats,
           let repeatCleaned = findConsecutiveRepeat(stripped, minUnitLen: minLen, minRepeats: minRepeats)
        {
            return repeatCleaned
        }

        let lines = content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard lines.count >= lineThreshold, let (mostCommonLine, count) = mostCommon(lines) else {
            return content
        }

        let ratio = Double(count) / Double(lines.count)
        guard count >= lineThreshold, ratio >= 0.8 else {
            return content
        }

        for i in lines.indices where lines[i] == mostCommonLine {
            let end = min(i + 3, lines.count)
            let consecutive = lines[i..<end].reduce(0) { partial, line in
                partial + (line == mostCommonLine ? 1 : 0)
            }

            if consecutive >= 3 {
                let originalLines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
                var nonEmptyCount = 0
                for (idx, line) in originalLines.enumerated() {
                    if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        nonEmptyCount += 1
                        if nonEmptyCount == i + 1 {
                            return originalLines[0...idx].joined(separator: "\n")
                        }
                    }
                }
                break
            }
        }

        return content
    }

    private func findConsecutiveRepeat(_ string: String, minUnitLen: Int, minRepeats: Int) -> String? {
        let n = string.count
        guard n >= minUnitLen * minRepeats else { return nil }
        let maxUnitLen = n / minRepeats
        guard maxUnitLen >= minUnitLen else { return nil }

        let pattern = "(.{\(minUnitLen),\(maxUnitLen)}?)\\\\1{\(minRepeats - 1),}"
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators]
        ) else {
            return nil
        }

        let ns = string as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = regex.firstMatch(in: string, options: [], range: range),
              match.numberOfRanges >= 2
        else {
            return nil
        }

        let prefix = ns.substring(to: match.range.location)
        let unit = ns.substring(with: match.range(at: 1))
        return prefix + unit
    }

    private func mostCommon(_ lines: [String]) -> (line: String, count: Int)? {
        guard !lines.isEmpty else { return nil }
        var counts: [String: Int] = [:]
        counts.reserveCapacity(lines.count)
        for line in lines {
            counts[line, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value }).map { ($0.key, $0.value) }
    }

    private func lstrip(_ string: String) -> String {
        guard let first = string.firstIndex(where: { !$0.isWhitespace }) else { return "" }
        return String(string[first...])
    }

    private func rstrip(_ string: String) -> String {
        guard let last = string.lastIndex(where: { !$0.isWhitespace }) else { return "" }
        return String(string[...last])
    }

    private func isAlphabeticWord<S: StringProtocol>(_ token: S) -> Bool {
        !token.isEmpty && token.allSatisfy(\.isLetter)
    }

    private static let imageLabels: Set<String> = ["chart", "image"]
    private static let tableLabels: Set<String> = ["table"]
    private static let formulaLabels: Set<String> = ["display_formula", "inline_formula", "formula"]
    private static let textLabels: Set<String> = [
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    ]
}
