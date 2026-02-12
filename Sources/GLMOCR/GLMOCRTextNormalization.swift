import Foundation

public enum GLMOCRTextNormalization {
    public static func cleanContent(_ input: String?) -> String {
        guard var content = input else { return "" }

        
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

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func normalizeText(_ input: String) -> String {
        var content = cleanContent(input)

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

    private static func removeCJKInterCharacterSpaces(_ input: String) -> String {
        input.replacingOccurrences(
            of: #"(?<=[\p{Han}\p{Hiragana}\p{Katakana}])[ \t]+(?=[\p{Han}\p{Hiragana}\p{Katakana}])"#,
            with: "",
            options: [.regularExpression]
        )
    }

    private static func stripSpuriousEmphasis(_ input: String) -> String {
        input.replacingOccurrences(
            of: #"\*([A-Za-z0-9._/\-]+)\*"#,
            with: "$1",
            options: [.regularExpression]
        )
    }

    private static func replaceSingleNewlines(_ input: String) -> String {
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

    private static func cleanRepeatedContent(
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

    private static func findConsecutiveRepeat(_ string: String, minUnitLen: Int, minRepeats: Int) -> String? {
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

    private static func mostCommon(_ lines: [String]) -> (line: String, count: Int)? {
        guard !lines.isEmpty else { return nil }
        var counts: [String: Int] = [:]
        counts.reserveCapacity(lines.count)
        for line in lines {
            counts[line, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value }).map { ($0.key, $0.value) }
    }
}
