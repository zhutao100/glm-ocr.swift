import XCTest

@testable import GLMOCR

final class GLMOCRResultFormatterTests: XCTestCase {
    func testMergesFormulaThenFormulaNumber() {
        let formatter = GLMOCRResultFormatter()

        let page: [GLMOCRFormattedRegion] = [
            .init(
                index: 0, label: "formula", bbox2D: [0, 0, 10, 10], content: "E = mc^2", nativeLabel: "display_formula"),
            .init(index: 1, label: "text", bbox2D: [0, 11, 10, 12], content: "(1)", nativeLabel: "formula_number"),
        ]

        let result = formatter.format(pages: [page])
        XCTAssertEqual(result.jsonResult.count, 1)
        XCTAssertEqual(result.jsonResult[0].count, 1)
        XCTAssertEqual(result.jsonResult[0][0].label, "formula")
        XCTAssertTrue(result.jsonResult[0][0].content?.contains("\\tag{1}") ?? false)
    }

    func testMergesFormulaNumberThenFormula() {
        let formatter = GLMOCRResultFormatter()

        let page: [GLMOCRFormattedRegion] = [
            .init(index: 0, label: "text", bbox2D: [0, 11, 10, 12], content: "（2.1）", nativeLabel: "formula_number"),
            .init(
                index: 1, label: "formula", bbox2D: [0, 0, 10, 10], content: "$$x+y$$", nativeLabel: "display_formula"),
        ]

        let result = formatter.format(pages: [page])
        XCTAssertEqual(result.jsonResult.count, 1)
        XCTAssertEqual(result.jsonResult[0].count, 1)
        XCTAssertEqual(result.jsonResult[0][0].label, "formula")
        XCTAssertTrue(result.jsonResult[0][0].content?.contains("\\tag{2.1}") ?? false)
    }

    func testTitleFormatting() {
        let formatter = GLMOCRResultFormatter()
        let page: [GLMOCRFormattedRegion] = [
            .init(index: 0, label: "text", bbox2D: nil, content: "My Document", nativeLabel: "doc_title"),
            .init(index: 1, label: "text", bbox2D: nil, content: "- Section 1", nativeLabel: "paragraph_title"),
        ]

        let result = formatter.format(pages: [page])
        XCTAssertEqual(result.jsonResult[0][0].content, "# My Document")
        XCTAssertEqual(result.jsonResult[0][1].content, "## Section 1")
    }

    func testMergesHyphenatedTextBlocks() {
        let formatter = GLMOCRResultFormatter()
        let page: [GLMOCRFormattedRegion] = [
            .init(index: 0, label: "text", bbox2D: [0, 0, 10, 10], content: "inter-", nativeLabel: "text"),
            .init(index: 1, label: "text", bbox2D: [0, 11, 10, 20], content: "national standard", nativeLabel: "text"),
        ]

        let result = formatter.format(pages: [page])
        XCTAssertEqual(result.jsonResult.count, 1)
        XCTAssertEqual(result.jsonResult[0].count, 1)
        XCTAssertEqual(result.jsonResult[0][0].content, "international standard")
    }

    func testFormatsMissingMiddleBulletByAlignment() {
        let formatter = GLMOCRResultFormatter()
        let page: [GLMOCRFormattedRegion] = [
            .init(index: 0, label: "text", bbox2D: [100, 0, 300, 40], content: "- item one", nativeLabel: "text"),
            .init(index: 1, label: "text", bbox2D: [104, 45, 320, 80], content: "item two", nativeLabel: "text"),
            .init(index: 2, label: "text", bbox2D: [102, 85, 330, 120], content: "- item three", nativeLabel: "text"),
        ]

        let result = formatter.format(pages: [page])
        XCTAssertEqual(result.jsonResult.count, 1)
        XCTAssertEqual(result.jsonResult[0].count, 3)
        XCTAssertEqual(result.jsonResult[0][1].content, "- item two")
    }
}
