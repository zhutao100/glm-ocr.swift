import XCTest

@testable import GLMOCR

final class GLMOCRDocumentParserTokenBudgetTests: XCTestCase {
    func testNonTextRegionUsesConfiguredBudget() {
        let resolved = GLMOCRDocumentParser.effectiveMaxNewTokensForRegion(
            task: .table,
            nativeLabel: "table",
            bbox2D: [380, 350, 620, 605],
            configuredMaxNewTokens: 4096
        )
        XCTAssertEqual(resolved, 4096)
    }

    func testTitleRegionGetsTightBudget() {
        let resolved = GLMOCRDocumentParser.effectiveMaxNewTokensForRegion(
            task: .text,
            nativeLabel: "figure_title",
            bbox2D: [430, 488, 567, 494],
            configuredMaxNewTokens: 2048
        )
        XCTAssertEqual(resolved, 96)
    }

    func testCompactTextStripGetsTightBudget() {
        let resolved = GLMOCRDocumentParser.effectiveMaxNewTokensForRegion(
            task: .text,
            nativeLabel: "text",
            bbox2D: [386, 559, 615, 577],
            configuredMaxNewTokens: 2048
        )
        XCTAssertEqual(resolved, 96)
    }

    func testMidHeightTextGetsModerateBudget() {
        let resolved = GLMOCRDocumentParser.effectiveMaxNewTokensForRegion(
            task: .text,
            nativeLabel: "text",
            bbox2D: [381, 594, 618, 649],
            configuredMaxNewTokens: 2048
        )
        XCTAssertEqual(resolved, 256)
    }

    func testTallTextPreservesConfiguredBudget() {
        let resolved = GLMOCRDocumentParser.effectiveMaxNewTokensForRegion(
            task: .text,
            nativeLabel: "text",
            bbox2D: [393, 353, 619, 648],
            configuredMaxNewTokens: 2048
        )
        XCTAssertEqual(resolved, 2048)
    }
}
