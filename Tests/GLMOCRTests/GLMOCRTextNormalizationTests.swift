import XCTest

@testable import GLMOCR

final class GLMOCRTextNormalizationTests: XCTestCase {
    func testCleanContentStripsLiteralTabsAndClampsPunctuation() {
        XCTAssertEqual(
            GLMOCRTextNormalization.cleanContent("\\t\\tHello......\\t"),
            "Hello..."
        )
    }

    func testNormalizeTextFormatsBullets() {
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("· item"), "- item")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("• item"), "- item")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("* item"), "- item")
    }

    func testNormalizeTextFormatsNumberingPrefixes() {
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("（1）foo"), "（1） foo")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("A)foo"), "A) foo")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("1）foo"), "1) foo")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("2.foo"), "2. foo")
    }

    func testNormalizeTextReplacesSingleNewlinesWithDoubleNewlines() {
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("a\nb"), "a\n\nb")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("a\n\nb"), "a\n\nb")
    }

    func testNormalizeTextStripsSpuriousEmphasis() {
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("*abc*"), "abc")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("x *a-b* y"), "x a-b y")
        XCTAssertEqual(GLMOCRTextNormalization.normalizeText("*a b*"), "*a b*")
    }
}
