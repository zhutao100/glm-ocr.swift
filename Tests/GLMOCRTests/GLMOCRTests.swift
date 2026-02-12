import XCTest
@testable import GLMOCR

final class GLMOCRTests: XCTestCase {
    func testRecognizeNotImplemented() async {
        do {
            _ = try await GLMOCR().recognize(imageAt: URL(fileURLWithPath: "/dev/null"))
            XCTFail("Expected GLMOCRNotImplementedError")
        } catch is GLMOCRNotImplementedError {
            XCTAssertTrue(true)
        } catch {
            XCTFail("Unexpected error: \\(error)")
        }
    }
}
