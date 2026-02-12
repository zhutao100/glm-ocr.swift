import XCTest
@testable import GLMOCR

final class GLMOCRPolygonMaskRasterizerTests: XCTestCase {
    func testRasterizePolygonMaskSquareBoundary() {
        let width = 5
        let height = 5
        let points = [(1, 1), (4, 1), (4, 4), (1, 4)]

        let mask = GLMOCRDocumentParser.rasterizePolygonMask(
            width: width,
            height: height,
            points: points
        )

        XCTAssertEqual(mask.count, width * height)
        for y in 0..<height {
            for x in 0..<width {
                let inside = (1...4).contains(x) && (1...4).contains(y)
                XCTAssertEqual(mask[y * width + x], inside ? 1 : 0, "mask[\(x),\(y)]")
            }
        }
    }

    func testRasterizePolygonMaskClipsOutOfBounds() {
        let width = 4
        let height = 4
        let points = [(-1, -1), (2, -1), (2, 2), (-1, 2)]

        let mask = GLMOCRDocumentParser.rasterizePolygonMask(
            width: width,
            height: height,
            points: points
        )

        let expected: [UInt8] = [
            1, 1, 1, 0,
            1, 1, 1, 0,
            1, 1, 1, 0,
            0, 0, 0, 0
        ]

        XCTAssertEqual(mask, expected)
    }

    func testApplyPolygonMaskWhitesOutside() {
        let width = 5
        let height = 5
        let points = [(1, 1), (4, 1), (4, 4), (1, 4)]
        var rgba = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                rgba[offset] = UInt8(x)
                rgba[offset + 1] = UInt8(y)
                rgba[offset + 2] = 10
                rgba[offset + 3] = 255
            }
        }

        rgba.withUnsafeMutableBufferPointer { buffer in
            GLMOCRDocumentParser.applyPolygonMask(
                width: width,
                height: height,
                rgba: buffer,
                points: points
            )
        }

        XCTAssertEqual(rgba[0], 255)
        XCTAssertEqual(rgba[1], 255)
        XCTAssertEqual(rgba[2], 255)
        XCTAssertEqual(rgba[3], 255)

        let insideOffset = (2 * width + 2) * 4
        XCTAssertEqual(rgba[insideOffset], 2)
        XCTAssertEqual(rgba[insideOffset + 1], 2)
        XCTAssertEqual(rgba[insideOffset + 2], 10)
        XCTAssertEqual(rgba[insideOffset + 3], 255)
    }
}
