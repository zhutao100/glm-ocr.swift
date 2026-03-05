import XCTest

@testable import GLMOCR

final class PPDocLayoutV3MaskPolygonExtractorTests: XCTestCase {
    func testRectangleMaskProducesExpectedPolygonCorners() {
        let imageSize = (width: 100, height: 100)
        let box = (x1: 10, y1: 20, x2: 30, y2: 50)

        let maskW = 200
        let maskH = 200
        var maskData = [UInt8](repeating: 0, count: maskW * maskH)

        for y in 40..<100 {
            let row = y * maskW
            for x in 20..<60 {
                maskData[row + x] = 1
            }
        }

        let polygon = PPDocLayoutV3MaskPolygonExtractor.extractPolygonPoints(
            boxPx: box,
            mask: PPDocLayoutV3Mask(width: maskW, height: maskH, data: maskData),
            imageSize: imageSize
        )

        XCTAssertNotNil(polygon)
        guard let polygon else { return }
        XCTAssertEqual(polygon.count, 4)

        let corners = Set(
            polygon.compactMap { pt -> String? in
                guard pt.count >= 2 else { return nil }
                return "\(Int(pt[0].rounded())):\(Int(pt[1].rounded()))"
            }
        )
        XCTAssertEqual(corners, Set(["10:20", "10:49", "29:49", "29:20"]))
    }

    func testEmptyMaskReturnsNilPolygon() {
        let polygon = PPDocLayoutV3MaskPolygonExtractor.extractPolygonPoints(
            boxPx: (x1: 10, y1: 10, x2: 40, y2: 40),
            mask: PPDocLayoutV3Mask(width: 200, height: 200, data: [UInt8](repeating: 0, count: 200 * 200)),
            imageSize: (width: 100, height: 100)
        )
        XCTAssertNil(polygon)
    }

    func testTriangleMaskProducesExpectedPolygonCorners() {
        let imageSize = (width: 20, height: 20)
        let box = (x1: 0, y1: 0, x2: 20, y2: 20)

        let maskW = 20
        let maskH = 20
        var maskData = [UInt8](repeating: 0, count: maskW * maskH)

        for y in 0..<maskH {
            let row = y * maskW
            for x in 0..<maskW where x + y <= 19 {
                maskData[row + x] = 1
            }
        }

        let polygon = PPDocLayoutV3MaskPolygonExtractor.extractPolygonPoints(
            boxPx: box,
            mask: PPDocLayoutV3Mask(width: maskW, height: maskH, data: maskData),
            imageSize: imageSize
        )

        XCTAssertNotNil(polygon)
        guard let polygon else { return }
        XCTAssertEqual(polygon.count, 3)

        let corners = Set(
            polygon.compactMap { pt -> String? in
                guard pt.count >= 2 else { return nil }
                return "\(Int(pt[0].rounded())):\(Int(pt[1].rounded()))"
            }
        )
        XCTAssertEqual(corners, Set(["0:0", "-2:9", "9:-2"]))
    }
}
