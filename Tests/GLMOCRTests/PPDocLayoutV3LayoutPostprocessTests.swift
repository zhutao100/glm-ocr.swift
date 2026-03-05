import XCTest

@testable import GLMOCR

final class PPDocLayoutV3LayoutPostprocessTests: XCTestCase {
    func testNMSPrefersHigherScoreSameClass() throws {
        let raw = PPDocLayoutV3RawDetections(
            scores: [0.9, 0.8, 0.95],
            labels: [1, 1, 2],
            boxes: [
                [0, 0, 100, 100],
                [10, 10, 110, 110],
                [10, 10, 110, 110],
            ],
            orderSeq: [1, 2, 3]
        )

        let id2label = [1: "content", 2: "table"]
        let out = PPDocLayoutV3LayoutPostprocess.apply(
            raw: raw,
            id2label: id2label,
            imageSize: (width: 1000, height: 1000),
            layoutNMS: true,
            layoutUnclipRatio: nil,
            layoutMergeMode: nil,
            mergeModeByClass: nil
        )

        XCTAssertEqual(out.count, 2)
        XCTAssertTrue(out.contains(where: { $0.clsId == 1 && abs($0.score - 0.9) < 1e-9 }))
        XCTAssertTrue(out.contains(where: { $0.clsId == 2 && abs($0.score - 0.95) < 1e-9 }))
    }

    func testFilterLargeImageDropsOversizedImageBox() throws {
        let raw = PPDocLayoutV3RawDetections(
            scores: [0.9, 0.8],
            labels: [14, 22],
            boxes: [
                [0, 0, 100, 100],
                [10, 10, 20, 20],
            ],
            orderSeq: [1, 2]
        )

        let id2label = [14: "image", 22: "text"]
        let out = PPDocLayoutV3LayoutPostprocess.apply(
            raw: raw,
            id2label: id2label,
            imageSize: (width: 100, height: 100),
            layoutNMS: false,
            layoutUnclipRatio: nil,
            layoutMergeMode: nil,
            mergeModeByClass: nil
        )

        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].clsId, 22)
    }

    func testMergeLargeDropsContainedBoxes() throws {
        let raw = PPDocLayoutV3RawDetections(
            scores: [0.9, 0.8],
            labels: [22, 21],
            boxes: [
                [0, 0, 100, 100],
                [10, 10, 20, 20],
            ],
            orderSeq: [1, 2]
        )

        let id2label = [21: "table", 22: "text"]
        let out = PPDocLayoutV3LayoutPostprocess.apply(
            raw: raw,
            id2label: id2label,
            imageSize: (width: 100, height: 100),
            layoutNMS: false,
            layoutUnclipRatio: nil,
            layoutMergeMode: .large,
            mergeModeByClass: nil
        )

        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].clsId, 22)
    }

    func testUnclipExpandsBoxes() throws {
        let raw = PPDocLayoutV3RawDetections(
            scores: [0.9],
            labels: [22],
            boxes: [
                [10, 10, 30, 30]
            ],
            orderSeq: [1]
        )

        let id2label = [22: "text"]
        let out = PPDocLayoutV3LayoutPostprocess.apply(
            raw: raw,
            id2label: id2label,
            imageSize: (width: 100, height: 100),
            layoutNMS: false,
            layoutUnclipRatio: (width: 2.0, height: 2.0),
            layoutMergeMode: nil,
            mergeModeByClass: nil
        )

        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].coordinate, [0, 0, 40, 40])
    }
}
