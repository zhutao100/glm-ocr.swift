import MLX
import XCTest

@testable import GLMOCR

final class GLMOCRModelMergeTests: XCTestCase {
    func testMergeInputEmbedsReplacingTokens_MergesInRowMajorOrder() {
        let imageTokenId = 100

        let inputTokenIds = [
            [1, imageTokenId, imageTokenId, 2],
            [3, imageTokenId, 4, 5],
        ]
        let inputIds = MLXArray(inputTokenIds.flatMap { $0.map(Int32.init) }).reshaped(2, 4)

        let inputEmbedsValues: [Float32] = [
            0.0, 0.1, 0.2,
            1.0, 1.1, 1.2,
            2.0, 2.1, 2.2,
            3.0, 3.1, 3.2,
            4.0, 4.1, 4.2,
            5.0, 5.1, 5.2,
            6.0, 6.1, 6.2,
            7.0, 7.1, 7.2,
        ]
        let inputEmbeds = MLXArray(inputEmbedsValues).reshaped(2, 4, 3)

        let imageFeatureValues: [Float32] = [
            10.0, 10.1, 10.2,
            20.0, 20.1, 20.2,
            30.0, 30.1, 30.2,
        ]
        let imageFeatures = MLXArray(imageFeatureValues).reshaped(3, 3)

        let merged = GLMOCRForConditionalGeneration.mergeInputEmbedsReplacingTokens(
            tokenIdToReplace: imageTokenId,
            expectedTokenCount: 3,
            imageFeatures: imageFeatures,
            inputEmbeds: inputEmbeds,
            inputIds: inputIds
        )

        var expected = inputEmbedsValues
        var featureRow = 0
        for (b, row) in inputTokenIds.enumerated() {
            for (i, token) in row.enumerated() where token == imageTokenId {
                let base = (b * 4 + i) * 3
                expected[base] = imageFeatureValues[featureRow * 3]
                expected[base + 1] = imageFeatureValues[featureRow * 3 + 1]
                expected[base + 2] = imageFeatureValues[featureRow * 3 + 2]
                featureRow += 1
            }
        }
        XCTAssertEqual(featureRow, 3)

        XCTAssertEqual(merged.asArray(Float32.self), expected)
    }

    func testMergeInputEmbedsReplacingTokens_ReplacesVideoTokens() {
        let videoTokenId = 101

        let inputTokenIds = [[videoTokenId, videoTokenId, 2]]
        let inputIds = MLXArray([Int32(videoTokenId), Int32(videoTokenId), 2]).reshaped(1, 3)

        let inputEmbedsValues: [Float32] = [
            0.0, 0.1,
            1.0, 1.1,
            2.0, 2.1,
        ]
        let inputEmbeds = MLXArray(inputEmbedsValues).reshaped(1, 3, 2)

        let imageFeatureValues: [Float32] = [
            9.0, 9.1,
            8.0, 8.1,
        ]
        let imageFeatures = MLXArray(imageFeatureValues).reshaped(2, 2)

        let merged = GLMOCRForConditionalGeneration.mergeInputEmbedsReplacingTokens(
            tokenIdToReplace: videoTokenId,
            expectedTokenCount: 2,
            imageFeatures: imageFeatures,
            inputEmbeds: inputEmbeds,
            inputIds: inputIds
        )

        let expected: [Float32] = [
            9.0, 9.1,
            8.0, 8.1,
            2.0, 2.1,
        ]
        XCTAssertEqual(merged.asArray(Float32.self), expected)
    }
}
