import MLX
import XCTest

@testable import GLMOCR

final class GLMOCRWeightsLoaderTests: XCTestCase {
    func testSanitizeTorchConvWeightsTransposesKnownKeys() {
        let conv3d = MLXArray(0..<24, [2, 3, 1, 2, 2]).asType(.float32)
        let conv2d = MLXArray(0..<32, [2, 4, 2, 2]).asType(.float32)

        let weights: [String: MLXArray] = [
            "model.visual.patch_embed.proj.weight": conv3d,
            "model.visual.downsample.weight": conv2d,
        ]

        let sanitized = GLMOCRWeightsLoader.sanitizeTorchConvWeights(weights)

        XCTAssertTrue(
            allClose(sanitized["model.visual.patch_embed.proj.weight"]!, conv3d.transposed(0, 2, 3, 4, 1)).item())
        XCTAssertTrue(allClose(sanitized["model.visual.downsample.weight"]!, conv2d.transposed(0, 2, 3, 1)).item())
    }
}
