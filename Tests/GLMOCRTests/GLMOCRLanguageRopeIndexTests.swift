import XCTest
import MLX
@testable import GLMOCR

final class GLMOCRLanguageRopeIndexTests: XCTestCase {
    private func makeTestConfig(spatialMergeSize: Int = 2) throws -> GLMOCRModelConfig {
        let json = """
        {
          "model_type": "glm_ocr",
          "text_config": {
            "model_type": "glm",
            "vocab_size": 1000,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5
          },
          "vision_config": {
            "model_type": "vision",
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 1,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
            "spatial_merge_size": \(spatialMergeSize),
            "rms_norm_eps": 1e-5
          },
          "image_token_id": 100,
          "video_token_id": 101,
          "image_start_token_id": 102,
          "image_end_token_id": 103,
          "video_start_token_id": 104,
          "video_end_token_id": 105
        }
        """
        let data = Data(json.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(GLMOCRModelConfig.self, from: data)
    }

    func testGetRopeIndex_TextOnly_NoPadding_IsArange() throws {
        let config = try makeTestConfig()
        let inputTokenIds = [
            [1, 2, 3],
            [4, 5, 6],
        ]

        let (positionIds, ropeDeltas) = GLMOCRLanguage.getRopeIndex(
            inputTokenIds: inputTokenIds,
            attentionMask: nil,
            imageGridTHW: nil,
            videoGridTHW: nil,
            config: config
        )

        XCTAssertEqual(positionIds.shape, [3, 2, 3])
        XCTAssertEqual(ropeDeltas.shape, [2, 1])

        let pos = positionIds.asArray(Int32.self)
        let expectedChannel: [Int32] = [0, 1, 2, 0, 1, 2]
        let expected = expectedChannel + expectedChannel + expectedChannel
        XCTAssertEqual(pos, expected)

        XCTAssertEqual(ropeDeltas.asArray(Int32.self), [0, 0])
    }

    func testGetRopeIndex_TextOnly_LeftPadding_UsesCumsumPositions() throws {
        let config = try makeTestConfig()

        let inputTokenIds = [
            [0, 0, 11, 12, 13],
            [0, 21, 22, 23, 24],
        ]
        let attentionMask = [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]

        let (positionIds, ropeDeltas) = GLMOCRLanguage.getRopeIndex(
            inputTokenIds: inputTokenIds,
            attentionMask: attentionMask,
            imageGridTHW: nil,
            videoGridTHW: nil,
            config: config
        )

        XCTAssertEqual(positionIds.shape, [3, 2, 5])
        XCTAssertEqual(ropeDeltas.shape, [2, 1])

        let pos = positionIds.asArray(Int32.self)
        let expectedC0: [Int32] = [
            1, 1, 0, 1, 2,
            1, 0, 1, 2, 3,
        ]
        let expected = expectedC0 + expectedC0 + expectedC0
        XCTAssertEqual(pos, expected)

        XCTAssertEqual(ropeDeltas.asArray(Int32.self), [-2, -1])
    }

    func testGetRopeIndex_ImageAndText_LeftPadding_MatchesTransformersLayout() throws {
        let config = try makeTestConfig(spatialMergeSize: 2)
        let imageTokenId = config.imageTokenId

        let inputTokenIds = [
            [0, 11, imageTokenId, imageTokenId, imageTokenId, imageTokenId, 12],
            [21, imageTokenId, imageTokenId, imageTokenId, imageTokenId, 22, 23],
        ]
        let attentionMask = [
            [0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
        let grids: [(t: Int, h: Int, w: Int)] = [
            (t: 1, h: 4, w: 4),
            (t: 1, h: 4, w: 4),
        ]

        let (positionIds, ropeDeltas) = GLMOCRLanguage.getRopeIndex(
            inputTokenIds: inputTokenIds,
            attentionMask: attentionMask,
            imageGridTHW: grids,
            videoGridTHW: nil,
            config: config
        )

        XCTAssertEqual(positionIds.shape, [3, 2, 7])
        XCTAssertEqual(ropeDeltas.shape, [2, 1])

        let pos = positionIds.asArray(Int32.self)
        let expectedC0: [Int32] = [
            1, 0, 1, 1, 1, 1, 3,
            0, 1, 1, 1, 1, 3, 4,
        ]
        let expectedC1: [Int32] = [
            1, 0, 1, 1, 2, 2, 3,
            0, 1, 1, 2, 2, 3, 4,
        ]
        let expectedC2: [Int32] = [
            1, 0, 1, 2, 1, 2, 3,
            0, 1, 2, 1, 2, 3, 4,
        ]
        let expected = expectedC0 + expectedC1 + expectedC2
        XCTAssertEqual(pos, expected)

        XCTAssertEqual(ropeDeltas.asArray(Int32.self), [-3, -2])
    }
}
