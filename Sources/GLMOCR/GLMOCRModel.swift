import Foundation
import MLX
import MLXNN

final class GLMOCRModelInner: Module {
    @ModuleInfo(key: "language_model") var languageModel: GLMOCRTextModel
    @ModuleInfo(key: "visual") var visual: GLMOCRVisionModel

    let config: GLMOCRModelConfig

    init(config: GLMOCRModelConfig) {
        self.config = config
        _languageModel.wrappedValue = GLMOCRTextModel(config: config.textConfig)
        _visual.wrappedValue = GLMOCRVisionModel(config: config.visionConfig)
        super.init()
    }
}

final class GLMOCRForConditionalGeneration: Module {
    @ModuleInfo(key: "model") var model: GLMOCRModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: GLMOCRModelConfig

    init(config: GLMOCRModelConfig) {
        self.config = config
        _model.wrappedValue = GLMOCRModelInner(config: config)
        _lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        super.init()
    }

    func forward(
        inputIds: MLXArray,
        inputEmbeddings: MLXArray?,
        cache: [GLMOCRKVCache]?,
        positionIds: MLXArray?,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let hidden = model.languageModel(
            inputIds: inputIds,
            inputEmbeddings: inputEmbeddings,
            cache: cache,
            positionIds: positionIds,
            attentionMask: attentionMask
        )
        return lmHead(hidden)
    }

    static func mergeInputEmbedsReplacingTokens(
        tokenIdToReplace: Int,
        expectedTokenCount: Int,
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        precondition(inputIds.ndim == 2, "inputIds must be rank-2 [B, L]")
        precondition(inputEmbeds.ndim == 3, "inputEmbeds must be rank-3 [B, L, H]")
        precondition(imageFeatures.ndim == 2, "imageFeatures must be rank-2 [N, H]")

        let batch = inputIds.dim(0)
        let seqLen = inputIds.dim(1)
        let hidden = inputEmbeds.dim(2)
        precondition(inputEmbeds.dim(0) == batch, "inputEmbeds batch mismatch")
        precondition(inputEmbeds.dim(1) == seqLen, "inputEmbeds length mismatch")

        let expectedCount = expectedTokenCount

        if expectedCount == 0 {
            return inputEmbeds
        }
        precondition(
            expectedCount == imageFeatures.dim(0),
            "Image feature/token mismatch (features=\(imageFeatures.dim(0)) expected=\(expectedCount))"
        )

        let maskFlat = (inputIds .== MLXArray(tokenIdToReplace)).reshaped(batch * seqLen)
        let flatEmbeds = inputEmbeds.reshaped(batch * seqLen, hidden)

        let featureIndex = cumsum(maskFlat.asType(.int32), axis: 0) - MLXArray(Int32(1))
        let safeIndex = which(maskFlat, featureIndex, zeros([batch * seqLen], dtype: .int32))
        let gathered = imageFeatures[safeIndex]

        let outFlat = which(maskFlat.reshaped(batch * seqLen, 1), gathered, flatEmbeds)
        return outFlat.reshaped(batch, seqLen, hidden)
    }
}
