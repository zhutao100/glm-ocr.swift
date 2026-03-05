import CoreGraphics
import Foundation
import MLX
import Tokenizers

public final class GLMOCRPipeline {
    public let resources: GLMOCRResources
    var model: GLMOCRForConditionalGeneration?
    private lazy var imageProcessor = GLMOCRImageProcessor(config: resources.preprocessorConfig)
    private lazy var eosTokenIds = Self.eosTokenIds(resources: resources)

    private struct PreparedBatchItem {
        let promptTokenIds: [Int]
        let promptLength: Int
        let imageGridTHW: (t: Int, h: Int, w: Int)
        let pixelValues: MLXArray
    }

    public init(modelURL: URL, strictTokenizer: Bool = true) async throws {
        self.resources = try await GLMOCRResources(modelURL: modelURL, strictTokenizer: strictTokenizer)
    }

    public func loadModel(dtype: MLX.DType? = nil) throws {
        if model != nil { return }
        try withGPUDevice {
            model = try GLMOCRModelLoader.load(
                from: resources.modelURL, config: resources.modelConfig, dtype: dtype ?? .float16)
        }
    }

    public func tokenize(messages: [Message]) throws -> [Int] {
        try resources.tokenizer.applyChatTemplate(messages: messages)
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        resources.tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    public func recognize(
        imageAt imagePath: String,
        prompt: String = GLMOCRPromptPresets.textRecognition,
        maxNewTokens: Int = 256,
        generationParameters: GLMOCRGenerationParameters = .greedy,
        skipSpecialTokens: Bool = false,
        dtypeOverride: MLX.DType? = nil,
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> String {
        try withGPUDevice {
            try loadModel(dtype: dtypeOverride)
            guard let model else {
                throw GLMOCRPipelineError.modelNotLoaded
            }

            let expandedImagePath = (imagePath as NSString).expandingTildeInPath

            let tokenIds = GLMOCRPrompt.tokenizeOCRPrompt(
                tokenizer: resources.tokenizer,
                prompt: prompt
            )

            let processedImage = try imageProcessor.process(
                imageAt: URL(fileURLWithPath: expandedImagePath),
                postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
            )

            let expandedTokenIds = try GLMOCRPrompt.expandImageTokens(
                inputIDs: tokenIds,
                imageTokenId: resources.modelConfig.imageTokenId,
                imageGridTHW: [processedImage.imageGridTHW],
                mergeSize: resources.preprocessorConfig.mergeSize
            )

            let visionDType = model.model.visual.patchEmbed.proj.weight.dtype
            let pixelValues = processedImage.pixelValues.asType(visionDType)

            let generatedTokenIds = GLMOCRGenerator.generate(
                model: model,
                config: resources.modelConfig,
                promptTokenIds: expandedTokenIds,
                pixelValues: pixelValues,
                imageGridTHW: [processedImage.imageGridTHW],
                maxNewTokens: maxNewTokens,
                eosTokenIds: eosTokenIds,
                parameters: generationParameters
            )

            let suffix = Array(generatedTokenIds.dropFirst(expandedTokenIds.count))
            return decode(tokens: suffix, skipSpecialTokens: skipSpecialTokens)
        }
    }

    public func recognizeBatch(
        imagePaths: [String],
        prompt: String = GLMOCRPromptPresets.textRecognition,
        maxNewTokens: Int = 256,
        generationParameters: GLMOCRGenerationParameters = .greedy,
        maxNewTokensPerImage: [Int]? = nil,
        skipSpecialTokens: Bool = false,
        dtypeOverride: MLX.DType? = nil,
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> [String] {
        try withGPUDevice {
            try loadModel(dtype: dtypeOverride)
            guard let model else {
                throw GLMOCRPipelineError.modelNotLoaded
            }

            precondition(!imagePaths.isEmpty, "imagePaths must be non-empty")

            let visionDType = model.model.visual.patchEmbed.proj.weight.dtype

            var expandedPaths: [String] = []
            expandedPaths.reserveCapacity(imagePaths.count)
            for path in imagePaths {
                expandedPaths.append((path as NSString).expandingTildeInPath)
            }

            let baseTokenIds = GLMOCRPrompt.tokenizeOCRPrompt(
                tokenizer: resources.tokenizer,
                prompt: prompt
            )

            var items: [PreparedBatchItem] = []
            items.reserveCapacity(expandedPaths.count)

            let imageURLs = expandedPaths.map { URL(fileURLWithPath: $0) }
            let processedImages = try imageProcessor.process(
                imageURLs: imageURLs,
                postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
            )

            for processedImage in processedImages {
                let expandedTokenIds = try GLMOCRPrompt.expandImageTokens(
                    inputIDs: baseTokenIds,
                    imageTokenId: resources.modelConfig.imageTokenId,
                    imageGridTHW: [processedImage.imageGridTHW],
                    mergeSize: resources.preprocessorConfig.mergeSize
                )

                items.append(
                    PreparedBatchItem(
                        promptTokenIds: expandedTokenIds,
                        promptLength: expandedTokenIds.count,
                        imageGridTHW: processedImage.imageGridTHW,
                        pixelValues: processedImage.pixelValues.asType(visionDType)
                    )
                )
            }

            return try recognizePreparedBatch(
                model: model,
                items: items,
                maxNewTokens: maxNewTokens,
                parameters: generationParameters,
                maxNewTokensPerItem: maxNewTokensPerImage,
                skipSpecialTokens: skipSpecialTokens
            )
        }
    }

    public func recognizeBatch(
        images: [CGImage],
        prompt: String = GLMOCRPromptPresets.textRecognition,
        maxNewTokens: Int = 256,
        generationParameters: GLMOCRGenerationParameters = .greedy,
        maxNewTokensPerImage: [Int]? = nil,
        skipSpecialTokens: Bool = false,
        dtypeOverride: MLX.DType? = nil,
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> [String] {
        try withGPUDevice {
            try loadModel(dtype: dtypeOverride)
            guard let model else {
                throw GLMOCRPipelineError.modelNotLoaded
            }

            precondition(!images.isEmpty, "images must be non-empty")

            let visionDType = model.model.visual.patchEmbed.proj.weight.dtype

            let baseTokenIds = GLMOCRPrompt.tokenizeOCRPrompt(
                tokenizer: resources.tokenizer,
                prompt: prompt
            )

            var items: [PreparedBatchItem] = []
            items.reserveCapacity(images.count)

            let processedImages = try imageProcessor.process(
                images,
                postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
            )

            for processedImage in processedImages {
                let expandedTokenIds = try GLMOCRPrompt.expandImageTokens(
                    inputIDs: baseTokenIds,
                    imageTokenId: resources.modelConfig.imageTokenId,
                    imageGridTHW: [processedImage.imageGridTHW],
                    mergeSize: resources.preprocessorConfig.mergeSize
                )

                items.append(
                    PreparedBatchItem(
                        promptTokenIds: expandedTokenIds,
                        promptLength: expandedTokenIds.count,
                        imageGridTHW: processedImage.imageGridTHW,
                        pixelValues: processedImage.pixelValues.asType(visionDType)
                    )
                )
            }

            return try recognizePreparedBatch(
                model: model,
                items: items,
                maxNewTokens: maxNewTokens,
                parameters: generationParameters,
                maxNewTokensPerItem: maxNewTokensPerImage,
                skipSpecialTokens: skipSpecialTokens
            )
        }
    }

    public func recognize(
        image: CGImage,
        prompt: String = GLMOCRPromptPresets.textRecognition,
        maxNewTokens: Int = 256,
        generationParameters: GLMOCRGenerationParameters = .greedy,
        skipSpecialTokens: Bool = false,
        dtypeOverride: MLX.DType? = nil,
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> String {
        try withGPUDevice {
            try loadModel(dtype: dtypeOverride)
            guard let model else {
                throw GLMOCRPipelineError.modelNotLoaded
            }

            let tokenIds = GLMOCRPrompt.tokenizeOCRPrompt(
                tokenizer: resources.tokenizer,
                prompt: prompt
            )

            let processedImage = try imageProcessor.process(
                image,
                postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
            )

            let expandedTokenIds = try GLMOCRPrompt.expandImageTokens(
                inputIDs: tokenIds,
                imageTokenId: resources.modelConfig.imageTokenId,
                imageGridTHW: [processedImage.imageGridTHW],
                mergeSize: resources.preprocessorConfig.mergeSize
            )

            let visionDType = model.model.visual.patchEmbed.proj.weight.dtype
            let pixelValues = processedImage.pixelValues.asType(visionDType)

            let generatedTokenIds = GLMOCRGenerator.generate(
                model: model,
                config: resources.modelConfig,
                promptTokenIds: expandedTokenIds,
                pixelValues: pixelValues,
                imageGridTHW: [processedImage.imageGridTHW],
                maxNewTokens: maxNewTokens,
                eosTokenIds: eosTokenIds,
                parameters: generationParameters
            )

            let suffix = Array(generatedTokenIds.dropFirst(expandedTokenIds.count))
            return decode(tokens: suffix, skipSpecialTokens: skipSpecialTokens)
        }
    }

    private func withGPUDevice<R>(_ operation: () throws -> R) rethrows -> R {
        try Device.withDefaultDevice(.gpu) {
            try operation()
        }
    }

    private func recognizePreparedBatch(
        model: GLMOCRForConditionalGeneration,
        items: [PreparedBatchItem],
        maxNewTokens: Int,
        parameters: GLMOCRGenerationParameters,
        maxNewTokensPerItem: [Int]? = nil,
        skipSpecialTokens: Bool
    ) throws -> [String] {
        let batch = items.count
        precondition(batch > 0, "items must be non-empty")
        if let maxNewTokensPerItem {
            precondition(maxNewTokensPerItem.count == batch, "maxNewTokensPerItem batch mismatch")
        }

        let promptLengths = items.map(\.promptLength)
        let promptLenMax = promptLengths.max() ?? 0
        precondition(promptLenMax > 0, "items must have non-empty prompts")

        let cacheDType = model.model.languageModel.norm.weight.dtype
        let budgetBytes = Self.kvCacheBudgetBytes()
        let fullBatchBytes = Self.estimateKVCacheBytes(
            config: resources.modelConfig,
            cacheDType: cacheDType,
            batch: batch,
            promptLen: promptLenMax,
            maxNewTokens: maxNewTokens
        )

        let microbatches: [[Int]] = {
            if fullBatchBytes <= budgetBytes {
                return [Array(0..<batch)]
            }
            return Self.microbatchIndices(
                promptLengths: promptLengths,
                config: resources.modelConfig,
                cacheDType: cacheDType,
                maxNewTokens: maxNewTokens,
                budgetBytes: budgetBytes
            )
        }()

        var outputs = Array(repeating: "", count: batch)

        let needsMicrobatchSeedDerivation =
            (parameters.temperature > 0
                && parameters.seed != nil
                && microbatches.count > 1)

        for (microbatchIndex, microbatch) in microbatches.enumerated() {
            let microItems = microbatch.map { items[$0] }
            let microPromptTokenIds = microItems.map(\.promptTokenIds)
            let microPromptLengths = microItems.map(\.promptLength)
            let microGrids = microItems.map(\.imageGridTHW)
            let microPixelValues = concatenated(microItems.map(\.pixelValues), axis: 0)
            let microMaxNewTokensPerSequence = maxNewTokensPerItem.map { maxByItem in
                microbatch.map { maxByItem[$0] }
            }

            var microParams = parameters
            if needsMicrobatchSeedDerivation, let seed = parameters.seed {
                microParams.seed = seed &+ UInt64(microbatchIndex)
            }

            let generatedTokenIds = GLMOCRGenerator.generateBatch(
                model: model,
                config: resources.modelConfig,
                promptTokenIds: microPromptTokenIds,
                pixelValues: microPixelValues,
                imageGridTHW: microGrids,
                maxNewTokens: maxNewTokens,
                eosTokenIds: eosTokenIds,
                parameters: microParams,
                maxNewTokensBySequence: microMaxNewTokensPerSequence
            )
            precondition(generatedTokenIds.count == microbatch.count, "internal error: microbatch size mismatch")

            for i in 0..<microbatch.count {
                let originalIndex = microbatch[i]
                let suffix = Array(generatedTokenIds[i].dropFirst(microPromptLengths[i]))
                outputs[originalIndex] = decode(tokens: suffix, skipSpecialTokens: skipSpecialTokens)
            }
        }

        return outputs
    }

    private static func kvCacheBudgetBytes() -> Int64 {
        let cacheLimit = Memory.cacheLimit
        let memoryLimit = Memory.memoryLimit
        let limit = cacheLimit > 0 ? cacheLimit : memoryLimit

        let scaled = Int64(Double(limit) * 0.75)
        let minimum = Int64(256 * 1024 * 1024)
        return max(scaled, minimum)
    }

    private static func estimateKVCacheBytes(
        config: GLMOCRModelConfig,
        cacheDType: DType,
        batch: Int,
        promptLen: Int,
        maxNewTokens: Int
    ) -> Int64 {
        let headDim = config.textConfig.headDim ?? (config.textConfig.hiddenSize / config.textConfig.numAttentionHeads)
        let maxLength = promptLen + maxNewTokens + 1

        let layers = Int64(config.textConfig.numHiddenLayers)
        let b = Int64(batch)
        let kvHeads = Int64(config.textConfig.numKeyValueHeads)
        let h = Int64(headDim)
        let l = Int64(maxLength)
        let bytes = Int64(cacheDType.size)

        return layers * 2 * b * kvHeads * l * h * bytes
    }

    private static func microbatchIndices(
        promptLengths: [Int],
        config: GLMOCRModelConfig,
        cacheDType: DType,
        maxNewTokens: Int,
        budgetBytes: Int64
    ) -> [[Int]] {
        precondition(!promptLengths.isEmpty)
        let sorted = (0..<promptLengths.count).sorted { promptLengths[$0] > promptLengths[$1] }

        var out: [[Int]] = []
        out.reserveCapacity(sorted.count)

        var current: [Int] = []
        current.reserveCapacity(sorted.count)

        var currentPromptLenMax = 0

        for idx in sorted {
            let len = promptLengths[idx]

            if current.isEmpty {
                current.append(idx)
                currentPromptLenMax = len
                continue
            }

            let candidatePromptLenMax = max(currentPromptLenMax, len)
            let candidateBatch = current.count + 1
            let requiredBytes = estimateKVCacheBytes(
                config: config,
                cacheDType: cacheDType,
                batch: candidateBatch,
                promptLen: candidatePromptLenMax,
                maxNewTokens: maxNewTokens
            )
            if requiredBytes <= budgetBytes {
                current.append(idx)
                currentPromptLenMax = candidatePromptLenMax
                continue
            }

            out.append(current)
            current = [idx]
            currentPromptLenMax = len
        }

        if !current.isEmpty {
            out.append(current)
        }

        return out
    }

    private static func eosTokenIds(resources: GLMOCRResources) -> [Int] {
        var eos = resources.generationConfig.eosTokenId?.values ?? resources.modelConfig.textConfig.eosTokenId ?? []

        let pad = resources.generationConfig.padTokenId ?? resources.modelConfig.textConfig.padTokenId
        if let pad, !eos.contains(pad) {
            eos.append(pad)
        }
        return eos
    }
}

enum GLMOCRPipelineError: Error, Sendable {
    case modelNotLoaded
}
