import Foundation
import MLX

enum GLMOCRGenerator {
    private static let maxPrefillChunkLength = 8191
    private static let adaptiveTopPInitialTopK = 256

    static func generate(
        model: GLMOCRForConditionalGeneration,
        config: GLMOCRModelConfig,
        promptTokenIds: [Int],
        pixelValues: MLXArray,
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        maxNewTokens: Int,
        eosTokenIds: [Int],
        parameters: GLMOCRGenerationParameters,
        maxNewTokensForSequence: Int? = nil
    ) -> [Int] {
        let maxNewTokensBySequence: [Int]? = {
            if let maxNewTokensForSequence {
                return [maxNewTokensForSequence]
            }
            return nil
        }()
        let outputs = generateBatch(
            model: model,
            config: config,
            promptTokenIds: [promptTokenIds],
            pixelValues: pixelValues,
            imageGridTHW: imageGridTHW,
            maxNewTokens: maxNewTokens,
            eosTokenIds: eosTokenIds,
            parameters: parameters,
            maxNewTokensBySequence: maxNewTokensBySequence
        )
        precondition(outputs.count == 1, "internal error: expected batch size 1")
        return outputs[0]
    }

    static func greedyGenerate(
        model: GLMOCRForConditionalGeneration,
        config: GLMOCRModelConfig,
        promptTokenIds: [Int],
        pixelValues: MLXArray,
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        maxNewTokens: Int,
        eosTokenIds: [Int]
    ) -> [Int] {
        generate(
            model: model,
            config: config,
            promptTokenIds: promptTokenIds,
            pixelValues: pixelValues,
            imageGridTHW: imageGridTHW,
            maxNewTokens: maxNewTokens,
            eosTokenIds: eosTokenIds,
            parameters: .greedy
        )
    }

    static func generateBatch(
        model: GLMOCRForConditionalGeneration,
        config: GLMOCRModelConfig,
        promptTokenIds: [[Int]],
        pixelValues: MLXArray,
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        maxNewTokens: Int,
        eosTokenIds: [Int],
        parameters: GLMOCRGenerationParameters,
        maxNewTokensBySequence: [Int]? = nil
    ) -> [[Int]] {
        return Stream.withNewDefaultStream {
        precondition(!promptTokenIds.isEmpty, "promptTokenIds must be non-empty (batch)")
        precondition(promptTokenIds.allSatisfy { !$0.isEmpty }, "promptTokenIds must be non-empty (per item)")
        precondition(maxNewTokens >= 0, "maxNewTokens must be >= 0")
        precondition(!eosTokenIds.isEmpty, "eosTokenIds must be non-empty")

        let batch = promptTokenIds.count
        let sequenceMaxNewTokens: [Int] = {
            if let maxNewTokensBySequence {
                precondition(maxNewTokensBySequence.count == batch, "maxNewTokensBySequence batch mismatch")
                precondition(maxNewTokensBySequence.allSatisfy { $0 >= 0 }, "maxNewTokensBySequence must be >= 0")
                precondition(maxNewTokensBySequence.allSatisfy { $0 <= maxNewTokens }, "maxNewTokensBySequence must be <= maxNewTokens")
                return maxNewTokensBySequence
            }
            return Array(repeating: maxNewTokens, count: batch)
        }()
        if maxNewTokens == 0 || sequenceMaxNewTokens.allSatisfy({ $0 == 0 }) {
            return promptTokenIds
        }

        let promptLengths = promptTokenIds.map(\.count)
        let promptLen = promptLengths.max() ?? 0
        precondition(promptLen > 0, "promptTokenIds must be non-empty (seqLen)")

        let padTokenId = config.textConfig.padTokenId ?? eosTokenIds[0]

        var paddedTokenIds: [[Int]] = []
        paddedTokenIds.reserveCapacity(batch)
        var promptAttentionMask: [[Int]] = []
        promptAttentionMask.reserveCapacity(batch)
        var padLens: [Int] = []
        padLens.reserveCapacity(batch)

        for (tokens, len) in zip(promptTokenIds, promptLengths) {
            let padLen = promptLen - len
            padLens.append(padLen)
            paddedTokenIds.append(Array(repeating: padTokenId, count: padLen) + tokens)
            promptAttentionMask.append(Array(repeating: 0, count: padLen) + Array(repeating: 1, count: len))
        }

        let needsPaddingMask = padLens.contains(where: { $0 > 0 })
        let maxLength = promptLen + maxNewTokens + 1
        let decodeSyncInterval = max(0, parameters.decodeSyncInterval)

        let attentionMask: MLXArray? = {
            guard needsPaddingMask else { return nil }
            var attentionMaskFlat: [Int32] = []
            attentionMaskFlat.reserveCapacity(batch * maxLength)
            for padLen in padLens {
                attentionMaskFlat.append(contentsOf: Array(repeating: Int32(0), count: padLen))
                attentionMaskFlat.append(contentsOf: Array(repeating: Int32(1), count: maxLength - padLen))
            }
            return MLXArray(attentionMaskFlat).reshaped(batch, maxLength)
        }()

        var promptIdsFlat: [Int32] = []
        promptIdsFlat.reserveCapacity(batch * promptLen)
        for row in paddedTokenIds {
            promptIdsFlat.append(contentsOf: row.map(Int32.init))
        }
        let promptIds = MLXArray(promptIdsFlat).reshaped(batch, promptLen)

        let (positionIds, ropeDeltas) = GLMOCRLanguage.getRopeIndex(
            inputTokenIds: paddedTokenIds,
            attentionMask: needsPaddingMask ? promptAttentionMask : nil,
            imageGridTHW: imageGridTHW,
            videoGridTHW: nil,
            config: config
        )

        let headDim = config.textConfig.headDim ?? (config.textConfig.hiddenSize / config.textConfig.numAttentionHeads)
        let cacheDType = model.model.languageModel.norm.weight.dtype
        let cacheStep = 1024
        let initialCacheCapacity = min(maxLength, ((promptLen + cacheStep - 1) / cacheStep) * cacheStep)
        let caches: [GLMOCRKVCache] = (0..<config.textConfig.numHiddenLayers).map { _ in
            GLMOCRKVCache(
                batch: batch,
                kvHeads: config.textConfig.numKeyValueHeads,
                headDim: headDim,
                maxLength: maxLength,
                dtype: cacheDType,
                step: cacheStep,
                initialCapacity: initialCacheCapacity
            )
        }

        let tokenEmbeds = model.model.languageModel.embedTokens(promptIds)
        let visionHidden = model.model.visual(pixelValues, gridTHW: imageGridTHW).asType(tokenEmbeds.dtype)
        var imageTokenCount = 0
        var videoTokenCount = 0
        for row in paddedTokenIds {
            for token in row {
                if token == config.imageTokenId {
                    imageTokenCount += 1
                } else if token == config.videoTokenId {
                    videoTokenCount += 1
                }
            }
        }
        let tokenIdToReplace = imageTokenCount > 0 ? config.imageTokenId : config.videoTokenId
        let expectedTokenCount = imageTokenCount > 0 ? imageTokenCount : videoTokenCount

        let mergedEmbeddings = GLMOCRForConditionalGeneration.mergeInputEmbedsReplacingTokens(
            tokenIdToReplace: tokenIdToReplace,
            expectedTokenCount: expectedTokenCount,
            imageFeatures: visionHidden,
            inputEmbeds: tokenEmbeds,
            inputIds: promptIds
        )

        let lastHidden: MLXArray = {
            var idx = 0
            var currentHidden: MLXArray? = nil
            while idx < promptLen {
                let chunkLen = min(maxPrefillChunkLength, promptLen - idx)
                let hidden = model.model.languageModel(
                    inputIds: promptIds[0..., idx ..< idx + chunkLen],
                    inputEmbeddings: mergedEmbeddings[0..., idx ..< idx + chunkLen, 0...],
                    cache: caches,
                    positionIds: positionIds[0..., 0..., idx ..< idx + chunkLen],
                    attentionMask: attentionMask
                )
                currentHidden = hidden[0..., chunkLen - 1 ..< chunkLen, 0...]
                idx += chunkLen
            }
            precondition(currentHidden != nil, "internal error: missing last hidden state")
            return currentHidden!
        }()

        let randomState: MLXRandom.RandomState? = {
            guard parameters.temperature > 0 else { return nil }
            if let seed = parameters.seed {
                return MLXRandom.RandomState(seed: seed)
            }
            return MLXRandom.RandomState()
        }()

        let useRepetitionPenalty = parameters.repetitionPenalty != 1.0
        let batchIndicesForRepetition = useRepetitionPenalty ? arange(batch, dtype: .int32) : nil
        var repetitionTokenMask: MLXArray? = {
            guard useRepetitionPenalty else { return nil }

            let vocabSize = config.textConfig.vocabSize
            precondition(vocabSize > 0, "vocabSize must be > 0 when repetition penalty is enabled")

            var maskFlat = Array(repeating: UInt8(0), count: batch * vocabSize)
            for b in 0..<batch {
                for token in promptTokenIds[b] {
                    precondition(token >= 0 && token < vocabSize, "prompt token id out of vocab bounds")
                    maskFlat[b * vocabSize + token] = 1
                }
            }
            return MLXArray(maskFlat).reshaped(batch, vocabSize)
        }()

        var outputs = promptTokenIds
        let eosPrimary = Int32(eosTokenIds[0])
        let eosVector = MLXArray(Array(repeating: eosPrimary, count: batch))
        let sequenceLimits = MLXArray(sequenceMaxNewTokens.map(Int32.init))

        var generatedTokenCounts = MLXArray.zeros([batch], dtype: .int32)
        let generatedTokenBuffer = MLXArray.zeros([batch, maxNewTokens], dtype: .int32) + eosVector.reshaped(batch, 1)
        var finishedMask = sequenceLimits .<= MLXArray(Int32(0))

        @inline(__always)
        func eosMask(for tokens: MLXArray) -> MLXArray {
            var mask = tokens .== MLXArray(Int32(eosTokenIds[0]))
            if eosTokenIds.count > 1 {
                for eos in eosTokenIds.dropFirst() {
                    mask = logicalOr(mask, tokens .== MLXArray(Int32(eos)))
                }
            }
            return mask
        }

        @inline(__always)
        func applyStep(_ sampledTokens: MLXArray, step: Int) -> MLXArray {
            let activeMask = logicalNot(finishedMask)
            let reachedEOS = eosMask(for: sampledTokens)
            let appendMask = logicalAnd(activeMask, logicalNot(reachedEOS))

            let currentStepValues = generatedTokenBuffer[0..., step]
            generatedTokenBuffer[0..., step] = which(appendMask, sampledTokens, currentStepValues)

            generatedTokenCounts = generatedTokenCounts + appendMask.asType(.int32)
            let reachedLimit = generatedTokenCounts .>= sequenceLimits
            finishedMask = logicalOr(finishedMask, logicalOr(reachedEOS, reachedLimit))

            if useRepetitionPenalty,
                let batchIndicesForRepetition,
                let tokenMask = repetitionTokenMask
            {
                let appendUpdates = appendMask.asType(tokenMask.dtype)
                repetitionTokenMask = tokenMask.at[batchIndicesForRepetition, sampledTokens].maximum(appendUpdates)
            }

            return which(finishedMask, eosVector, sampledTokens).asType(.int32)
        }

        let nextLogits = model.lmHead(lastHidden)[0..., 0, 0...]
        var currentTokens = sampleTokens(
            logits: nextLogits,
            repetitionTokenMask: repetitionTokenMask,
            parameters: parameters,
            randomState: randomState
        )
        currentTokens = applyStep(currentTokens, step: 0)

        var syncToken = currentTokens
        asyncEval(syncToken)

        if maxNewTokens > 1 {
            for step in 1..<maxNewTokens {
                let stepInput = currentTokens.reshaped(batch, 1)
                let cacheOffset = caches.first?.offset ?? 0
                let decodePositionIds = GLMOCRLanguage.decodePositionIds(
                    batch: batch,
                    length: 1,
                    cacheOffset: cacheOffset,
                    ropeDeltas: ropeDeltas
                )

                let logits = model.forward(
                    inputIds: stepInput,
                    inputEmbeddings: nil,
                    cache: caches,
                    positionIds: decodePositionIds,
                    attentionMask: attentionMask
                )[0..., 0, 0...]

                let next = sampleTokens(
                    logits: logits,
                    repetitionTokenMask: repetitionTokenMask,
                    parameters: parameters,
                    randomState: randomState
                )
                currentTokens = applyStep(next, step: step)

                syncToken = currentTokens
                asyncEval(syncToken)

                if decodeSyncInterval > 0 && (step % decodeSyncInterval) == 0 {
                    eval(syncToken)
                    if finishedMask.all().item(Bool.self) {
                        break
                    }
                }
            }
        }

        eval(syncToken)

        let generatedCounts = generatedTokenCounts.asArray(Int32.self).map(Int.init)
        let generatedFlat = generatedTokenBuffer.asArray(Int32.self).map(Int.init)

        for b in 0..<batch {
            let count = generatedCounts[b]
            guard count > 0 else { continue }
            let rowStart = b * maxNewTokens
            outputs[b].append(contentsOf: generatedFlat[rowStart ..< rowStart + count])
        }

        return outputs
        }
    }

    private static func sampleTokens(
        logits: MLXArray,
        repetitionTokenMask: MLXArray?,
        parameters: GLMOCRGenerationParameters,
        randomState: MLXRandom.RandomState?
    ) -> MLXArray {
        precondition(logits.ndim == 2, "sampleTokens expects logits [B, vocab]")
        let batch = logits.dim(0)
        let vocabSize = logits.dim(1)
        if let repetitionTokenMask {
            precondition(repetitionTokenMask.ndim == 2, "repetitionTokenMask expects shape [B, vocab]")
            precondition(repetitionTokenMask.shape == [batch, vocabSize], "repetitionTokenMask shape mismatch")
        }

        var logits2d = logits

        let needsFloat32SamplingPath = (
            logits2d.dtype == .bfloat16
                && (parameters.temperature > 0 || parameters.repetitionPenalty != 1.0)
        )
        if needsFloat32SamplingPath {
            logits2d = logits2d.asType(.float32)
        }

        if parameters.repetitionPenalty != 1.0, let repetitionTokenMask {
            let penalized = which(
                logits2d .< 0,
                logits2d * parameters.repetitionPenalty,
                logits2d / parameters.repetitionPenalty
            )
            logits2d = which(repetitionTokenMask .!= 0, penalized, logits2d)
        }

        if parameters.temperature <= 0 {
            return argMax(logits2d, axis: 1).asType(.int32)
        }

        let invTemperature: Float = 1.0 / parameters.temperature
        let scaledLogits = logits2d * invTemperature
        let hasTopP = parameters.topP > 0 && parameters.topP < 1

        var topK = parameters.topK > 0 ? min(parameters.topK, vocabSize) : vocabSize
        var fullLogNorm: MLXArray? = nil
        if hasTopP, parameters.topK <= 0, topK == vocabSize {
            let adaptive = adaptiveTopKForTopP(
                logits: scaledLogits,
                topP: parameters.topP,
                initialTopK: adaptiveTopPInitialTopK
            )
            topK = adaptive.topK
            fullLogNorm = adaptive.fullLogNorm
        }

        if topK < vocabSize {
            let neg = -scaledLogits
            let partitioned = argPartition(neg, kth: topK - 1, axis: 1)
            let topKIndices = partitioned[0..., 0 ..< topK]
            let topKLogits = takeAlong(scaledLogits, topKIndices, axis: 1)

            if hasTopP {
                let topPLogNorm: MLXArray? = parameters.topK > 0 ? nil : fullLogNorm
                return sampleTopPBatch(
                    logits: topKLogits,
                    tokenIndices: topKIndices,
                    topP: parameters.topP,
                    fullLogNorm: topPLogNorm,
                    randomState: randomState
                )
            }

            let idxInTopK: MLXArray = {
                if let randomState {
                    return withRandomState(randomState) {
                        categorical(topKLogits)
                    }
                }
                return categorical(topKLogits)
            }().asType(.int32)

            let idxInTopK2d = idxInTopK.reshaped(batch, 1)
            let picked = takeAlong(topKIndices, idxInTopK2d, axis: 1).reshaped(batch)
            return picked.asType(.int32)
        }

        if hasTopP {
            return sampleTopPBatch(
                logits: scaledLogits,
                tokenIndices: nil,
                topP: parameters.topP,
                fullLogNorm: nil,
                randomState: randomState
            )
        }

        let sampled: MLXArray = {
            if let randomState {
                return withRandomState(randomState) {
                    categorical(scaledLogits)
                }
            }
            return categorical(scaledLogits)
        }()
        return sampled.asType(.int32)
    }

    private static func adaptiveTopKForTopP(
        logits: MLXArray,
        topP: Float,
        initialTopK: Int
    ) -> (topK: Int, fullLogNorm: MLXArray) {
        precondition(logits.ndim == 2, "adaptiveTopKForTopP expects logits [B, vocab]")

        let vocabSize = logits.dim(1)
        let fullLogNorm = logSumExp(logits, axis: -1, keepDims: true)
        if vocabSize <= 1 {
            return (topK: vocabSize, fullLogNorm: fullLogNorm)
        }

        var topK = min(max(1, initialTopK), vocabSize)
        if topK == vocabSize {
            return (topK: topK, fullLogNorm: fullLogNorm)
        }

        let topPArray = MLXArray(topP)
        while true {
            let partitioned = argPartition(-logits, kth: topK - 1, axis: 1)
            let topKIndices = partitioned[0..., 0 ..< topK]
            let topKLogits = takeAlong(logits, topKIndices, axis: 1)
            let coveredMass = exp(logSumExp(topKLogits, axis: -1, keepDims: true) - fullLogNorm)
            if (coveredMass .>= topPArray).all().item(Bool.self) || topK == vocabSize {
                break
            }
            topK = min(vocabSize, topK * 2)
        }

        return (topK: topK, fullLogNorm: fullLogNorm)
    }

    private static func sampleTopPBatch(
        logits: MLXArray,
        tokenIndices: MLXArray?,
        topP: Float,
        fullLogNorm: MLXArray?,
        randomState: MLXRandom.RandomState?
    ) -> MLXArray {
        precondition(logits.ndim == 2, "sampleTopPBatch expects logits [B, n]")
        let batch = logits.dim(0)

        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = takeAlong(logits, sortedIndices, axis: 1)

        let cumulativeProbs: MLXArray
        let cumulativeThreshold: MLXArray
        if let fullLogNorm {
            let fullProbs = exp(sortedLogits - fullLogNorm)
            cumulativeProbs = cumsum(fullProbs, axis: -1)

            let candidateMass = fullProbs.sum(axis: -1, keepDims: true)
            cumulativeThreshold = clip(candidateMass - MLXArray(topP), min: Float(0))
        } else {
            let probs = softmax(sortedLogits, axis: -1)
            cumulativeProbs = cumsum(probs, axis: -1)
            cumulativeThreshold = MLXArray(Float(1.0) - topP)
        }

        let filteredLogits = which(
            cumulativeProbs .> cumulativeThreshold,
            sortedLogits,
            zeros(like: sortedLogits) + MLXArray(-Float.infinity).asType(sortedLogits.dtype)
        )

        let idxInSorted: MLXArray = {
            if let randomState {
                return withRandomState(randomState) {
                    categorical(filteredLogits)
                }
            }
            return categorical(filteredLogits)
        }().asType(.int32)

        let idxInSorted2d = idxInSorted.reshaped(batch, 1)
        let idxInOriginal = takeAlong(sortedIndices, idxInSorted2d, axis: 1).reshaped(batch)

        if let tokenIndices {
            let idx2d = idxInOriginal.reshaped(batch, 1)
            return takeAlong(tokenIndices, idx2d, axis: 1).reshaped(batch)
        }

        return idxInOriginal
    }
}
