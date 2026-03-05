import Foundation
import MLX
import MLXNN

private func glmOcrRepeatInterleave(_ x: MLXArray, repeats: Int, axis: Int) -> MLXArray {
    let axis = axis >= 0 ? axis : x.ndim + axis
    precondition(axis >= 0 && axis < x.ndim, "axis out of bounds")

    var shape = x.shape
    let expandedAxis = axis + 1

    var expanded = x.expandedDimensions(axis: expandedAxis)

    var tileShape = Array(repeating: 1, count: expanded.ndim)
    tileShape[expandedAxis] = repeats
    expanded = tiled(expanded, repetitions: tileShape)

    shape[axis] = shape[axis] * repeats
    return expanded.reshaped(shape)
}

private func glmOcrRotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
    let lastDim = x.dim(-1)
    precondition(lastDim % 2 == 0, "RoPE rotateHalf requires even last dimension (got \(lastDim))")

    let prefixShape = Array(x.shape.dropLast())
    let reshaped = x.reshaped(prefixShape + [lastDim / 2, 2])
    let x1 = reshaped[.ellipsis, 0]
    let x2 = reshaped[.ellipsis, 1]
    let stackedPairs = stacked([-x2, x1], axis: -1)
    return stackedPairs.flattened(start: -2, end: -1)
}

private func glmOcrApplyRotaryPosEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cos = cos[0..., .newAxis, 0..., 0...]
    let sin = sin[0..., .newAxis, 0..., 0...]

    let rotaryDim = cos.dim(-1)
    let qRot = q[0..., 0..., 0..., ..<rotaryDim]
    let qPass = q[0..., 0..., 0..., rotaryDim...]
    let kRot = k[0..., 0..., 0..., ..<rotaryDim]
    let kPass = k[0..., 0..., 0..., rotaryDim...]

    let qEmbed = (qRot * cos) + (glmOcrRotateHalfInterleaved(qRot) * sin)
    let kEmbed = (kRot * cos) + (glmOcrRotateHalfInterleaved(kRot) * sin)

    return (concatenated([qEmbed, qPass], axis: -1), concatenated([kEmbed, kPass], axis: -1))
}

final class GLMOCRTextRotaryEmbedding {
    private let invFreq: MLXArray
    private let mropeSection: [Int]
    private let splitIndices: [Int]

    init(config: GLMOCRModelConfig.TextConfig) {
        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
        let partialRotaryFactor = Float(config.ropeParameters?.partialRotaryFactor ?? 1.0)
        let ropeDim = Int((Float(headDim) * partialRotaryFactor).rounded(.down))
        let base = Float(config.ropeParameters?.ropeTheta ?? 10_000.0)
        let mropeSection = config.ropeParameters?.mropeSection ?? [16, 24, 24]

        var freq = MLXArray(stride(from: 0, to: ropeDim, by: 2)).asType(.float32)
        freq = freq / Float(ropeDim)
        self.invFreq = MLXArray(Float(1.0)) / pow(MLXArray(base), freq)
        self.mropeSection = mropeSection

        var splitIndices: [Int] = []
        splitIndices.reserveCapacity(max(mropeSection.count - 1, 0))
        var running = 0
        for v in mropeSection.dropLast() {
            running += v
            splitIndices.append(running)
        }
        self.splitIndices = splitIndices
    }

    func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var positionIds = positionIds
        if positionIds.ndim == 2 {
            positionIds = tiled(positionIds[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }

        let pos = positionIds.asType(.float32)
        let invFreq = self.invFreq[.newAxis, .newAxis, .newAxis, 0...]
        var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq

        let chunks = split(freqs, indices: splitIndices, axis: -1)
        let selected = chunks.enumerated().map { i, chunk in
            chunk[i % 3, 0..., 0..., 0...]
        }
        freqs = concatenated(selected, axis: -1)
        let emb = glmOcrRepeatInterleave(freqs, repeats: 2, axis: -1)
        let cosValues = cos(emb).asType(x.dtype)
        let sinValues = sin(emb).asType(x.dtype)
        return (cosValues, sinValues)
    }
}

final class GLMOCRTextAttention: Module {
    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: GLMOCRModelConfig.TextConfig) {
        let hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim ?? (hiddenSize / numHeads)
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias ?? false)
        _kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias ?? false)
        _vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias ?? false)
        _oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: GLMOCRKVCache?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        (q, k) = glmOcrApplyRotaryPosEmb(q: q, k: k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        let out = attn.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return oProj(out)
    }
}

final class GLMOCRTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(config: GLMOCRModelConfig.TextConfig) {
        _gateUp.wrappedValue = Linear(config.hiddenSize, config.intermediateSize * 2, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = gateUp(x)
        let parts = split(x, parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

final class GLMOCRTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GLMOCRTextAttention
    @ModuleInfo(key: "mlp") var mlp: GLMOCRTextMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

    init(config: GLMOCRModelConfig.TextConfig) {
        _selfAttn.wrappedValue = GLMOCRTextAttention(config: config)
        _mlp.wrappedValue = GLMOCRTextMLP(config: config)

        let eps = Float(config.rmsNormEps)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _postSelfAttnLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        _postMlpLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: GLMOCRKVCache?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        var x =
            x
            + postSelfAttnLayerNorm(
                selfAttn(inputLayerNorm(x), mask: mask, cache: cache, positionEmbeddings: positionEmbeddings)
            )
        let residual = x
        x = postMlpLayerNorm(mlp(postAttentionLayerNorm(x))) + residual
        return x
    }
}

final class GLMOCRTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [GLMOCRTextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let rotaryEmbedding: GLMOCRTextRotaryEmbedding

    let config: GLMOCRModelConfig.TextConfig

    init(config: GLMOCRModelConfig.TextConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map { _ in GLMOCRTextDecoderLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self.rotaryEmbedding = GLMOCRTextRotaryEmbedding(config: config)
        super.init()
    }

    func callAsFunction(
        inputIds: MLXArray,
        inputEmbeddings: MLXArray?,
        cache: [GLMOCRKVCache]?,
        positionIds: MLXArray?,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = inputIds.dim(0)
        let seqLen = inputIds.dim(1)

        var hidden: MLXArray
        if let inputEmbeddings {
            hidden = inputEmbeddings.asType(norm.weight.dtype)
        } else {
            hidden = embedTokens(inputIds).asType(norm.weight.dtype)
        }

        let cacheOffset = cache?.first?.offset ?? 0
        let queryMask3d: MLXArray? = {
            guard let attentionMask else { return nil }
            precondition(attentionMask.ndim == 2, "attentionMask must be rank-2 [B, L]")
            precondition(attentionMask.dim(0) == batch, "attentionMask batch mismatch")
            let keyLen = cacheOffset + seqLen
            precondition(attentionMask.dim(1) >= keyLen, "attentionMask length must cover cached keys")
            let queryMask2d = (attentionMask[0..., cacheOffset..<keyLen] .== 1)
            return queryMask2d.expandedDimensions(axis: 2)
        }()

        @inline(__always)
        func applyQueryMask(_ x: MLXArray) -> MLXArray {
            guard let queryMask3d else { return x }
            return which(queryMask3d, x, zeros(like: x))
        }

        hidden = applyQueryMask(hidden)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let attentionMask else {
                if seqLen == 1 { return .none }
                return .causal
            }

            precondition(attentionMask.ndim == 2, "attentionMask must be rank-2 [B, L]")
            precondition(attentionMask.dim(0) == batch, "attentionMask batch mismatch")
            let keyLen = cacheOffset + seqLen
            precondition(attentionMask.dim(1) >= keyLen, "attentionMask length must cover cached keys")

            let keyMask = (attentionMask[0..., 0..<keyLen] .== 1)
            let keyMask4d = keyMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            let maskDType = hidden.dtype
            let keyAdditiveMask4d = which(
                keyMask4d,
                zeros(like: keyMask4d).asType(maskDType),
                MLXArray(Float(-1e9), dtype: maskDType)
            )

            if seqLen == 1 {
                return .array(keyAdditiveMask4d)
            }

            let causal = Self.createCausalMask(n: seqLen, offset: cacheOffset)
            let causal4d = causal.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            let causalAdditive = which(
                causal4d,
                zeros(like: causal4d).asType(maskDType),
                MLXArray(Float(-1e9), dtype: maskDType)
            )
            return .array(causalAdditive + keyAdditiveMask4d)
        }()

        let positionIds: MLXArray = {
            if let positionIds { return positionIds }

            let offset = cache?.first?.offset ?? 0
            var base = MLXArray(stride(from: offset, to: offset + seqLen, by: 1)).asType(.int32)
            base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
            return tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }()

        let positionEmbeddings = rotaryEmbedding(x: hidden, positionIds: positionIds)

        for (i, layer) in layers.enumerated() {
            let layerCache: GLMOCRKVCache? = cache == nil ? nil : cache?[i]
            hidden = layer(hidden, mask: mask, cache: layerCache, positionEmbeddings: positionEmbeddings)
            hidden = applyQueryMask(hidden)
        }

        return norm(hidden)
    }
}

extension GLMOCRTextModel {
    fileprivate static func createCausalMask(n: Int, offset: Int) -> MLXArray {
        precondition(n >= 1, "createCausalMask expects n >= 1")
        precondition(offset >= 0, "createCausalMask expects offset >= 0")

        let keyLen = offset + n
        var rinds = MLXArray(Int32(0)..<Int32(keyLen))
        let lrange = MLXArray(Int32(offset)..<Int32(offset + n))
        let linds = lrange[0..., .newAxis]
        rinds = rinds[.newAxis]
        return linds .>= rinds
    }
}
