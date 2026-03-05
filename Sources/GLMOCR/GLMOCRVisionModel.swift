import Foundation
import MLX
import MLXNN

private func glmOcrRotateHalf(_ x: MLXArray) -> MLXArray {
    let axis = x.ndim - 1
    let parts = x.split(parts: 2, axis: axis)
    let x1 = parts[0]
    let x2 = parts[1]
    return concatenated([-x2, x1], axis: axis)
}

private func glmOcrApplyRotaryPosEmbVision(
    queries: MLXArray,
    keys: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {

    let cos = cos.expandedDimensions(axis: 1)
    let sin = sin.expandedDimensions(axis: 1)

    let qEmbed = (queries * cos) + (glmOcrRotateHalf(queries) * sin)
    let kEmbed = (keys * cos) + (glmOcrRotateHalf(keys) * sin)

    return (qEmbed, kEmbed)
}

final class GLMOCRVisionRotaryEmbedding {
    private let inverseFreq: MLXArray

    init(dimensions: Int, theta: Float = 10_000.0) {
        let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        self.inverseFreq = MLXArray(Float(1.0)) / pow(theta, p)
    }

    func frequencies(positions: MLXArray) -> MLXArray {
        outer(positions, inverseFreq)
    }
}

final class GLMOCRVisionPatchEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Conv3d

    private let patchSize: Int
    private let temporalPatchSize: Int
    private let inChannels: Int
    private let embedDim: Int

    init(config: GLMOCRModelConfig.VisionConfig) {
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.inChannels = config.inChannels
        self.embedDim = config.hiddenSize

        let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
        self._proj.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: kernelSize,
            stride: kernelSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates = hiddenStates.reshaped(
            -1, inChannels, temporalPatchSize, patchSize, patchSize
        ).movedAxis(source: 1, destination: 4)

        hiddenStates = proj(hiddenStates)
        return hiddenStates.reshaped(-1, embedDim)
    }
}

final class GLMOCRVisionPatchMerger: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, contextDim: Int) {
        self._proj.wrappedValue = Linear(dim, dim, bias: false)
        self._postProjectionNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        self._gateProj.wrappedValue = Linear(dim, contextDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, contextDim, bias: false)
        self._downProj.wrappedValue = Linear(contextDim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = proj(x)
        x = gelu(postProjectionNorm(x))
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class GLMOCRVisionAttention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: GLMOCRModelConfig.VisionConfig) {
        self.numHeads = config.numHeads
        self.headDim = config.hiddenSize / config.numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: config.attentionBias ?? true)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.attentionBias ?? true)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        splitIndices: [Int],
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let attn = attentionPreProj(
            hiddenStates,
            splitIndices: splitIndices,
            positionEmbeddings: positionEmbeddings
        )
        return proj(attn)
    }

    func attentionPreProj(
        _ hiddenStates: MLXArray,
        splitIndices: [Int],
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let seqLen = hiddenStates.dim(0)

        let qkvStates = qkv(hiddenStates)
        let parts = split(qkvStates, parts: 3, axis: -1)
        var queries = parts[0].reshaped(seqLen, numHeads, headDim)
        var keys = parts[1].reshaped(seqLen, numHeads, headDim)
        let values = parts[2].reshaped(seqLen, numHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        (queries, keys) = glmOcrApplyRotaryPosEmbVision(
            queries: queries,
            keys: keys,
            cos: positionEmbeddings.cos,
            sin: positionEmbeddings.sin
        )

        let q = queries.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = keys.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = values.transposed(1, 0, 2).expandedDimensions(axis: 0)

        if splitIndices.isEmpty {
            let attn = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
            return attn.transposed(0, 2, 1, 3).reshaped(seqLen, -1)
        }

        let qSplits = q.split(indices: splitIndices, axis: 2)
        let kSplits = k.split(indices: splitIndices, axis: 2)
        let vSplits = v.split(indices: splitIndices, axis: 2)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(qSplits.count)
        for (qChunk, kChunk, vChunk) in zip(zip(qSplits, kSplits), vSplits).map({ ($0.0.0, $0.0.1, $0.1) }) {
            let out = MLXFast.scaledDotProductAttention(
                queries: qChunk,
                keys: kChunk,
                values: vChunk,
                scale: scale,
                mask: .none
            )
            outputs.append(out)
        }

        return concatenated(outputs, axis: 2)
            .transposed(0, 2, 1, 3)
            .reshaped(seqLen, -1)
    }

}

final class GLMOCRVisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: GLMOCRModelConfig.VisionConfig) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.attentionBias ?? true)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.attentionBias ?? true)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.attentionBias ?? true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        let fused = (gate * sigmoid(gate)) * up
        return downProj(fused)
    }
}

final class GLMOCRVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attn: GLMOCRVisionAttention
    @ModuleInfo(key: "mlp") var mlp: GLMOCRVisionMLP

    init(config: GLMOCRModelConfig.VisionConfig) {
        self._norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._attn.wrappedValue = GLMOCRVisionAttention(config: config)
        self._mlp.wrappedValue = GLMOCRVisionMLP(config: config)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        splitIndices: [Int],
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        var hiddenStates = hiddenStates

        hiddenStates =
            hiddenStates
            + attn(
                norm1(hiddenStates),
                splitIndices: splitIndices,
                positionEmbeddings: positionEmbeddings
            )
        hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
        return hiddenStates
    }
}

final class GLMOCRVisionModel: Module {
    private let spatialMergeSize: Int
    private let headDim: Int
    private let rotary: GLMOCRVisionRotaryEmbedding

    @ModuleInfo(key: "patch_embed") var patchEmbed: GLMOCRVisionPatchEmbed
    var blocks: [GLMOCRVisionBlock]
    @ModuleInfo(key: "merger") var merger: GLMOCRVisionPatchMerger
    @ModuleInfo(key: "downsample") var downsample: Conv2d
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: RMSNorm

    init(config: GLMOCRModelConfig.VisionConfig) {
        self.spatialMergeSize = config.spatialMergeSize
        self.headDim = config.hiddenSize / config.numHeads
        self.rotary = GLMOCRVisionRotaryEmbedding(dimensions: (headDim / 2))

        self._patchEmbed.wrappedValue = GLMOCRVisionPatchEmbed(config: config)
        self.blocks = (0..<config.depth).map { _ in GLMOCRVisionBlock(config: config) }
        self._merger.wrappedValue = GLMOCRVisionPatchMerger(
            dim: config.outHiddenSize,
            contextDim: config.outHiddenSize * config.inChannels
        )
        self._downsample.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.outHiddenSize,
            kernelSize: .init(config.spatialMergeSize),
            stride: .init(config.spatialMergeSize),
            padding: 0,
            bias: true
        )
        self._postLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray, gridTHW: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        var hiddenStates = patchEmbed(pixelValues)

        let splitIndices = glmOcrVisionSplitIndices(gridTHW: gridTHW)
        let positionEmbeddings = glmOcrVisionPositionEmbeddings(
            gridTHW: gridTHW,
            spatialMergeSize: spatialMergeSize,
            rotary: rotary,
            dtype: hiddenStates.dtype
        )

        for block in blocks {
            hiddenStates = block(hiddenStates, splitIndices: splitIndices, positionEmbeddings: positionEmbeddings)
        }

        hiddenStates = postLayerNorm(hiddenStates)

        hiddenStates = hiddenStates.asType(downsample.weight.dtype)
        hiddenStates = hiddenStates.reshaped(-1, spatialMergeSize, spatialMergeSize, hiddenStates.dim(-1))
        hiddenStates = downsample(hiddenStates).reshaped(-1, downsample.weight.dim(0))
        return merger(hiddenStates)
    }

}

private func glmOcrVisionSplitIndices(gridTHW: [(t: Int, h: Int, w: Int)]) -> [Int] {
    var cuSeqlens: [Int] = [0]
    cuSeqlens.reserveCapacity(gridTHW.reduce(0) { $0 + $1.t } + 1)

    var running = 0
    for grid in gridTHW {
        let seqLen = grid.h * grid.w
        for _ in 0..<grid.t {
            running += seqLen
            cuSeqlens.append(running)
        }
    }

    guard cuSeqlens.count >= 2 else { return [] }
    if cuSeqlens.count == 2 { return [] }
    return Array(cuSeqlens[1..<(cuSeqlens.count - 1)])
}

private func glmOcrVisionPositionEmbeddings(
    gridTHW: [(t: Int, h: Int, w: Int)],
    spatialMergeSize: Int,
    rotary: GLMOCRVisionRotaryEmbedding,
    dtype: DType
) -> (cos: MLXArray, sin: MLXArray) {
    var hPositions: [Int] = []
    var wPositions: [Int] = []

    let merge = spatialMergeSize
    for grid in gridTHW {
        let t = grid.t
        let h = grid.h
        let w = grid.w

        let hBlocks = h / merge
        let wBlocks = w / merge

        var localH: [Int] = []
        var localW: [Int] = []
        localH.reserveCapacity(h * w)
        localW.reserveCapacity(h * w)

        for hb in 0..<hBlocks {
            for wb in 0..<wBlocks {
                for mr in 0..<merge {
                    for mc in 0..<merge {
                        localH.append(hb * merge + mr)
                        localW.append(wb * merge + mc)
                    }
                }
            }
        }

        for _ in 0..<t {
            hPositions.append(contentsOf: localH)
            wPositions.append(contentsOf: localW)
        }
    }

    let hIds = MLXArray(hPositions).asType(.float32)
    let wIds = MLXArray(wPositions).asType(.float32)

    let freqsH = rotary.frequencies(positions: hIds)
    let freqsW = rotary.frequencies(positions: wIds)
    let rotaryEmb = concatenated([freqsH, freqsW], axis: -1)
    let emb = concatenated([rotaryEmb, rotaryEmb], axis: -1)

    return (cos(emb).asType(dtype), sin(emb).asType(dtype))
}
