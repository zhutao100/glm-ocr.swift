import Foundation
import MLX
import MLXNN

public struct PPDocLayoutV3ForObjectDetectionOutput {
    public let logits: MLXArray 
    public let predBoxes: MLXArray 
    public let orderLogits: MLXArray 
    public let outMasks: MLXArray 
}

private func ppdoclayoutActivation(_ name: String, _ x: MLXArray) -> MLXArray {
    switch name {
    case "relu":
        return relu(x)
    case "gelu":
        return gelu(x)
    case "silu":
        return silu(x)
    default:
        return x
    }
}

private func ppdoclayoutInverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let x = clip(x, min: 0.0, max: 1.0)
    let x1 = clip(x, min: eps, max: 1.0)
    let x2 = clip(1.0 - x, min: eps, max: 1.0)
    return log(x1 / x2)
}

private func ppdoclayoutPadBottomRight(_ x: MLXArray, by pad: Int) -> MLXArray {
    precondition(x.ndim == 4, "expected NHWC input")
    let widths: [IntOrPair] = [
        0,
        .init((0, pad)),
        .init((0, pad)),
        0,
    ]
    return padded(x, widths: widths, mode: .constant)
}



final class PPDocLayoutV3ConvLayer: Module, UnaryLayer {
    @ModuleInfo(key: "convolution") var convolution: Conv2d
    @ModuleInfo(key: "normalization") var normalization: BatchNorm
    private let activation: String

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 3, stride: Int = 1, activation: String = "relu") {
        _convolution.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: .init(kernelSize / 2),
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(featureCount: outChannels, eps: 1e-5, momentum: 0.1, affine: true, trackRunningStats: true)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ppdoclayoutActivation(activation, normalization(convolution(x)))
    }
}

final class PPDocLayoutV3ConvNormLayer: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    private let activation: String?

    init(config: PPDocLayoutV3Config, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int? = nil, activation: String? = nil) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: .init(padding ?? ((kernelSize - 1) / 2)),
            bias: false
        )
        _norm.wrappedValue = BatchNorm(
            featureCount: outChannels,
            eps: config.batchNormEps,
            momentum: 0.1,
            affine: true,
            trackRunningStats: true
        )
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        guard let activation else { return y }
        return ppdoclayoutActivation(activation, y)
    }
}

final class PPDocLayoutV3ConvBN: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(config: PPDocLayoutV3Config, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int, bias: Bool = false) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: .init(padding),
            bias: bias
        )
        _bn.wrappedValue = BatchNorm(
            featureCount: outChannels,
            eps: config.batchNormEps,
            momentum: 0.1,
            affine: true,
            trackRunningStats: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        bn(conv(x))
    }
}



final class PPDocLayoutV3MLPPredictionHead: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(_ inputDim: Int, _ hiddenDim: Int, _ outputDim: Int, numLayers: Int) {
        let hidden = Array(repeating: hiddenDim, count: max(numLayers - 1, 0))
        let dimsIn = [inputDim] + hidden
        let dimsOut = hidden + [outputDim]
        _layers.wrappedValue = zip(dimsIn, dimsOut).map { Linear($0, $1, bias: true) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for (i, layer) in layers.enumerated() {
            x = layer(x)
            if i < layers.count - 1 {
                x = relu(x)
            }
        }
        return x
    }
}

final class PPDocLayoutV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    private let activationFunction: String

    init(config: PPDocLayoutV3Config, hiddenSize: Int, intermediateSize: Int, activationFunction: String) {
        _fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
        self.activationFunction = activationFunction
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = ppdoclayoutActivation(activationFunction, x)
        x = fc2(x)
        return x
    }
}



final class PPDocLayoutV3SelfAttention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        precondition(hiddenSize % numHeads == 0, "hiddenSize must be divisible by numHeads")
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(headDim), -0.5)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil, positionEmbeddings: MLXArray? = nil) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)
        let hiddenSize = hiddenStates.dim(2)

        let queryKeyInput = positionEmbeddings == nil ? hiddenStates : (hiddenStates + positionEmbeddings!)

        var q = qProj(queryKeyInput).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(queryKeyInput).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(hiddenStates).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let attentionMask {
            maskMode = .array(attentionMask)
        } else {
            maskMode = .none
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: maskMode
        )

        let out = attn.transposed(0, 2, 1, 3).reshaped(batch, seqLen, hiddenSize)
        return oProj(out)
    }
}

final class PPDocLayoutV3EncoderLayer: Module, UnaryLayer {
    private let normalizeBefore: Bool
    private let dropout: Float

    @ModuleInfo(key: "self_attn") var selfAttn: PPDocLayoutV3SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: PPDocLayoutV3MLP
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: PPDocLayoutV3Config) {
        self.normalizeBefore = config.normalizeBefore
        self.dropout = config.dropout
        let hiddenSize = config.encoderHiddenDim
        _selfAttn.wrappedValue = PPDocLayoutV3SelfAttention(hiddenSize: hiddenSize, numHeads: config.encoderAttentionHeads)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = PPDocLayoutV3MLP(
            config: config,
            hiddenSize: hiddenSize,
            intermediateSize: config.encoderFfnDim,
            activationFunction: config.encoderActivationFunction
        )
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        callAsFunction(hiddenStates, attentionMask: nil, spatialPositionEmbeddings: nil)
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray?, spatialPositionEmbeddings: MLXArray?) -> MLXArray {
        var hiddenStates = hiddenStates
        var residual = hiddenStates
        if normalizeBefore {
            hiddenStates = selfAttnLayerNorm(hiddenStates)
        }

        hiddenStates = selfAttn(hiddenStates: hiddenStates, attentionMask: attentionMask, positionEmbeddings: spatialPositionEmbeddings)
        if dropout > 0 && training {
            hiddenStates = Dropout(p: dropout)(hiddenStates)
        }
        hiddenStates = residual + hiddenStates
        if !normalizeBefore {
            hiddenStates = selfAttnLayerNorm(hiddenStates)
        }

        if normalizeBefore {
            hiddenStates = finalLayerNorm(hiddenStates)
        }
        residual = hiddenStates

        hiddenStates = mlp(hiddenStates)
        if dropout > 0 && training {
            hiddenStates = Dropout(p: dropout)(hiddenStates)
        }
        hiddenStates = residual + hiddenStates
        if !normalizeBefore {
            hiddenStates = finalLayerNorm(hiddenStates)
        }

        return hiddenStates
    }
}

final class PPDocLayoutV3SinePositionEmbedding {
    private let embedDim: Int
    private let temperature: Int

    private var cache: [String: MLXArray] = [:]

    init(embedDim: Int, temperature: Int) {
        self.embedDim = embedDim
        self.temperature = temperature
    }

    func callAsFunction(width: Int, height: Int, dtype: DType) -> MLXArray {
        let key = "\(width)x\(height):\(dtype)"
        if let cached = cache[key] { return cached }

        precondition(embedDim % 4 == 0, "embedDim must be divisible by 4")
        let posDim = embedDim / 4

        let gridW = broadcast(arange(width, dtype: .float32).reshaped(1, width), to: [height, width])
        let gridH = broadcast(arange(height, dtype: .float32).reshaped(height, 1), to: [height, width])

        let omega = arange(posDim, dtype: .float32) / Float(posDim)
        let omegaScaled = 1.0 / pow(MLXArray(Float(temperature)), omega)

        let outW = gridW.flattened()[.newAxis, 0...] * omegaScaled[0..., .newAxis] 
        let outH = gridH.flattened()[.newAxis, 0...] * omegaScaled[0..., .newAxis]

        let outW2 = outW.transposed(1, 0)
        let outH2 = outH.transposed(1, 0)

        let emb = concatenated(
            [
                sin(outH2),
                cos(outH2),
                sin(outW2),
                cos(outW2),
            ],
            axis: 1
        )[.newAxis, 0..., 0...].asType(dtype)

        cache[key] = emb
        return emb
    }
}

final class PPDocLayoutV3AIFILayer: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [PPDocLayoutV3EncoderLayer]

    private let positionEmbedding: PPDocLayoutV3SinePositionEmbedding
    private let encoderHiddenDim: Int

    init(config: PPDocLayoutV3Config) {
        self.encoderHiddenDim = config.encoderHiddenDim
        self.positionEmbedding = PPDocLayoutV3SinePositionEmbedding(embedDim: config.encoderHiddenDim, temperature: config.positionalEncodingTemperature)
        _layers.wrappedValue = (0..<config.encoderLayers).map { _ in PPDocLayoutV3EncoderLayer(config: config) }
        super.init()
    }

    func callAsFunction(_ featureMap: MLXArray) -> MLXArray {
        
        let batch = featureMap.dim(0)
        let height = featureMap.dim(1)
        let width = featureMap.dim(2)

        var hidden = featureMap.reshaped(batch, height * width, encoderHiddenDim)
        let pos = positionEmbedding(width: width, height: height, dtype: hidden.dtype)

        for layer in layers {
            hidden = layer(hidden, attentionMask: nil, spatialPositionEmbeddings: pos)
        }

        return hidden.reshaped(batch, height, width, encoderHiddenDim)
    }
}



final class PPDocLayoutV3RepVggBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: PPDocLayoutV3ConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: PPDocLayoutV3ConvNormLayer
    private let activation: String

    init(config: PPDocLayoutV3Config) {
        self.activation = config.activationFunction
        let hiddenChannels = Int(Double(config.encoderHiddenDim) * config.hiddenExpansion)
        _conv1.wrappedValue = PPDocLayoutV3ConvNormLayer(config: config, inChannels: hiddenChannels, outChannels: hiddenChannels, kernelSize: 3, stride: 1, padding: 1)
        _conv2.wrappedValue = PPDocLayoutV3ConvNormLayer(config: config, inChannels: hiddenChannels, outChannels: hiddenChannels, kernelSize: 1, stride: 1, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ppdoclayoutActivation(activation, conv1(x) + conv2(x))
    }
}

final class PPDocLayoutV3CSPRepLayer: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: PPDocLayoutV3ConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: PPDocLayoutV3ConvNormLayer
    @ModuleInfo(key: "bottlenecks") var bottlenecks: [PPDocLayoutV3RepVggBlock]

    init(config: PPDocLayoutV3Config) {
        let inChannels = config.encoderHiddenDim * 2
        let outChannels = config.encoderHiddenDim
        let hiddenChannels = Int(Double(outChannels) * config.hiddenExpansion)
        _conv1.wrappedValue = PPDocLayoutV3ConvNormLayer(config: config, inChannels: inChannels, outChannels: hiddenChannels, kernelSize: 1, stride: 1, activation: config.activationFunction)
        _conv2.wrappedValue = PPDocLayoutV3ConvNormLayer(config: config, inChannels: inChannels, outChannels: hiddenChannels, kernelSize: 1, stride: 1, activation: config.activationFunction)
        _bottlenecks.wrappedValue = (0..<3).map { _ in PPDocLayoutV3RepVggBlock(config: config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h1 = conv1(x)
        for b in bottlenecks {
            h1 = b(h1)
        }
        let h2 = conv2(x)
        
        return h1 + h2
    }
}

final class PPDocLayoutV3ScaleHead: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(inChannels: Int, featureChannels: Int, fpnStride: Int, baseStride: Int, alignCorners: Bool = false) {
        let headLength = max(1, Int(log2(Double(fpnStride)) - log2(Double(baseStride))))
        var out: [Module] = []
        out.reserveCapacity(headLength * 2)
        for k in 0..<headLength {
            let inC = (k == 0) ? inChannels : featureChannels
            out.append(PPDocLayoutV3ConvLayer(inChannels: inC, outChannels: featureChannels, kernelSize: 3, stride: 1, activation: "silu"))
            if fpnStride != baseStride {
                out.append(Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: alignCorners)))
            }
        }
        _layers.wrappedValue = out
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            if let layer = layer as? (any UnaryLayer) {
                x = layer(x)
            } else {
                fatalError("Unexpected non-unary layer in scale head: \(type(of: layer))")
            }
        }
        return x
    }
}

final class PPDocLayoutV3MaskFeatFPN: Module {
    @ModuleInfo(key: "scale_heads") var scaleHeads: [PPDocLayoutV3ScaleHead]
    @ModuleInfo(key: "output_conv") var outputConv: PPDocLayoutV3ConvLayer

    private let reorderIndex: [Int]
    private let fpnStrides: [Int]
    private let alignCorners: Bool

    init(inChannels: [Int], fpnStrides: [Int], featureChannels: Int, outChannels: Int, alignCorners: Bool = false) {
        let zipped = zip(fpnStrides, Array(0..<fpnStrides.count)).sorted { $0.0 < $1.0 }
        self.reorderIndex = zipped.map { $0.1 }
        self.fpnStrides = zipped.map { $0.0 }
        self.alignCorners = alignCorners

        let inChannelsReordered = reorderIndex.map { inChannels[$0] }
        let baseStride = self.fpnStrides[0]

        _scaleHeads.wrappedValue = zip(inChannelsReordered, self.fpnStrides).map { inC, stride in
            PPDocLayoutV3ScaleHead(inChannels: inC, featureChannels: featureChannels, fpnStride: stride, baseStride: baseStride, alignCorners: alignCorners)
        }
        _outputConv.wrappedValue = PPDocLayoutV3ConvLayer(inChannels: featureChannels, outChannels: outChannels, kernelSize: 3, stride: 1, activation: "silu")
        super.init()
    }

    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        let x = reorderIndex.map { inputs[$0] }
        var output = scaleHeads[0](x[0])

        for i in 1..<fpnStrides.count {
            let scaled = scaleHeads[i](x[i])
            let up = Upsample(scaleFactor: [Float(output.dim(1)) / Float(scaled.dim(1)), Float(output.dim(2)) / Float(scaled.dim(2))], mode: .linear(alignCorners: alignCorners))
            output = output + up(scaled)
        }

        return outputConv(output)
    }
}

final class PPDocLayoutV3EncoderMaskOutput: Module, UnaryLayer {
    @ModuleInfo(key: "base_conv") var baseConv: PPDocLayoutV3ConvLayer
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(inChannels: Int, numPrototypes: Int) {
        _baseConv.wrappedValue = PPDocLayoutV3ConvLayer(inChannels: inChannels, outChannels: inChannels, kernelSize: 3, stride: 1, activation: "silu")
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: numPrototypes, kernelSize: 1, stride: 1, padding: 0, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(baseConv(x))
    }
}

final class PPDocLayoutV3HybridEncoder: Module {
    @ModuleInfo(key: "aifi") var aifi: [PPDocLayoutV3AIFILayer]
    @ModuleInfo(key: "lateral_convs") var lateralConvs: [PPDocLayoutV3ConvNormLayer]
    @ModuleInfo(key: "fpn_blocks") var fpnBlocks: [PPDocLayoutV3CSPRepLayer]
    @ModuleInfo(key: "downsample_convs") var downsampleConvs: [PPDocLayoutV3ConvNormLayer]
    @ModuleInfo(key: "pan_blocks") var panBlocks: [PPDocLayoutV3CSPRepLayer]

    @ModuleInfo(key: "mask_feature_head") var maskFeatureHead: PPDocLayoutV3MaskFeatFPN
    @ModuleInfo(key: "encoder_mask_lateral") var encoderMaskLateral: PPDocLayoutV3ConvLayer
    @ModuleInfo(key: "encoder_mask_output") var encoderMaskOutput: PPDocLayoutV3EncoderMaskOutput

    private let encodeProjLayers: [Int]
    private let featStrides: [Int]

    init(config: PPDocLayoutV3Config) {
        self.encodeProjLayers = config.encodeProjLayers
        self.featStrides = config.featureStrides

        _aifi.wrappedValue = (0..<encodeProjLayers.count).map { _ in PPDocLayoutV3AIFILayer(config: config) }

        let numStages = config.encoderInChannels.count
        let numFpnStages = max(numStages - 1, 0)
        let numPanStages = numFpnStages

        _lateralConvs.wrappedValue = (0..<numFpnStages).map { _ in
            PPDocLayoutV3ConvNormLayer(config: config, inChannels: config.encoderHiddenDim, outChannels: config.encoderHiddenDim, kernelSize: 1, stride: 1, activation: config.activationFunction)
        }
        _fpnBlocks.wrappedValue = (0..<numFpnStages).map { _ in PPDocLayoutV3CSPRepLayer(config: config) }

        _downsampleConvs.wrappedValue = (0..<numPanStages).map { _ in
            PPDocLayoutV3ConvNormLayer(config: config, inChannels: config.encoderHiddenDim, outChannels: config.encoderHiddenDim, kernelSize: 3, stride: 2, activation: config.activationFunction)
        }
        _panBlocks.wrappedValue = (0..<numPanStages).map { _ in PPDocLayoutV3CSPRepLayer(config: config) }

        let maskFeatureChannels = config.maskFeatureChannels
        _maskFeatureHead.wrappedValue = PPDocLayoutV3MaskFeatFPN(
            inChannels: Array(repeating: config.encoderHiddenDim, count: featStrides.count),
            fpnStrides: featStrides,
            featureChannels: maskFeatureChannels[0],
            outChannels: maskFeatureChannels[1]
        )
        _encoderMaskLateral.wrappedValue = PPDocLayoutV3ConvLayer(inChannels: config.x4FeatDim, outChannels: maskFeatureChannels[1], kernelSize: 3, stride: 1, activation: "silu")
        _encoderMaskOutput.wrappedValue = PPDocLayoutV3EncoderMaskOutput(inChannels: maskFeatureChannels[1], numPrototypes: config.numPrototypes)

        super.init()
    }

    func callAsFunction(_ featureMaps: [MLXArray], x4Feat: MLXArray) -> (panFeatureMaps: [MLXArray], maskFeat: MLXArray) {
        var featureMaps = featureMaps
        if !aifi.isEmpty {
            for (i, encInd) in encodeProjLayers.enumerated() {
                featureMaps[encInd] = aifi[i](featureMaps[encInd])
            }
        }

        
        var fpnFeatureMaps: [MLXArray] = [featureMaps.last!]
        for (idx, (lateralConv, fpnBlock)) in zip(lateralConvs, fpnBlocks).enumerated() {
            let backboneFeatureMap = featureMaps[(featureMaps.count - 2) - idx]
            var top = fpnFeatureMaps[fpnFeatureMaps.count - 1]
            top = lateralConv(top)
            fpnFeatureMaps[fpnFeatureMaps.count - 1] = top
            top = Upsample(scaleFactor: 2.0, mode: .nearest)(top)
            let fused = concatenated([top, backboneFeatureMap], axis: -1)
            let newMap = fpnBlock(fused)
            fpnFeatureMaps.append(newMap)
        }
        fpnFeatureMaps.reverse()

        
        var panFeatureMaps: [MLXArray] = [fpnFeatureMaps[0]]
        for idx in 0..<downsampleConvs.count {
            let top = panFeatureMaps.last!
            let fpn = fpnFeatureMaps[idx + 1]
            let downsampled = downsampleConvs[idx](top)
            let fused = concatenated([downsampled, fpn], axis: -1)
            let newMap = panBlocks[idx](fused)
            panFeatureMaps.append(newMap)
        }

        var maskFeat = maskFeatureHead(panFeatureMaps)
        maskFeat = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))(maskFeat)
        maskFeat = maskFeat + encoderMaskLateral(x4Feat)
        maskFeat = encoderMaskOutput(maskFeat)

        return (panFeatureMaps: panFeatureMaps, maskFeat: maskFeat)
    }
}



private enum PPDocLayoutV3Metal {
    static let gridSampleForward = MLXFast.metalKernel(
        name: "ppdoclayout_grid_sample_forward",
        inputNames: ["x", "grid"],
        outputNames: ["out"],
        source: """
            uint elem = thread_position_in_grid.x;

            int B = x_shape[0];
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            int c = elem % C;
            int w = (elem / C) % gW;
            int h = (elem / (C * gW)) % gH;
            int b = elem / (C * gW * gH);

            if (b >= B) return;

            uint grid_idx = ((b * gH + h) * gW + w) * 2;
            float gx = grid[grid_idx];
            float gy = grid[grid_idx + 1];

            float ix = ((gx + 1.0) * float(W) - 1.0) / 2.0;
            float iy = ((gy + 1.0) * float(H) - 1.0) / 2.0;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;
            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;
            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            float nw = (float(ix_se) - ix) * (float(iy_se) - iy);
            float ne = (ix - float(ix_sw)) * (float(iy_sw) - iy);
            float sw = (float(ix_ne) - ix) * (iy - float(iy_ne));
            float se = (ix - float(ix_nw)) * (iy - float(iy_nw));

            int base_idx = b * b_stride + c;
            float I_nw = 0.0;
            float I_ne = 0.0;
            float I_sw = 0.0;
            float I_se = 0.0;

            if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
                I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
            if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W)
                I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
            if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W)
                I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
            if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W)
                I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

            int out_idx = ((b * gH + h) * gW + w) * C + c;
            out[out_idx] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
            """
    )

    static func gridSampleBilinear(_ x: MLXArray, _ grid: MLXArray) -> MLXArray {
        precondition(x.ndim == 4, "x must be NHWC")
        precondition(grid.ndim == 4 && grid.dim(3) == 2, "grid must be [B, gH, gW, 2]")
        let b = x.dim(0)
        let gH = grid.dim(1)
        let gW = grid.dim(2)
        let c = x.dim(3)
        let total = b * gH * gW * c
        let outShape = [b, gH, gW, c]
        return gridSampleForward(
            [x, grid],
            grid: (total, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [outShape],
            outputDTypes: [x.dtype]
        )[0]
    }
}

final class PPDocLayoutV3MultiscaleDeformableAttention: Module {
    private let dModel: Int
    private let nLevels: Int
    private let nHeads: Int
    private let nPoints: Int
    private let headDim: Int

    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    init(config: PPDocLayoutV3Config, numHeads: Int, nPoints: Int) {
        self.dModel = config.dModel
        self.nLevels = config.numFeatureLevels
        self.nHeads = numHeads
        self.nPoints = nPoints
        precondition(dModel % numHeads == 0, "d_model must be divisible by numHeads")
        self.headDim = dModel / numHeads

        _samplingOffsets.wrappedValue = Linear(dModel, numHeads * nLevels * nPoints * 2, bias: true)
        _attentionWeights.wrappedValue = Linear(dModel, numHeads * nLevels * nPoints, bias: true)
        _valueProj.wrappedValue = Linear(dModel, dModel, bias: true)
        _outputProj.wrappedValue = Linear(dModel, dModel, bias: true)
        super.init()
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        positionEmbeddings: MLXArray?,
        referencePoints: MLXArray,
        spatialShapes: [(height: Int, width: Int)]
    ) -> MLXArray {
        var hiddenStates = hiddenStates
        if let positionEmbeddings {
            hiddenStates = hiddenStates + positionEmbeddings
        }

        let batch = hiddenStates.dim(0)
        let numQueries = hiddenStates.dim(1)
        let sequenceLength = encoderHiddenStates.dim(1)

        var value = valueProj(encoderHiddenStates) 
        value = value.reshaped(batch, sequenceLength, nHeads, headDim)

        var offsets = samplingOffsets(hiddenStates).reshaped(batch, numQueries, nHeads, nLevels, nPoints, 2)
        var weights = attentionWeights(hiddenStates).reshaped(batch, numQueries, nHeads, nLevels * nPoints)
        weights = softmax(weights, axes: [-1]).reshaped(batch, numQueries, nHeads, nLevels, nPoints)

        
        let rpXY = referencePoints[0..., 0..., 0..., 0 ..< 2]
        let rpWH = referencePoints[0..., 0..., 0..., 2 ..< 4]
        let samplingLocations = rpXY[0..., 0..., .newAxis, 0..., .newAxis, 0...]
            + (offsets / Float(nPoints)) * rpWH[0..., 0..., .newAxis, 0..., .newAxis, 0...] * 0.5

        let samplingGrids = (samplingLocations * 2.0) - 1.0 

        
        var valueByLevel: [MLXArray] = []
        valueByLevel.reserveCapacity(nLevels)
        var cursor = 0
        for (h, w) in spatialShapes {
            let count = h * w
            valueByLevel.append(value[0..., cursor ..< cursor + count, 0..., 0...])
            cursor += count
        }
        precondition(cursor == sequenceLength, "spatialShapes do not sum to sequenceLength")

        var sampledByLevel: [MLXArray] = []
        sampledByLevel.reserveCapacity(nLevels)

        for (levelId, (h, w)) in spatialShapes.enumerated() {
            let vLevel = valueByLevel[levelId] 
            let v = vLevel.transposed(0, 2, 1, 3).reshaped(batch * nHeads, h, w, headDim)

            let gridLevel = samplingGrids[0..., 0..., 0..., levelId, 0..., 0...] 
            let grid = gridLevel.transposed(0, 2, 1, 3, 4).reshaped(batch * nHeads, numQueries, nPoints, 2)

            let sampled = PPDocLayoutV3Metal.gridSampleBilinear(v, grid) 
            sampledByLevel.append(sampled)
        }

        
        let sampledStack = stacked(sampledByLevel, axis: 2).reshaped(batch * nHeads, numQueries, nLevels * nPoints, headDim)

        var attn = weights.transposed(0, 2, 1, 3, 4).reshaped(batch * nHeads, numQueries, nLevels * nPoints)
        attn = attn.expandedDimensions(axis: -1)

        var output = (sampledStack * attn).sum(axes: [2]) 
        output = output.reshaped(batch, nHeads, numQueries, headDim).transposed(0, 2, 1, 3).reshaped(batch, numQueries, dModel)

        return outputProj(output)
    }
}



final class PPDocLayoutV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: PPDocLayoutV3SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: PPDocLayoutV3MultiscaleDeformableAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: PPDocLayoutV3MLP
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    private let dropout: Float

    init(config: PPDocLayoutV3Config) {
        self.dropout = config.dropout
        _selfAttn.wrappedValue = PPDocLayoutV3SelfAttention(hiddenSize: config.dModel, numHeads: config.decoderAttentionHeads)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _encoderAttn.wrappedValue = PPDocLayoutV3MultiscaleDeformableAttention(config: config, numHeads: config.decoderAttentionHeads, nPoints: config.decoderNPoints)
        _encoderAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _mlp.wrappedValue = PPDocLayoutV3MLP(config: config, hiddenSize: config.dModel, intermediateSize: config.decoderFfnDim, activationFunction: config.decoderActivationFunction)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, objectQueriesPositionEmbeddings: MLXArray, referencePoints: MLXArray, encoderHiddenStates: MLXArray, spatialShapes: [(height: Int, width: Int)]) -> MLXArray {
        var hiddenStates = hiddenStates
        var residual = hiddenStates

        hiddenStates = selfAttn(hiddenStates: hiddenStates, attentionMask: nil, positionEmbeddings: objectQueriesPositionEmbeddings)
        if dropout > 0 && training {
            hiddenStates = Dropout(p: dropout)(hiddenStates)
        }
        hiddenStates = residual + hiddenStates
        hiddenStates = selfAttnLayerNorm(hiddenStates)

        residual = hiddenStates

        hiddenStates = encoderAttn(
            hiddenStates: hiddenStates,
            encoderHiddenStates: encoderHiddenStates,
            positionEmbeddings: objectQueriesPositionEmbeddings,
            referencePoints: referencePoints,
            spatialShapes: spatialShapes
        )
        if dropout > 0 && training {
            hiddenStates = Dropout(p: dropout)(hiddenStates)
        }
        hiddenStates = residual + hiddenStates
        hiddenStates = encoderAttnLayerNorm(hiddenStates)

        residual = hiddenStates
        hiddenStates = mlp(hiddenStates)
        if dropout > 0 && training {
            hiddenStates = Dropout(p: dropout)(hiddenStates)
        }
        hiddenStates = residual + hiddenStates
        hiddenStates = finalLayerNorm(hiddenStates)
        return hiddenStates
    }
}

final class PPDocLayoutV3Decoder: Module {
    @ModuleInfo(key: "layers") var layers: [PPDocLayoutV3DecoderLayer]
    @ModuleInfo(key: "query_pos_head") var queryPosHead: PPDocLayoutV3MLPPredictionHead

    private let numQueries: Int

    init(config: PPDocLayoutV3Config) {
        _layers.wrappedValue = (0..<config.decoderLayers).map { _ in PPDocLayoutV3DecoderLayer(config: config) }
        _queryPosHead.wrappedValue = PPDocLayoutV3MLPPredictionHead(4, 2 * config.dModel, config.dModel, numLayers: 2)
        self.numQueries = config.numQueries
        super.init()
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        encoderHiddenStates: MLXArray,
        referencePointsUnact: MLXArray,
        spatialShapes: [(height: Int, width: Int)],
        bboxEmbed: PPDocLayoutV3MLPPredictionHead,
        classEmbed: Linear,
        orderHead: [Linear],
        globalPointer: PPDocLayoutV3GlobalPointer,
        maskQueryHead: PPDocLayoutV3MLPPredictionHead,
        norm: LayerNorm,
        maskFeat: MLXArray
    ) -> (logits: MLXArray, predBoxes: MLXArray, orderLogits: MLXArray, outMasks: MLXArray) {
        var hiddenStates = inputsEmbeds

        var referencePoints = sigmoid(referencePointsUnact) 

        var lastLogits: MLXArray = .mlxNone
        var lastPredBoxes: MLXArray = referencePoints
        var lastOrderLogits: MLXArray = .mlxNone
        var lastOutMasks: MLXArray = .mlxNone

        let batch = hiddenStates.dim(0)
        let maskH = maskFeat.dim(1)
        let maskW = maskFeat.dim(2)
        let proto = maskFeat.dim(3)
        let maskFlat = maskFeat.reshaped(batch, maskH * maskW, proto).transposed(0, 2, 1) 

        let seq = hiddenStates.dim(1)
        let mask2d = tril(MLXArray.ones([seq, seq], type: Float.self), k: 0) .> 0
        let lowerTriMask = mask2d[.newAxis, 0..., 0...]

        for (idx, layer) in layers.enumerated() {
            let referencePointsInput = referencePoints.expandedDimensions(axis: 2) 
            let queryPos = queryPosHead(referencePoints)

            hiddenStates = layer(
                hiddenStates,
                objectQueriesPositionEmbeddings: queryPos,
                referencePoints: referencePointsInput,
                encoderHiddenStates: encoderHiddenStates,
                spatialShapes: spatialShapes
            )

            let predictedCorners = bboxEmbed(hiddenStates)
            let newReferencePoints = sigmoid(predictedCorners + ppdoclayoutInverseSigmoid(referencePoints))
            referencePoints = newReferencePoints

            let outQuery = norm(hiddenStates)
            lastLogits = classEmbed(outQuery)
            lastPredBoxes = newReferencePoints

            if idx < orderHead.count {
                let validQuery = outQuery 
                lastOrderLogits = globalPointer(orderHead[idx](validQuery), lowerTriangleMask: lowerTriMask)
            }

            if idx == layers.count - 1 {
                let maskQueryEmbed = maskQueryHead(outQuery) 
                lastOutMasks = matmul(maskQueryEmbed, maskFlat).reshaped(batch, numQueries, maskH, maskW)
            }
        }

        return (logits: lastLogits, predBoxes: lastPredBoxes, orderLogits: lastOrderLogits, outMasks: lastOutMasks)
    }
}



final class PPDocLayoutV3GlobalPointer: Module, UnaryLayer {
    private let headSize: Int

    @ModuleInfo(key: "dense") var dense: Linear

    init(config: PPDocLayoutV3Config) {
        self.headSize = config.globalPointerHeadSize
        _dense.wrappedValue = Linear(config.dModel, headSize * 2, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, lowerTriangleMask: MLXArray) -> MLXArray {
        
        let batch = x.dim(0)
        let seq = x.dim(1)

        var qk = dense(x).reshaped(batch, seq, 2, headSize)
        let q = qk[0..., 0..., 0, 0...]
        let k = qk[0..., 0..., 1, 0...]

        var logits = matmul(q, k.transposed(0, 2, 1)) / sqrt(Float(headSize))

        
        logits = which(lowerTriangleMask, MLXArray(-1e4, dtype: logits.dtype), logits)
        return logits
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seq = x.dim(1)
        let mask2d = tril(MLXArray.ones([seq, seq], type: Float.self), k: 0) .> 0
        let lowerTriMask = mask2d[.newAxis, 0..., 0...]
        return callAsFunction(x, lowerTriangleMask: lowerTriMask)
    }
}



final class HGNetV2ConvLayer: Module, UnaryLayer {
    @ModuleInfo(key: "convolution") var convolution: Conv2d
    @ModuleInfo(key: "normalization") var normalization: BatchNorm
    private let activation: String?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, groups: Int = 1, activation: String? = "relu") {
        _convolution.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: .init((kernelSize - 1) / 2),
            groups: groups,
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(featureCount: outChannels, eps: 1e-5, momentum: 0.1, affine: true, trackRunningStats: true)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = normalization(convolution(x))
        guard let activation else { return y }
        return ppdoclayoutActivation(activation, y)
    }
}

final class HGNetV2ConvLayerLight: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: HGNetV2ConvLayer
    @ModuleInfo(key: "conv2") var conv2: HGNetV2ConvLayer

    init(inChannels: Int, outChannels: Int, kernelSize: Int) {
        _conv1.wrappedValue = HGNetV2ConvLayer(inChannels: inChannels, outChannels: outChannels, kernelSize: 1, stride: 1, groups: 1, activation: nil)
        _conv2.wrappedValue = HGNetV2ConvLayer(inChannels: outChannels, outChannels: outChannels, kernelSize: kernelSize, stride: 1, groups: outChannels, activation: "relu")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(conv1(x))
    }
}

final class HGNetV2Embeddings: Module, UnaryLayer {
    @ModuleInfo(key: "stem1") var stem1: HGNetV2ConvLayer
    @ModuleInfo(key: "stem2a") var stem2a: HGNetV2ConvLayer
    @ModuleInfo(key: "stem2b") var stem2b: HGNetV2ConvLayer
    @ModuleInfo(key: "stem3") var stem3: HGNetV2ConvLayer
    @ModuleInfo(key: "stem4") var stem4: HGNetV2ConvLayer

    private let pool = MaxPool2d(kernelSize: 2, stride: 1, padding: 0)

    override init() {
        
        _stem1.wrappedValue = HGNetV2ConvLayer(inChannels: 3, outChannels: 32, kernelSize: 3, stride: 2, activation: "relu")
        _stem2a.wrappedValue = HGNetV2ConvLayer(inChannels: 32, outChannels: 16, kernelSize: 2, stride: 1, activation: "relu")
        _stem2b.wrappedValue = HGNetV2ConvLayer(inChannels: 16, outChannels: 32, kernelSize: 2, stride: 1, activation: "relu")
        _stem3.wrappedValue = HGNetV2ConvLayer(inChannels: 64, outChannels: 32, kernelSize: 3, stride: 2, activation: "relu")
        _stem4.wrappedValue = HGNetV2ConvLayer(inChannels: 32, outChannels: 48, kernelSize: 1, stride: 1, activation: "relu")
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var x = stem1(pixelValues)
        x = ppdoclayoutPadBottomRight(x, by: 1)
        var y = stem2a(x)
        y = ppdoclayoutPadBottomRight(y, by: 1)
        y = stem2b(y)
        let pooled = pool(x)
        x = concatenated([pooled, y], axis: -1)
        x = stem3(x)
        x = stem4(x)
        return x
    }
}

final class HGNetV2BasicLayer: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [HGNetV2ConvLayer]
    @ModuleInfo(key: "aggregation") var aggregation: [HGNetV2ConvLayer]
    private let residual: Bool

    init(inChannels: Int, middleChannels: Int, outChannels: Int, layerNum: Int, kernelSize: Int, residual: Bool) {
        self.residual = residual
        _layers.wrappedValue = (0..<layerNum).map { i in
            let inC = (i == 0) ? inChannels : middleChannels
            return HGNetV2ConvLayer(inChannels: inC, outChannels: middleChannels, kernelSize: kernelSize, stride: 1, activation: "relu")
        }

        let totalChannels = inChannels + layerNum * middleChannels
        let squeeze = HGNetV2ConvLayer(inChannels: totalChannels, outChannels: outChannels / 2, kernelSize: 1, stride: 1, activation: "relu")
        let excite = HGNetV2ConvLayer(inChannels: outChannels / 2, outChannels: outChannels, kernelSize: 1, stride: 1, activation: "relu")
        _aggregation.wrappedValue = [squeeze, excite]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        var hidden = x
        var outputs: [MLXArray] = [x]
        outputs.reserveCapacity(layers.count + 1)
        for layer in layers {
            hidden = layer(hidden)
            outputs.append(hidden)
        }
        hidden = concatenated(outputs, axis: -1)
        for layer in aggregation {
            hidden = layer(hidden)
        }
        if residual {
            hidden = hidden + identity
        }
        return hidden
    }
}

final class HGNetV2BasicLayerLight: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [HGNetV2ConvLayerLight]
    @ModuleInfo(key: "aggregation") var aggregation: [HGNetV2ConvLayer]
    private let residual: Bool

    init(inChannels: Int, middleChannels: Int, outChannels: Int, layerNum: Int, kernelSize: Int, residual: Bool) {
        self.residual = residual
        _layers.wrappedValue = (0..<layerNum).map { i in
            let inC = (i == 0) ? inChannels : middleChannels
            return HGNetV2ConvLayerLight(inChannels: inC, outChannels: middleChannels, kernelSize: kernelSize)
        }

        let totalChannels = inChannels + layerNum * middleChannels
        let squeeze = HGNetV2ConvLayer(inChannels: totalChannels, outChannels: outChannels / 2, kernelSize: 1, stride: 1, activation: "relu")
        let excite = HGNetV2ConvLayer(inChannels: outChannels / 2, outChannels: outChannels, kernelSize: 1, stride: 1, activation: "relu")
        _aggregation.wrappedValue = [squeeze, excite]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        var hidden = x
        var outputs: [MLXArray] = [x]
        outputs.reserveCapacity(layers.count + 1)
        for layer in layers {
            hidden = layer(hidden)
            outputs.append(hidden)
        }
        hidden = concatenated(outputs, axis: -1)
        for layer in aggregation {
            hidden = layer(hidden)
        }
        if residual {
            hidden = hidden + identity
        }
        return hidden
    }
}

final class HGNetV2Stage: Module, UnaryLayer {
    @ModuleInfo(key: "downsample") var downsample: Module
    @ModuleInfo(key: "blocks") var blocks: [Module]

    init(
        inChannels: Int,
        midChannels: Int,
        outChannels: Int,
        numBlocks: Int,
        numLayers: Int,
        downsample: Bool,
        lightBlock: Bool,
        kernelSize: Int
    ) {
        if downsample {
            _downsample.wrappedValue = HGNetV2ConvLayer(inChannels: inChannels, outChannels: inChannels, kernelSize: 3, stride: 2, groups: inChannels, activation: nil)
        } else {
            _downsample.wrappedValue = Identity()
        }

        var out: [Module] = []
        out.reserveCapacity(numBlocks)
        for i in 0..<numBlocks {
            let inC = (i == 0) ? inChannels : outChannels
            let residual = (i != 0)
            if lightBlock {
                out.append(HGNetV2BasicLayerLight(inChannels: inC, middleChannels: midChannels, outChannels: outChannels, layerNum: numLayers, kernelSize: kernelSize, residual: residual))
            } else {
                out.append(HGNetV2BasicLayer(inChannels: inC, middleChannels: midChannels, outChannels: outChannels, layerNum: numLayers, kernelSize: kernelSize, residual: residual))
            }
        }
        _blocks.wrappedValue = out
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden: MLXArray
        if let down = downsample as? (any UnaryLayer) {
            hidden = down(x)
        } else {
            fatalError("downsample is not unary: \(type(of: downsample))")
        }

        for block in blocks {
            if let b = block as? (any UnaryLayer) {
                hidden = b(hidden)
            } else {
                fatalError("block is not unary: \(type(of: block))")
            }
        }
        return hidden
    }
}

final class HGNetV2Encoder: Module {
    @ModuleInfo(key: "stages") var stages: [HGNetV2Stage]

    override init() {
        
        
        
        
        
        
        
        
        
        _stages.wrappedValue = [
            HGNetV2Stage(inChannels: 48, midChannels: 48, outChannels: 128, numBlocks: 1, numLayers: 6, downsample: false, lightBlock: false, kernelSize: 3),
            HGNetV2Stage(inChannels: 128, midChannels: 96, outChannels: 512, numBlocks: 1, numLayers: 6, downsample: true, lightBlock: false, kernelSize: 3),
            HGNetV2Stage(inChannels: 512, midChannels: 192, outChannels: 1024, numBlocks: 3, numLayers: 6, downsample: true, lightBlock: true, kernelSize: 5),
            HGNetV2Stage(inChannels: 1024, midChannels: 384, outChannels: 2048, numBlocks: 1, numLayers: 6, downsample: true, lightBlock: true, kernelSize: 5),
        ]
        super.init()
    }

    func forward(_ x: MLXArray) -> ([MLXArray], MLXArray) {
        var hidden = x
        var hiddenStates: [MLXArray] = []
        hiddenStates.reserveCapacity(stages.count + 1)
        for stage in stages {
            hiddenStates.append(hidden)
            hidden = stage(hidden)
        }
        hiddenStates.append(hidden)
        return (hiddenStates, hidden)
    }
}

final class HGNetV2Backbone: Module {
    @ModuleInfo(key: "embedder") var embedder: HGNetV2Embeddings
    @ModuleInfo(key: "encoder") var encoder: HGNetV2Encoder

    private let outFeatures: [String]

    init(outFeatures: [String]) {
        _embedder.wrappedValue = HGNetV2Embeddings()
        _encoder.wrappedValue = HGNetV2Encoder()
        self.outFeatures = outFeatures
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> [MLXArray] {
        let embedding = embedder(pixelValues)
        let (hiddenStates, _) = encoder.forward(embedding)

        let stageNames = ["stem", "stage1", "stage2", "stage3", "stage4"]
        precondition(hiddenStates.count == stageNames.count, "hiddenStates mismatch")

        var featureMaps: [MLXArray] = []
        featureMaps.reserveCapacity(outFeatures.count)
        for (idx, stage) in stageNames.enumerated() {
            if outFeatures.contains(stage) {
                featureMaps.append(hiddenStates[idx])
            }
        }
        return featureMaps
    }
}

final class PPDocLayoutV3ConvEncoder: Module {
    @ModuleInfo(key: "model") var model: HGNetV2Backbone

    init(config: PPDocLayoutV3Config) {
        let outFeatures = config.backboneConfig?.outFeatures ?? ["stage1", "stage2", "stage3", "stage4"]
        _model.wrappedValue = HGNetV2Backbone(outFeatures: outFeatures)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> [MLXArray] {
        model(pixelValues)
    }
}



final class PPDocLayoutV3Model: Module {
    @ModuleInfo(key: "backbone") var backbone: PPDocLayoutV3ConvEncoder
    @ModuleInfo(key: "encoder_input_proj") var encoderInputProj: [PPDocLayoutV3ConvBN]
    @ModuleInfo(key: "encoder") var encoder: PPDocLayoutV3HybridEncoder

    @ModuleInfo(key: "enc_output") var encOutput: (Linear, LayerNorm)
    @ModuleInfo(key: "enc_score_head") var encScoreHead: Linear
    @ModuleInfo(key: "enc_bbox_head") var encBBoxHead: PPDocLayoutV3MLPPredictionHead

    @ModuleInfo(key: "decoder_input_proj") var decoderInputProj: [PPDocLayoutV3ConvBN]
    @ModuleInfo(key: "decoder") var decoder: PPDocLayoutV3Decoder

    @ModuleInfo(key: "decoder_order_head") var decoderOrderHead: [Linear]
    @ModuleInfo(key: "decoder_global_pointer") var decoderGlobalPointer: PPDocLayoutV3GlobalPointer
    @ModuleInfo(key: "decoder_norm") var decoderNorm: LayerNorm
    @ModuleInfo(key: "mask_query_head") var maskQueryHead: PPDocLayoutV3MLPPredictionHead

    private let config: PPDocLayoutV3Config

    init(config: PPDocLayoutV3Config) {
        self.config = config
        _backbone.wrappedValue = PPDocLayoutV3ConvEncoder(config: config)

        
        _encoderInputProj.wrappedValue = [
            PPDocLayoutV3ConvBN(config: config, inChannels: 512, outChannels: config.encoderHiddenDim, kernelSize: 1, stride: 1, padding: 0, bias: false),
            PPDocLayoutV3ConvBN(config: config, inChannels: 1024, outChannels: config.encoderHiddenDim, kernelSize: 1, stride: 1, padding: 0, bias: false),
            PPDocLayoutV3ConvBN(config: config, inChannels: 2048, outChannels: config.encoderHiddenDim, kernelSize: 1, stride: 1, padding: 0, bias: false),
        ]

        _encoder.wrappedValue = PPDocLayoutV3HybridEncoder(config: config)

        _encOutput.wrappedValue = (
            Linear(config.dModel, config.dModel, bias: true),
            LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        )
        _encScoreHead.wrappedValue = Linear(config.dModel, config.numLabels, bias: true)
        _encBBoxHead.wrappedValue = PPDocLayoutV3MLPPredictionHead(config.dModel, config.dModel, 4, numLayers: 3)

        _decoderInputProj.wrappedValue = [
            PPDocLayoutV3ConvBN(config: config, inChannels: 256, outChannels: config.dModel, kernelSize: 1, stride: 1, padding: 0, bias: false),
            PPDocLayoutV3ConvBN(config: config, inChannels: 256, outChannels: config.dModel, kernelSize: 1, stride: 1, padding: 0, bias: false),
            PPDocLayoutV3ConvBN(config: config, inChannels: 256, outChannels: config.dModel, kernelSize: 1, stride: 1, padding: 0, bias: false),
        ]
        _decoder.wrappedValue = PPDocLayoutV3Decoder(config: config)

        _decoderOrderHead.wrappedValue = (0..<config.decoderLayers).map { _ in Linear(config.dModel, config.dModel, bias: true) }
        _decoderGlobalPointer.wrappedValue = PPDocLayoutV3GlobalPointer(config: config)
        _decoderNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _maskQueryHead.wrappedValue = PPDocLayoutV3MLPPredictionHead(config.dModel, config.dModel, config.numPrototypes, numLayers: 3)

        super.init()
    }

    private func generateAnchors(spatialShapes: [(height: Int, width: Int)], gridSize: Float = 0.05, dtype: DType) -> (anchors: MLXArray, validMask: MLXArray) {
        var anchors: [MLXArray] = []
        anchors.reserveCapacity(spatialShapes.count)

        for (level, (h, w)) in spatialShapes.enumerated() {
            let gridX = broadcast(arange(w, dtype: .float32).reshaped(1, w), to: [h, w])
            let gridY = broadcast(arange(h, dtype: .float32).reshaped(h, 1), to: [h, w])
            var grid = stacked([gridX, gridY], axis: -1) 
            grid = (grid + 0.5)
            grid[0..., 0..., 0] = grid[0..., 0..., 0] / Float(w)
            grid[0..., 0..., 1] = grid[0..., 0..., 1] / Float(h)

            let wh = MLXArray.ones(like: grid) * (gridSize * pow(2.0, Float(level)))
            let a = concatenated([grid, wh], axis: -1).reshaped(1, h * w, 4)
            anchors.append(a)
        }

        var all = concatenated(anchors, axis: 1).asType(dtype)
        let eps: Float = 1e-2
        let valid = ((all .> eps) * (all .< (1.0 - eps))).all(axes: [-1], keepDims: true)

        all = log(all / (1.0 - all))
        let maxValue = MLXArray(Float.greatestFiniteMagnitude, dtype: dtype)
        all = which(valid, all, maxValue)

        return (anchors: all, validMask: valid)
    }

    func callAsFunction(pixelValues: MLXArray) -> PPDocLayoutV3ForObjectDetectionOutput {
        
        let batch = pixelValues.dim(0)

        let features = backbone(pixelValues) 
        precondition(features.count == 4, "Expected 4 backbone feature maps")
        let x4Feat = features[0]
        let stages = Array(features.dropFirst()) 

        var projFeats: [MLXArray] = []
        projFeats.reserveCapacity(stages.count)
        for (i, f) in stages.enumerated() {
            projFeats.append(encoderInputProj[i](f))
        }

        let encoderOut = encoder(projFeats, x4Feat: x4Feat)
        var sources: [MLXArray] = []
        sources.reserveCapacity(config.numFeatureLevels)

        for (i, src) in encoderOut.panFeatureMaps.enumerated() {
            sources.append(decoderInputProj[i](src))
        }

        
        var sourceFlatten: [MLXArray] = []
        sourceFlatten.reserveCapacity(sources.count)
        var spatialShapes: [(height: Int, width: Int)] = []
        spatialShapes.reserveCapacity(sources.count)

        for src in sources {
            let h = src.dim(1)
            let w = src.dim(2)
            let c = src.dim(3)
            spatialShapes.append((height: h, width: w))
            sourceFlatten.append(src.reshaped(batch, h * w, c))
        }

        let encoderHiddenStates = concatenated(sourceFlatten, axis: 1)

        let (anchors, validMask) = generateAnchors(spatialShapes: spatialShapes, dtype: encoderHiddenStates.dtype)

        let memory = encoderHiddenStates * validMask.asType(encoderHiddenStates.dtype)
        let outputMemory = encOutput.1(encOutput.0(memory))

        let encOutputsClass = encScoreHead(outputMemory)
        let encOutputsCoordLogits = encBBoxHead(outputMemory) + anchors

        
        let scoresMax = encOutputsClass.max(axis: -1) 
        let topk = ppdoclayoutTopKIndices(scoresMax, k: config.numQueries)

        let referencePointsUnact = ppdoclayoutGather3D(encOutputsCoordLogits, indices: topk) 
        let target = ppdoclayoutGather3D(outputMemory, indices: topk) 

        let outQuery = decoderNorm(target)
        let maskQueryEmbed = maskQueryHead(outQuery) 

        var initReferencePointsUnact = referencePointsUnact
        if config.maskEnhanced {
            let maskFeat = encoderOut.maskFeat 
            let maskH = maskFeat.dim(1)
            let maskW = maskFeat.dim(2)
            let proto = maskFeat.dim(3)

            let maskFlat = maskFeat.reshaped(batch, maskH * maskW, proto).transposed(0, 2, 1) 
            let encOutMasks = matmul(maskQueryEmbed, maskFlat).reshaped(batch, config.numQueries, maskH, maskW)
            let ref = ppdoclayoutMaskToBoxCoordinate(encOutMasks .> 0, dtype: initReferencePointsUnact.dtype)
            initReferencePointsUnact = ppdoclayoutInverseSigmoid(ref)
        }

        let decoderOut = decoder(
            inputsEmbeds: target,
            encoderHiddenStates: encoderHiddenStates,
            referencePointsUnact: initReferencePointsUnact,
            spatialShapes: spatialShapes,
            bboxEmbed: encBBoxHead,
            classEmbed: encScoreHead,
            orderHead: decoderOrderHead,
            globalPointer: decoderGlobalPointer,
            maskQueryHead: maskQueryHead,
            norm: decoderNorm,
            maskFeat: encoderOut.maskFeat
        )

        return PPDocLayoutV3ForObjectDetectionOutput(
            logits: decoderOut.logits,
            predBoxes: decoderOut.predBoxes,
            orderLogits: decoderOut.orderLogits,
            outMasks: decoderOut.outMasks
        )
    }
}

final class PPDocLayoutV3ForObjectDetection: Module {
    @ModuleInfo(key: "model") var model: PPDocLayoutV3Model

    init(config: PPDocLayoutV3Config) {
        _model.wrappedValue = PPDocLayoutV3Model(config: config)
        super.init()
    }

    func callAsFunction(pixelValues: MLXArray) -> PPDocLayoutV3ForObjectDetectionOutput {
        model(pixelValues: pixelValues)
    }
}



private func ppdoclayoutGather3D(_ array: MLXArray, indices: MLXArray) -> MLXArray {
    
    precondition(array.ndim == 3)
    precondition(indices.ndim == 2)
    let b = array.dim(0)
    let k = indices.dim(1)
    let c = array.dim(2)
    let idx = indices.reshaped(b, k, 1)
    let idxB = broadcast(idx, to: [b, k, c])
    return takeAlong(array, idxB, axis: 1)
}

private func ppdoclayoutTopKIndices(_ values: MLXArray, k: Int) -> MLXArray {
    
    precondition(values.ndim == 2)
    let neg = -values
    let sortedIdx = argSort(neg, axis: -1) 
    return sortedIdx[0..., 0..<k]
}

private func ppdoclayoutMaskToBoxCoordinate(_ mask: MLXArray, dtype: DType) -> MLXArray {
    
    precondition(mask.ndim == 4)
    let h = mask.dim(2)
    let w = mask.dim(3)

    let yCoords = broadcast(arange(h, dtype: dtype).reshaped(1, 1, h, 1), to: [mask.dim(0), mask.dim(1), h, w])
    let xCoords = broadcast(arange(w, dtype: dtype).reshaped(1, 1, 1, w), to: [mask.dim(0), mask.dim(1), h, w])

    let maskF = mask.asType(dtype)
    let xMasked = xCoords * maskF
    let yMasked = yCoords * maskF

    let xMax = xMasked.flattened(start: -2).max(axis: -1) + 1.0
    let yMax = yMasked.flattened(start: -2).max(axis: -1) + 1.0

    let maxValue = MLXArray(Float.greatestFiniteMagnitude, dtype: dtype)
    let xMin = which(mask, xMasked, maxValue).flattened(start: -2).min(axis: -1)
    let yMin = which(mask, yMasked, maxValue).flattened(start: -2).min(axis: -1)

    var bbox = stacked([xMin, yMin, xMax, yMax], axis: -1)
    let nonEmpty = mask.any(axes: [-2, -1]).expandedDimensions(axis: -1).asType(dtype)
    bbox = bbox * nonEmpty

    let norm = MLXArray([Float(w), Float(h), Float(w), Float(h)]).asType(dtype)
    let normalized = bbox / norm

    let x1 = normalized[0..., 0..., 0]
    let y1 = normalized[0..., 0..., 1]
    let x2 = normalized[0..., 0..., 2]
    let y2 = normalized[0..., 0..., 3]

    let centerX = (x1 + x2) / 2.0
    let centerY = (y1 + y2) / 2.0
    let boxW = x2 - x1
    let boxH = y2 - y1

    return stacked([centerX, centerY, boxW, boxH], axis: -1)
}
