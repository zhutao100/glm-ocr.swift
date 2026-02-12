import Foundation

public struct PPDocLayoutV3Config: Decodable, Sendable {
    public struct BackboneConfig: Decodable, Sendable {
        public var outFeatures: [String]?

        enum CodingKeys: String, CodingKey {
            case outFeatures = "out_features"
        }
    }

    public var backboneConfig: BackboneConfig?

    public var encoderHiddenDim: Int
    public var encoderInChannels: [Int]
    public var featureStrides: [Int]
    public var encoderLayers: Int
    public var encoderFfnDim: Int
    public var encoderAttentionHeads: Int
    public var dropout: Float
    public var activationDropout: Float
    public var encodeProjLayers: [Int]
    public var positionalEncodingTemperature: Int
    public var encoderActivationFunction: String
    public var activationFunction: String
    public var evalSize: [Int]?
    public var normalizeBefore: Bool
    public var hiddenExpansion: Double
    public var maskFeatureChannels: [Int]
    public var x4FeatDim: Int

    public var dModel: Int
    public var numPrototypes: Int
    public var maskEnhanced: Bool
    public var numQueries: Int
    public var decoderInChannels: [Int]
    public var decoderFfnDim: Int
    public var numFeatureLevels: Int
    public var decoderNPoints: Int
    public var decoderLayers: Int
    public var decoderAttentionHeads: Int
    public var decoderActivationFunction: String
    public var attentionDropout: Float
    public var learnInitialQuery: Bool
    public var anchorImageSize: [Int]?
    public var disableCustomKernels: Bool
    public var globalPointerHeadSize: Int
    public var gpDropoutValue: Float

    public var batchNormEps: Float
    public var layerNormEps: Float

    public var id2label: [Int: String]

    public var numLabels: Int { id2label.count }

    enum CodingKeys: String, CodingKey {
        case backboneConfig = "backbone_config"
        case encoderHiddenDim = "encoder_hidden_dim"
        case encoderInChannels = "encoder_in_channels"
        case featureStrides = "feature_strides"
        case encoderLayers = "encoder_layers"
        case encoderFfnDim = "encoder_ffn_dim"
        case encoderAttentionHeads = "encoder_attention_heads"
        case dropout
        case activationDropout = "activation_dropout"
        case encodeProjLayers = "encode_proj_layers"
        case positionalEncodingTemperature = "positional_encoding_temperature"
        case encoderActivationFunction = "encoder_activation_function"
        case activationFunction = "activation_function"
        case evalSize = "eval_size"
        case normalizeBefore = "normalize_before"
        case hiddenExpansion = "hidden_expansion"
        case maskFeatureChannels = "mask_feature_channels"
        case x4FeatDim = "x4_feat_dim"
        case dModel = "d_model"
        case numPrototypes = "num_prototypes"
        case maskEnhanced = "mask_enhanced"
        case numQueries = "num_queries"
        case decoderInChannels = "decoder_in_channels"
        case decoderFfnDim = "decoder_ffn_dim"
        case numFeatureLevels = "num_feature_levels"
        case decoderNPoints = "decoder_n_points"
        case decoderLayers = "decoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderActivationFunction = "decoder_activation_function"
        case attentionDropout = "attention_dropout"
        case learnInitialQuery = "learn_initial_query"
        case anchorImageSize = "anchor_image_size"
        case disableCustomKernels = "disable_custom_kernels"
        case globalPointerHeadSize = "global_pointer_head_size"
        case gpDropoutValue = "gp_dropout_value"
        case batchNormEps = "batch_norm_eps"
        case layerNormEps = "layer_norm_eps"
        case id2label
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        backboneConfig = try container.decodeIfPresent(BackboneConfig.self, forKey: .backboneConfig)

        encoderHiddenDim = try container.decodeIfPresent(Int.self, forKey: .encoderHiddenDim) ?? 256
        encoderInChannels = try container.decodeIfPresent([Int].self, forKey: .encoderInChannels) ?? [512, 1024, 2048]
        featureStrides = try container.decodeIfPresent([Int].self, forKey: .featureStrides) ?? [8, 16, 32]
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 1
        encoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 1024
        encoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 8
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.0
        activationDropout = try container.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.0
        encodeProjLayers = try container.decodeIfPresent([Int].self, forKey: .encodeProjLayers) ?? [2]
        positionalEncodingTemperature = try container.decodeIfPresent(Int.self, forKey: .positionalEncodingTemperature) ?? 10_000
        encoderActivationFunction = try container.decodeIfPresent(String.self, forKey: .encoderActivationFunction) ?? "gelu"
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "silu"
        evalSize = try container.decodeIfPresent([Int].self, forKey: .evalSize)
        normalizeBefore = try container.decodeIfPresent(Bool.self, forKey: .normalizeBefore) ?? false
        hiddenExpansion = try container.decodeIfPresent(Double.self, forKey: .hiddenExpansion) ?? 1.0
        maskFeatureChannels = try container.decodeIfPresent([Int].self, forKey: .maskFeatureChannels) ?? [64, 64]
        x4FeatDim = try container.decodeIfPresent(Int.self, forKey: .x4FeatDim) ?? 128

        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 256
        numPrototypes = try container.decodeIfPresent(Int.self, forKey: .numPrototypes) ?? 32
        maskEnhanced = try container.decodeIfPresent(Bool.self, forKey: .maskEnhanced) ?? true
        numQueries = try container.decodeIfPresent(Int.self, forKey: .numQueries) ?? 300
        decoderInChannels = try container.decodeIfPresent([Int].self, forKey: .decoderInChannels) ?? [256, 256, 256]
        decoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .decoderFfnDim) ?? 1024
        numFeatureLevels = try container.decodeIfPresent(Int.self, forKey: .numFeatureLevels) ?? 3
        decoderNPoints = try container.decodeIfPresent(Int.self, forKey: .decoderNPoints) ?? 4
        decoderLayers = try container.decodeIfPresent(Int.self, forKey: .decoderLayers) ?? 6
        decoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .decoderAttentionHeads) ?? 8
        decoderActivationFunction = try container.decodeIfPresent(String.self, forKey: .decoderActivationFunction) ?? "relu"
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        learnInitialQuery = try container.decodeIfPresent(Bool.self, forKey: .learnInitialQuery) ?? false
        anchorImageSize = try container.decodeIfPresent([Int].self, forKey: .anchorImageSize)
        disableCustomKernels = try container.decodeIfPresent(Bool.self, forKey: .disableCustomKernels) ?? true
        globalPointerHeadSize = try container.decodeIfPresent(Int.self, forKey: .globalPointerHeadSize) ?? 64
        gpDropoutValue = try container.decodeIfPresent(Float.self, forKey: .gpDropoutValue) ?? 0.1

        batchNormEps = try container.decodeIfPresent(Float.self, forKey: .batchNormEps) ?? 1e-5
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5

        let rawId2Label = try container.decodeIfPresent([String: String].self, forKey: .id2label) ?? [:]
        var mapped: [Int: String] = [:]
        mapped.reserveCapacity(rawId2Label.count)
        for (k, v) in rawId2Label {
            if let idx = Int(k) {
                mapped[idx] = v
            }
        }
        id2label = mapped
    }
}

