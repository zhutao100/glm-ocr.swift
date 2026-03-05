import Foundation
import Tokenizers

public struct GLMOCRModelConfig: Decodable, Sendable {
    public struct RopeParameters: Decodable, Sendable {
        public let ropeType: String?
        public let mropeSection: [Int]?
        public let partialRotaryFactor: Double?
        public let ropeTheta: Double?
    }

    public struct TextConfig: Decodable, Sendable {
        public let modelType: String?
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int?
        public let hiddenAct: String?
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Double
        public let useCache: Bool?
        public let attentionBias: Bool?
        public let attentionDropout: Double?
        public let ropeParameters: RopeParameters?
        public let padTokenId: Int?
        public let eosTokenId: [Int]?
        public let tieWordEmbeddings: Bool?
    }

    public struct VisionConfig: Decodable, Sendable {
        enum CodingKeys: String, CodingKey {
            case modelType
            case depth
            case hiddenSize
            case intermediateSize
            case numHeads
            case inChannels
            case patchSize
            case temporalPatchSize
            case outHiddenSize
            case spatialMergeSize
            case rmsNormEps
            case attentionBias
            case attentionDropout
            case hiddenAct
        }

        public let modelType: String?
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let inChannels: Int
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let outHiddenSize: Int
        public let spatialMergeSize: Int
        public let rmsNormEps: Double
        public let attentionBias: Bool?
        public let attentionDropout: Double?
        public let hiddenAct: String?

        public init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            self.depth = try container.decode(Int.self, forKey: .depth)
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.numHeads = try container.decode(Int.self, forKey: .numHeads)
            self.inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
            self.patchSize = try container.decode(Int.self, forKey: .patchSize)
            self.temporalPatchSize = try container.decode(Int.self, forKey: .temporalPatchSize)
            self.outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
            self.spatialMergeSize = try container.decode(Int.self, forKey: .spatialMergeSize)
            self.rmsNormEps = try container.decode(Double.self, forKey: .rmsNormEps)
            self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
            self.attentionDropout = try container.decodeIfPresent(Double.self, forKey: .attentionDropout)
            self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
        }
    }

    public let modelType: String
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig

    public let imageTokenId: Int
    public let videoTokenId: Int
    public let imageStartTokenId: Int
    public let imageEndTokenId: Int
    public let videoStartTokenId: Int
    public let videoEndTokenId: Int
}

public struct GLMOCRPreprocessorConfig: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let shortestEdge: Int
        public let longestEdge: Int
    }

    public let size: Size
    public let doRescale: Bool
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let mergeSize: Int
    public let imageMean: [Double]
    public let imageStd: [Double]
    public let imageProcessorType: String?
    public let processorClass: String?
}

public struct GLMOCRGenerationConfig: Decodable, Sendable {
    public enum OneOrManyInt: Decodable, Sendable {
        case one(Int)
        case many([Int])

        public var values: [Int] {
            switch self {
            case .one(let value): [value]
            case .many(let values): values
            }
        }

        public init(from decoder: Swift.Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let intValue = try? container.decode(Int.self) {
                self = .one(intValue)
                return
            }
            self = .many(try container.decode([Int].self))
        }
    }

    public let eosTokenId: OneOrManyInt?
    public let padTokenId: Int?
    public let doSample: Bool?
}

public struct GLMOCRResources {
    public let modelURL: URL
    public let modelConfig: GLMOCRModelConfig
    public let preprocessorConfig: GLMOCRPreprocessorConfig
    public let generationConfig: GLMOCRGenerationConfig
    public let tokenizer: Tokenizer

    public init(modelURL: URL, strictTokenizer: Bool = true) async throws {
        self.modelURL = modelURL
        self.modelConfig = try Self.decodeJSON(
            GLMOCRModelConfig.self, from: modelURL.appendingPathComponent("config.json"))
        self.preprocessorConfig = try Self.decodeJSON(
            GLMOCRPreprocessorConfig.self,
            from: modelURL.appendingPathComponent("preprocessor_config.json")
        )
        self.generationConfig = try Self.decodeJSON(
            GLMOCRGenerationConfig.self,
            from: modelURL.appendingPathComponent("generation_config.json")
        )
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelURL, strict: strictTokenizer)
    }

    private static func decodeJSON<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: data)
    }

}
