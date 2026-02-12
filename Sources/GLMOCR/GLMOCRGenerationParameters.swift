public struct GLMOCRGenerationParameters: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var decodeSyncInterval: Int

    public init(
        temperature: Float = 0,
        topP: Float = 1,
        topK: Int = 0,
        repetitionPenalty: Float = 1,
        seed: UInt64? = nil,
        decodeSyncInterval: Int = 16
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.decodeSyncInterval = decodeSyncInterval
    }

    public static let greedy = GLMOCRGenerationParameters()

    public static let `default` = GLMOCRGenerationParameters(
        temperature: 0.8,
        topP: 0.9,
        topK: 50,
        repetitionPenalty: 1.1,
        seed: 0
    )

    public static let sampledStable = GLMOCRGenerationParameters(
        temperature: 0.8,
        topP: 0.9,
        topK: 50,
        repetitionPenalty: 0.98,
        seed: 42
    )
}
