import Foundation

public struct GLMOCR {
    public init() {}

    public func recognize(imageAt imageURL: URL, prompt: String = GLMOCRPromptPresets.default) async throws -> String {
        _ = imageURL
        _ = prompt
        throw GLMOCRNotImplementedError()
    }
}

public struct GLMOCRNotImplementedError: Error {
    public init() {}
}
