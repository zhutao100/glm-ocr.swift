import Foundation
import Tokenizers

public enum GLMOCRPrompt {
    public static func makeOCRPromptText(prompt: String) -> String {
        "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>\(prompt)<|assistant|>\n"
    }

    public static func tokenizeOCRPrompt(
        tokenizer: Tokenizer,
        prompt: String
    ) -> [Int] {
        let rendered = makeOCRPromptText(prompt: prompt)
        return tokenizer.encode(text: rendered, addSpecialTokens: false)
    }

    public static func makeMessages(imagePath: String, prompt: String) -> [Message] {
        let imageItem: [String: any Sendable] = [
            "type": "image",
            "url": imagePath,
        ]
        let textItem: [String: any Sendable] = [
            "type": "text",
            "text": prompt,
        ]
        let content: [any Sendable] = [imageItem, textItem]
        let message: Message = [
            "role": "user",
            "content": content,
        ]
        return [message]
    }

    public static func expandImageTokens(
        inputIDs: [Int],
        imageTokenId: Int,
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        mergeSize: Int
    ) throws -> [Int] {
        let mergeLength = mergeSize * mergeSize
        var expanded: [Int] = []
        expanded.reserveCapacity(inputIDs.count)

        var imageIndex = 0
        for tokenId in inputIDs {
            if tokenId == imageTokenId {
                guard imageIndex < imageGridTHW.count else {
                    throw GLMOCRPromptError.placeholderCountMismatch
                }
                let grid = imageGridTHW[imageIndex]
                let numImageTokens = (grid.t * grid.h * grid.w) / mergeLength
                expanded.append(contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
                imageIndex += 1
            } else {
                expanded.append(tokenId)
            }
        }

        if imageIndex != imageGridTHW.count {
            throw GLMOCRPromptError.placeholderCountMismatch
        }

        return expanded
    }
}

public enum GLMOCRPromptError: Error, Sendable {
    case placeholderCountMismatch
}
