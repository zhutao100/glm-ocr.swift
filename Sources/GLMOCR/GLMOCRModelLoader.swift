import Foundation
import MLX
import MLXNN

enum GLMOCRModelLoader {
    static func load(from modelDirectory: URL, config: GLMOCRModelConfig, dtype: DType? = nil) throws -> GLMOCRForConditionalGeneration {
        let resolvedURL = modelDirectory.resolvingSymlinksInPath()
        let model = GLMOCRForConditionalGeneration(config: config)

        let safetensorFiles = try enumerateSafetensors(modelDirectory: resolvedURL)
        guard !safetensorFiles.isEmpty else {
            throw GLMOCRModelLoaderError.modelLoadFailed("No .safetensors files found under: \(resolvedURL.lastPathComponent)")
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if isGitLFSPointer(file) {
                throw GLMOCRModelLoaderError.modelLoadFailed(
                    """
                    Found a Git LFS pointer instead of real weights: \(file.lastPathComponent).
                    Download weights via Hugging Face snapshot (recommended) or run `git lfs pull`.
                    """
                )
            }
            let arrays = try MLX.loadArrays(url: file)
            for (key, value) in arrays {
                weights[key] = value
            }
        }

        weights = GLMOCRWeightsLoader.sanitizeTorchConvWeights(weights)
        weights = filterUnsupportedLayers(weights, config: config)
        if let dtype {
            weights = castFloatingWeights(weights, to: dtype)
        }

        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.allModelKeysSet, .shapeMismatch])
        try checkedEval(model)
        return model
    }

    private static func enumerateSafetensors(modelDirectory: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else {
            throw GLMOCRModelLoaderError.modelLoadFailed("Failed to enumerate model directory: \(modelDirectory.lastPathComponent)")
        }

        var safetensorFiles: [URL] = []
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors" else { continue }
            safetensorFiles.append(url)
        }
        return safetensorFiles
    }

    private static func isGitLFSPointer(_ url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url) else { return false }
        guard data.count < 512 else { return false }
        guard let text = String(data: data, encoding: .utf8) else { return false }
        return text.hasPrefix("version https://git-lfs.github.com/spec/v1")
    }

    private static func castFloatingWeights(_ weights: [String: MLXArray], to dtype: DType) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            if value.dtype.isFloatingPoint {
                out[key] = value.asType(dtype)
            } else {
                out[key] = value
            }
        }
        return out
    }

    private static func filterUnsupportedLayers(_ weights: [String: MLXArray], config: GLMOCRModelConfig) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        let maxTextLayers = config.textConfig.numHiddenLayers
        let maxVisionBlocks = config.visionConfig.depth

        for (key, value) in weights {
            if key.hasPrefix("model.language_model.layers.") {
                let suffix = key.dropFirst("model.language_model.layers.".count)
                if let idxStr = suffix.split(separator: ".").first, let idx = Int(idxStr), idx >= maxTextLayers {
                    continue
                }
            }
            if key.hasPrefix("model.visual.blocks.") {
                let suffix = key.dropFirst("model.visual.blocks.".count)
                if let idxStr = suffix.split(separator: ".").first, let idx = Int(idxStr), idx >= maxVisionBlocks {
                    continue
                }
            }
            out[key] = value
        }
        return out
    }
}

enum GLMOCRModelLoaderError: Error, Sendable {
    case modelLoadFailed(String)
}
