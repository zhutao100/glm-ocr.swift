import Foundation
import MLX
import MLXNN

enum PPDocLayoutV3ModelLoader {
    static func load(from modelDirectory: URL, dtype: DType? = nil) throws -> (model: PPDocLayoutV3ForObjectDetection, config: PPDocLayoutV3Config) {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(PPDocLayoutV3Config.self, from: data)

        let model = PPDocLayoutV3ForObjectDetection(config: config)

        let safetensorFiles = try enumerateSafetensors(modelDirectory: modelDirectory)
        guard !safetensorFiles.isEmpty else {
            throw PPDocLayoutV3ModelLoaderError.modelLoadFailed("No .safetensors files found under: \(modelDirectory.lastPathComponent)")
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if isGitLFSPointer(file) {
                throw PPDocLayoutV3ModelLoaderError.modelLoadFailed(
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

        weights = rewriteAIFIKeys(weights)
        weights = rewriteMLPKeys(weights)
        weights = rewriteConvBNKeys(weights)
        weights = transposeTorchConvWeights(weights)
        if let dtype {
            weights = castFloatingWeights(weights, to: dtype)
        }

        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.allModelKeysSet, .shapeMismatch])
        model.train(false)
        try checkedEval(model)
        return (model, config)
    }

    private static func enumerateSafetensors(modelDirectory: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else {
            throw PPDocLayoutV3ModelLoaderError.modelLoadFailed("Failed to enumerate model directory: \(modelDirectory.lastPathComponent)")
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

    private static func transposeTorchConvWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            
            if value.ndim == 4, key.hasSuffix(".weight") {
                out[key] = value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }

        return out
    }

    private static func rewriteConvBNKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        
        
        
        
        func rewrite(_ key: String) -> String {
            let parts = key.split(separator: ".", omittingEmptySubsequences: false)
            
            guard parts.count >= 5 else { return key }
            guard parts[0] == "model" else { return key }
            guard parts[1] == "encoder_input_proj" || parts[1] == "decoder_input_proj" else { return key }
            guard parts[3] == "0" || parts[3] == "1" else { return key }
            var newParts = parts.map(String.init)
            newParts[3] = (parts[3] == "0") ? "conv" : "bn"
            return newParts.joined(separator: ".")
        }

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            out[rewrite(key)] = value
        }
        return out
    }

    private static func rewriteAIFIKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let oldPrefix = "model.encoder.encoder."
        let newPrefix = "model.encoder.aifi."

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.hasPrefix(oldPrefix) {
                let suffix = key.dropFirst(oldPrefix.count)
                out[newPrefix + suffix] = value
            } else {
                out[key] = value
            }
        }

        return out
    }

    private static func rewriteMLPKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        func isInt(_ s: Substring) -> Bool { Int(s) != nil }

        func rewrite(_ key: String) -> String {
            let parts = key.split(separator: ".", omittingEmptySubsequences: false)

            
            if parts.count >= 6,
               parts[0] == "model",
               parts[1] == "decoder",
               parts[2] == "layers",
               isInt(parts[3]),
               (parts[4] == "fc1" || parts[4] == "fc2")
            {
                var newParts = parts.map(String.init)
                newParts.insert("mlp", at: 4)
                return newParts.joined(separator: ".")
            }

            
            if parts.count >= 8,
               parts[0] == "model",
               parts[1] == "encoder",
               parts[2] == "aifi",
               isInt(parts[3]),
               parts[4] == "layers",
               isInt(parts[5]),
               (parts[6] == "fc1" || parts[6] == "fc2")
            {
                var newParts = parts.map(String.init)
                newParts.insert("mlp", at: 6)
                return newParts.joined(separator: ".")
            }

            return key
        }

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            out[rewrite(key)] = value
        }

        return out
    }
}

enum PPDocLayoutV3ModelLoaderError: Error, Sendable {
    case modelLoadFailed(String)
}
