import Foundation
import MLX

enum GLMOCRWeightsLoader {
    static func loadSafetensors(fromSafetensorsFile fileURL: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fileURL)
    }

    static func loadSafetensors(from modelDirectory: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        guard let enumerator = FileManager.default.enumerator(at: modelDirectory, includingPropertiesForKeys: nil) else {
            throw GLMOCRWeightsLoaderError.unableToEnumerate(modelDirectory.lastPathComponent)
        }
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors" else { continue }
            let arrays = try MLX.loadArrays(url: url)
            for (key, value) in arrays {
                weights[key] = value
            }
        }
        return weights
    }

    private static func looksLikeMLXConvWeights(_ value: MLXArray) -> Bool {
        let shape = value.shape
        if shape.count == 4 {
            
            let out = shape[0]
            let kH = shape[1]
            let kW = shape[2]
            return (out >= kH) && (out >= kW) && (kH == kW)
        }
        if shape.count == 5 {
            
            let out = shape[0]
            let kH = shape[2]
            let kW = shape[3]
            return (out >= kH) && (out >= kW) && (kH == kW)
        }
        return false
    }

    static func sanitizeTorchConvWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }
            if key == "model.visual.patch_embed.proj.weight", value.ndim == 5 {
                if looksLikeMLXConvWeights(value) {
                    out[key] = value
                } else {
                    
                    
                    out[key] = value.transposed(0, 2, 3, 4, 1)
                }
            } else if key == "model.visual.downsample.weight", value.ndim == 4 {
                if looksLikeMLXConvWeights(value) {
                    out[key] = value
                } else {
                    
                    
                    out[key] = value.transposed(0, 2, 3, 1)
                }
            } else {
                out[key] = value
            }
        }

        return out
    }
}

enum GLMOCRWeightsLoaderError: Error, Sendable {
    case unableToEnumerate(String)
}
