import Foundation
import MLX

final class GLMOCRKVCache {
    let kvHeads: Int
    let headDim: Int
    let maxLength: Int
    private let step: Int

    private(set) var offset: Int = 0
    private var keyCache: MLXArray
    private var valueCache: MLXArray

    init(
        batch: Int,
        kvHeads: Int,
        headDim: Int,
        maxLength: Int,
        dtype: MLX.DType,
        step: Int = 1024,
        initialCapacity: Int? = nil
    ) {
        precondition(batch > 0)
        precondition(kvHeads > 0)
        precondition(headDim > 0)
        precondition(maxLength > 0)
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxLength = maxLength
        self.step = max(1, step)

        let requestedCapacity = max(1, initialCapacity ?? self.step)
        let roundedCapacity = ((requestedCapacity + self.step - 1) / self.step) * self.step
        let effectiveCapacity = min(maxLength, roundedCapacity)
        self.keyCache = MLXArray.zeros([batch, kvHeads, effectiveCapacity, headDim], dtype: dtype)
        self.valueCache = MLXArray.zeros([batch, kvHeads, effectiveCapacity, headDim], dtype: dtype)
    }

    private func ensureCapacity(required: Int) {
        let currentCapacity = keyCache.dim(2)
        guard required > currentCapacity else { return }

        let newCapacity = min(maxLength, ((required + step - 1) / step) * step)
        precondition(newCapacity > currentCapacity, "KV cache capacity did not grow")

        let batch = keyCache.dim(0)
        let extra = newCapacity - currentCapacity

        let extraKeys = MLXArray.zeros([batch, kvHeads, extra, headDim], dtype: keyCache.dtype)
        let extraValues = MLXArray.zeros([batch, kvHeads, extra, headDim], dtype: valueCache.dtype)

        keyCache = concatenated([keyCache, extraKeys], axis: 2)
        valueCache = concatenated([valueCache, extraValues], axis: 2)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys.dim(1) == kvHeads)
        precondition(keys.dim(3) == headDim)
        precondition(values.dim(1) == kvHeads)
        precondition(values.dim(3) == headDim)

        let length = keys.dim(2)
        let required = offset + length
        precondition(required <= maxLength, "KV cache overflow (offset=\(offset) length=\(length) max=\(maxLength))")
        ensureCapacity(required: required)

        keyCache[0..., 0..., offset ..< required, 0...] = keys
        valueCache[0..., 0..., offset ..< required, 0...] = values
        offset = required

        let cachedKeys = keyCache[0..., 0..., 0 ..< offset, 0...]
        let cachedValues = valueCache[0..., 0..., 0 ..< offset, 0...]
        return (cachedKeys, cachedValues)
    }

    func reset() {
        offset = 0
    }
}
