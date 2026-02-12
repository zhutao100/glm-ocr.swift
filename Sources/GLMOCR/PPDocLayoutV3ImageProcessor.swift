import CoreGraphics
import Foundation
import ImageIO
import MLX

public struct PPDocLayoutV3PreprocessedImage {
    public let pixelValues: MLXArray
    public let originalWidth: Int
    public let originalHeight: Int
}

public struct PPDocLayoutV3ImageProcessor: Sendable {
    public var size: (height: Int, width: Int) = (800, 800)

    public init() {}

    private final class SharedResizeState: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var resizedRGBs: [Data?]
        private(set) var originalSizes: [(width: Int, height: Int)?]
        private var firstError: Error?

        init(count: Int) {
            resizedRGBs = Array(repeating: nil, count: count)
            originalSizes = Array(repeating: nil, count: count)
        }

        func shouldSkipWork() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return firstError != nil
        }

        func recordSuccess(index: Int, rgb: Data, size: (width: Int, height: Int)) {
            lock.lock()
            resizedRGBs[index] = rgb
            originalSizes[index] = size
            lock.unlock()
        }

        func recordFailure(_ error: Error) {
            lock.lock()
            if firstError == nil {
                firstError = error
            }
            lock.unlock()
        }

        func throwIfFailed() throws {
            lock.lock()
            let error = firstError
            lock.unlock()
            if let error {
                throw error
            }
        }
    }

    private func decodeAndResizeRGB(image: CGImage) throws -> (resizedRGB: Data, originalWidth: Int, originalHeight: Int) {
        let originalWidth = image.width
        let originalHeight = image.height
        guard originalWidth > 0, originalHeight > 0 else {
            throw PPDocLayoutV3ImageProcessorError.invalidImageExtent(width: originalWidth, height: originalHeight)
        }

        let decoded = try decodeRGBA(image: image)
        let resized = try resizeBicubicRGB(
            rgba: decoded.data,
            srcWidth: decoded.width,
            srcHeight: decoded.height,
            dstWidth: size.width,
            dstHeight: size.height
        )

        return (resizedRGB: resized, originalWidth: originalWidth, originalHeight: originalHeight)
    }

    public func process(_ image: CGImage) throws -> PPDocLayoutV3PreprocessedImage {
        let resized = try decodeAndResizeRGB(image: image)

        var array = MLXArray(resized.resizedRGB, [size.height, size.width, 3], type: UInt8.self).asType(.float32)
        let inv255: Float = 1.0 / 255.0
        array = array * inv255
        array = array.expandedDimensions(axis: 0)

        return PPDocLayoutV3PreprocessedImage(
            pixelValues: array,
            originalWidth: resized.originalWidth,
            originalHeight: resized.originalHeight
        )
    }

    public func process(_ images: [CGImage]) throws -> (pixelValues: MLXArray, originalSizes: [(width: Int, height: Int)]) {
        precondition(!images.isEmpty, "images must be non-empty")

        if images.count == 1 {
            let processed = try process(images[0])
            return (pixelValues: processed.pixelValues, originalSizes: [(width: processed.originalWidth, height: processed.originalHeight)])
        }

        let state = SharedResizeState(count: images.count)
        DispatchQueue.concurrentPerform(iterations: images.count) { index in
            if state.shouldSkipWork() {
                return
            }

            do {
                let resized = try decodeAndResizeRGB(image: images[index])
                state.recordSuccess(
                    index: index,
                    rgb: resized.resizedRGB,
                    size: (width: resized.originalWidth, height: resized.originalHeight)
                )
            } catch {
                state.recordFailure(error)
            }
        }
        try state.throwIfFailed()

        var pixelValuesList: [MLXArray] = []
        pixelValuesList.reserveCapacity(images.count)

        var outSizes: [(width: Int, height: Int)] = []
        outSizes.reserveCapacity(images.count)

        for index in images.indices {
            guard let rgb = state.resizedRGBs[index] else {
                throw PPDocLayoutV3ImageProcessorError.resizeFailed
            }
            guard let size = state.originalSizes[index] else {
                throw PPDocLayoutV3ImageProcessorError.resizeFailed
            }

            var array = MLXArray(rgb, [self.size.height, self.size.width, 3], type: UInt8.self).asType(.float32)
            let inv255: Float = 1.0 / 255.0
            array = array * inv255
            array = array.expandedDimensions(axis: 0)

            pixelValuesList.append(array)
            outSizes.append(size)
        }

        let pixelValuesBatch = pixelValuesList.count == 1 ? pixelValuesList[0] : concatenated(pixelValuesList, axis: 0)
        return (pixelValues: pixelValuesBatch, originalSizes: outSizes)
    }

    public func process(imageAt path: String) throws -> PPDocLayoutV3PreprocessedImage {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw PPDocLayoutV3ImageProcessorError.imageLoadFailed(url.lastPathComponent.isEmpty ? "image" : url.lastPathComponent)
        }
        return try process(image)
    }

    private struct DecodedRGBA {
        let data: Data
        let width: Int
        let height: Int
    }

    private struct ResizeCoeff {
        let i0: Int
        let i1: Int
        let i2: Int
        let i3: Int
        let w0: Double
        let w1: Double
        let w2: Double
        let w3: Double
    }

    private func decodeRGBA(image: CGImage) throws -> DecodedRGBA {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        var data = Data(count: height * bytesPerRow)
        try data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else {
                throw PPDocLayoutV3ImageProcessorError.bitmapAllocationFailed
            }
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
                throw PPDocLayoutV3ImageProcessorError.colorSpaceFailed
            }

            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                throw PPDocLayoutV3ImageProcessorError.contextCreateFailed
            }

            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        return DecodedRGBA(data: data, width: width, height: height)
    }

    private func cubicKernel(_ x: Double) -> Double {
        
        let a = -0.75
        let ax = abs(x)
        if ax <= 1 {
            return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1
        } else if ax < 2 {
            return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a
        } else {
            return 0
        }
    }

    private func buildResizeCoeffs(src: Int, dst: Int) -> [ResizeCoeff] {
        let scale = Double(src) / Double(dst)
        var coeffs: [ResizeCoeff] = []
        coeffs.reserveCapacity(dst)

        for iOut in 0..<dst {
            let x = (Double(iOut) + 0.5) * scale - 0.5
            let ix = Int(floor(x))

            let i0 = max(0, min(src - 1, ix - 1))
            let i1 = max(0, min(src - 1, ix))
            let i2 = max(0, min(src - 1, ix + 1))
            let i3 = max(0, min(src - 1, ix + 2))

            let w0 = cubicKernel(x - Double(ix - 1))
            let w1 = cubicKernel(x - Double(ix))
            let w2 = cubicKernel(x - Double(ix + 1))
            let w3 = cubicKernel(x - Double(ix + 2))

            let sum = w0 + w1 + w2 + w3
            let inv = sum != 0 ? (1.0 / sum) : 1.0
            coeffs.append(
                ResizeCoeff(
                    i0: i0, i1: i1, i2: i2, i3: i3,
                    w0: w0 * inv, w1: w1 * inv, w2: w2 * inv, w3: w3 * inv
                )
            )
        }

        return coeffs
    }

    private func resizeBicubicRGB(
        rgba: Data,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int
    ) throws -> Data {
        guard srcWidth > 0, srcHeight > 0, dstWidth > 0, dstHeight > 0 else {
            throw PPDocLayoutV3ImageProcessorError.resizeFailed
        }
        guard rgba.count == srcWidth * srcHeight * 4 else {
            throw PPDocLayoutV3ImageProcessorError.resizeFailed
        }

        if srcWidth == dstWidth && srcHeight == dstHeight {
            var out = Data(count: dstWidth * dstHeight * 3)
            out.withUnsafeMutableBytes { outPtr in
                rgba.withUnsafeBytes { srcPtr in
                    guard let outBase = outPtr.bindMemory(to: UInt8.self).baseAddress,
                          let srcBase = srcPtr.bindMemory(to: UInt8.self).baseAddress
                    else { return }

                    var dstIndex = 0
                    var srcIndex = 0
                    for _ in 0..<(srcWidth * srcHeight) {
                        outBase[dstIndex] = srcBase[srcIndex]
                        outBase[dstIndex + 1] = srcBase[srcIndex + 1]
                        outBase[dstIndex + 2] = srcBase[srcIndex + 2]
                        dstIndex += 3
                        srcIndex += 4
                    }
                }
            }
            return out
        }

        let xCoeffs = buildResizeCoeffs(src: srcWidth, dst: dstWidth)
        let yCoeffs = buildResizeCoeffs(src: srcHeight, dst: dstHeight)

        var out = Data(count: dstWidth * dstHeight * 3)

        try out.withUnsafeMutableBytes { outPtr in
            try rgba.withUnsafeBytes { srcPtr in
                guard let outBase = outPtr.bindMemory(to: UInt8.self).baseAddress,
                      let srcBase = srcPtr.bindMemory(to: UInt8.self).baseAddress
                else {
                    throw PPDocLayoutV3ImageProcessorError.resizeFailed
                }

                func readRGB(y: Int, x: Int) -> (Double, Double, Double) {
                    let idx = (y * srcWidth + x) * 4
                    return (
                        Double(srcBase[idx]),
                        Double(srcBase[idx + 1]),
                        Double(srcBase[idx + 2])
                    )
                }

                let rowElementCount = dstWidth * 3

                func horizontalResampleRow(_ srcY: Int, into dstRow: inout [Double]) {
                    var outIndex = 0
                    for xOut in 0..<dstWidth {
                        let c = xCoeffs[xOut]
                        let (r0, g0, b0) = readRGB(y: srcY, x: c.i0)
                        let (r1, g1, b1) = readRGB(y: srcY, x: c.i1)
                        let (r2, g2, b2) = readRGB(y: srcY, x: c.i2)
                        let (r3, g3, b3) = readRGB(y: srcY, x: c.i3)

                        dstRow[outIndex] = c.w0 * r0 + c.w1 * r1 + c.w2 * r2 + c.w3 * r3
                        dstRow[outIndex + 1] = c.w0 * g0 + c.w1 * g1 + c.w2 * g2 + c.w3 * g3
                        dstRow[outIndex + 2] = c.w0 * b0 + c.w1 * b1 + c.w2 * b2 + c.w3 * b3
                        outIndex += 3
                    }
                }

                let rowCacheSize = 8
                var rowCacheSourceY = Array(repeating: Int.min, count: rowCacheSize)
                var rowCacheRows = Array(repeating: [Double](repeating: 0, count: rowElementCount), count: rowCacheSize)
                var rowCacheNextSlot = 0

                @inline(__always)
                func cachedHorizontalRowSlot(_ srcY: Int) -> Int {
                    for idx in 0..<rowCacheSize where rowCacheSourceY[idx] == srcY {
                        return idx
                    }

                    let slot = rowCacheNextSlot
                    rowCacheNextSlot = (rowCacheNextSlot + 1) % rowCacheSize
                    horizontalResampleRow(srcY, into: &rowCacheRows[slot])
                    rowCacheSourceY[slot] = srcY
                    return slot
                }

                func clampToByte(_ value: Double) -> UInt8 {
                    if value <= 0 { return 0 }
                    if value >= 255 { return 255 }
                    return UInt8(Int(value + 0.5))
                }

                for yOut in 0..<dstHeight {
                    let c = yCoeffs[yOut]
                    let row0Slot = cachedHorizontalRowSlot(c.i0)
                    let row1Slot = cachedHorizontalRowSlot(c.i1)
                    let row2Slot = cachedHorizontalRowSlot(c.i2)
                    let row3Slot = cachedHorizontalRowSlot(c.i3)

                    let rowOffset = yOut * rowElementCount
                    for i in 0..<rowElementCount {
                        let v =
                            c.w0 * rowCacheRows[row0Slot][i]
                            + c.w1 * rowCacheRows[row1Slot][i]
                            + c.w2 * rowCacheRows[row2Slot][i]
                            + c.w3 * rowCacheRows[row3Slot][i]
                        outBase[rowOffset + i] = clampToByte(v)
                    }
                }
            }
        }

        return out
    }
}

public enum PPDocLayoutV3ImageProcessorError: Error, Sendable {
    case imageLoadFailed(String)
    case invalidImageExtent(width: Int, height: Int)
    case bitmapAllocationFailed
    case colorSpaceFailed
    case contextCreateFailed
    case resizeFailed
}
