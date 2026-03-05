import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public struct GLMOCRPreprocessedImage {
    public let pixelValues: MLXArray
    public let imageGridTHW: (t: Int, h: Int, w: Int)
    public let resizedHeight: Int
    public let resizedWidth: Int
}

public struct GLMOCRImageProcessor {
    private let config: GLMOCRPreprocessorConfig
    private let normalizationMean: MLXArray?
    private let normalizationStd: MLXArray?

    public init(config: GLMOCRPreprocessorConfig) {
        self.config = config
        if config.imageMean.count == 3, config.imageStd.count == 3 {
            self.normalizationMean = MLXArray(config.imageMean.map(Float.init))
            self.normalizationStd = MLXArray(config.imageStd.map(Float.init))
        } else {
            self.normalizationMean = nil
            self.normalizationStd = nil
        }
    }

    public func process(imageAt url: URL, postResizeJPEGRoundTripQuality: Double? = nil) throws
        -> GLMOCRPreprocessedImage
    {
        let image = try Self.loadCGImage(from: url)
        return try process(image, postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality)
    }

    public func process(imageAt path: String) throws -> GLMOCRPreprocessedImage {
        try process(imageAt: URL(fileURLWithPath: path))
    }

    public func process(_ image: CIImage) throws -> GLMOCRPreprocessedImage {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else {
            throw GLMOCRImageProcessorError.invalidImageExtent(width: extent.width, height: extent.height)
        }
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(image, from: extent) else {
            throw GLMOCRImageProcessorError.imageLoadFailed("CIImage->CGImage conversion failed")
        }
        return try process(cgImage)
    }

    public func process(
        imageURLs: [URL],
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> [GLMOCRPreprocessedImage] {
        if imageURLs.isEmpty {
            return []
        }
        let images = try Self.loadCGImagesConcurrently(imageURLs)
        return try process(images, postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality)
    }

    public func process(
        _ image: CGImage,
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> GLMOCRPreprocessedImage {
        let decoded = try Self.decodeRGBA(image: image)
        return try process(decodedRGBA: decoded, postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality)
    }

    public func process(
        _ images: [CGImage],
        postResizeJPEGRoundTripQuality: Double? = nil
    ) throws -> [GLMOCRPreprocessedImage] {
        if images.isEmpty {
            return []
        }

        let stage1 = try preprocessBatchCPUStage(
            images,
            postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
        )

        var output: [GLMOCRPreprocessedImage] = []
        output.reserveCapacity(stage1.count)

        for resizedRGB in stage1 {
            output.append(try process(resizedRGB: resizedRGB))
        }

        return output
    }

    private func process(
        decodedRGBA: DecodedRGBA,
        postResizeJPEGRoundTripQuality: Double?
    ) throws -> GLMOCRPreprocessedImage {
        let resizedRGB = try Self.prepareResizedRGB(
            decodedRGBA: decodedRGBA,
            temporalPatchSize: config.temporalPatchSize,
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.shortestEdge,
            maxPixels: config.size.longestEdge,
            postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
        )
        return try process(resizedRGB: resizedRGB)
    }

    private func process(resizedRGB: ResizedRGB) throws -> GLMOCRPreprocessedImage {
        let frame = try normalizeAndConvertToChannelsFirst(
            resizedRGB: resizedRGB.rgb,
            resizedHeight: resizedRGB.height,
            resizedWidth: resizedRGB.width
        )
        return patchify(frame: frame, resizedHeight: resizedRGB.height, resizedWidth: resizedRGB.width)
    }

    private func patchify(frame: MLXArray, resizedHeight: Int, resizedWidth: Int) -> GLMOCRPreprocessedImage {
        let patchSize = config.patchSize
        let temporalPatchSize = config.temporalPatchSize
        let mergeSize = config.mergeSize

        var patches = stacked([frame], axis: 0)
        let remainder = patches.shape[0] % temporalPatchSize
        if remainder != 0 {
            let extra = temporalPatchSize - remainder
            let repeats = stacked(Array(repeating: frame, count: extra), axis: 0)
            patches = concatenated([patches, repeats], axis: 0)
        }

        let channel = patches.shape[1]
        let gridT = patches.shape[0] / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT,
            temporalPatchSize,
            channel,
            gridH / mergeSize,
            mergeSize,
            patchSize,
            gridW / mergeSize,
            mergeSize,
            patchSize
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let patchDim = channel * temporalPatchSize * patchSize * patchSize
        let flattenPatches = patches.reshaped(gridT * gridH * gridW, patchDim)

        return GLMOCRPreprocessedImage(
            pixelValues: flattenPatches,
            imageGridTHW: (t: gridT, h: gridH, w: gridW),
            resizedHeight: resizedHeight,
            resizedWidth: resizedWidth
        )
    }

    private func preprocessBatchCPUStage(
        _ images: [CGImage],
        postResizeJPEGRoundTripQuality: Double?
    ) throws -> [ResizedRGB] {
        let imageCount = images.count
        if imageCount == 0 {
            return []
        }

        let temporalPatchSize = config.temporalPatchSize
        let factor = config.patchSize * config.mergeSize
        let minPixels = config.size.shortestEdge
        let maxPixels = config.size.longestEdge
        let chunkSize = Self.batchChunkSize(imageCount: imageCount)
        let chunkCount = (imageCount + chunkSize - 1) / chunkSize

        var stage1Results = [BatchStage1Result?](repeating: nil, count: imageCount)
        stage1Results.withUnsafeMutableBufferPointer { results in
            guard let resultsBase = results.baseAddress else {
                return
            }
            let resultsBaseAddress = Int(bitPattern: resultsBase)

            images.withUnsafeBufferPointer { imageBuffer in
                guard let imageBase = imageBuffer.baseAddress else {
                    return
                }
                let imageBaseAddress = Int(bitPattern: imageBase)

                DispatchQueue.concurrentPerform(iterations: chunkCount) { chunkIndex in
                    guard let resultsBase = UnsafeMutablePointer<BatchStage1Result?>(bitPattern: resultsBaseAddress),
                        let imageBase = UnsafePointer<CGImage>(bitPattern: imageBaseAddress)
                    else {
                        return
                    }

                    let start = chunkIndex * chunkSize
                    let end = min(start + chunkSize, imageCount)
                    guard start < end else {
                        return
                    }

                    for index in start..<end {
                        do {
                            let decoded = try Self.decodeRGBA(image: imageBase[index])
                            let resizedRGB = try Self.prepareResizedRGB(
                                decodedRGBA: decoded,
                                temporalPatchSize: temporalPatchSize,
                                factor: factor,
                                minPixels: minPixels,
                                maxPixels: maxPixels,
                                postResizeJPEGRoundTripQuality: postResizeJPEGRoundTripQuality
                            )
                            resultsBase[index] = .success(resizedRGB)
                        } catch let processorError as GLMOCRImageProcessorError {
                            resultsBase[index] = .failure(processorError)
                        } catch {
                            resultsBase[index] = .failure(.resizeFailed)
                        }
                    }
                }
            }
        }

        var output: [ResizedRGB] = []
        output.reserveCapacity(imageCount)
        for index in 0..<imageCount {
            guard let result = stage1Results[index] else {
                throw GLMOCRImageProcessorError.resizeFailed
            }
            switch result {
            case .success(let resized):
                output.append(resized)
            case .failure(let error):
                throw error
            }
        }

        return output
    }

    private static func loadCGImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw GLMOCRImageProcessorError.imageLoadFailed(
                url.lastPathComponent.isEmpty ? "image" : url.lastPathComponent)
        }
        return image
    }

    private static func loadCGImagesConcurrently(_ imageURLs: [URL]) throws -> [CGImage] {
        let imageCount = imageURLs.count
        if imageCount == 0 {
            return []
        }

        let chunkSize = batchChunkSize(imageCount: imageCount)
        let chunkCount = (imageCount + chunkSize - 1) / chunkSize

        var loadResults = [BatchImageLoadResult?](repeating: nil, count: imageCount)
        loadResults.withUnsafeMutableBufferPointer { results in
            guard let resultsBase = results.baseAddress else {
                return
            }
            let resultsBaseAddress = Int(bitPattern: resultsBase)

            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunkIndex in
                guard let resultsBase = UnsafeMutablePointer<BatchImageLoadResult?>(bitPattern: resultsBaseAddress)
                else {
                    return
                }

                let start = chunkIndex * chunkSize
                let end = min(start + chunkSize, imageCount)
                guard start < end else {
                    return
                }

                for index in start..<end {
                    do {
                        let image = try loadCGImage(from: imageURLs[index])
                        resultsBase[index] = .success(image)
                    } catch let processorError as GLMOCRImageProcessorError {
                        resultsBase[index] = .failure(processorError)
                    } catch {
                        let fileName = imageURLs[index].lastPathComponent
                        resultsBase[index] = .failure(.imageLoadFailed(fileName.isEmpty ? "image" : fileName))
                    }
                }
            }
        }

        var output: [CGImage] = []
        output.reserveCapacity(imageCount)
        for index in 0..<imageCount {
            guard let result = loadResults[index] else {
                throw GLMOCRImageProcessorError.imageLoadFailed("image")
            }
            switch result {
            case .success(let image):
                output.append(image)
            case .failure(let error):
                throw error
            }
        }

        return output
    }

    private func normalizeAndConvertToChannelsFirst(
        resizedRGB: Data,
        resizedHeight: Int,
        resizedWidth: Int
    ) throws -> MLXArray {
        var array = MLXArray(resizedRGB, [resizedHeight, resizedWidth, 3], type: UInt8.self).asType(.float32)
        if config.doRescale {
            let inv255: Float = 1.0 / 255.0
            array = array * inv255
        }
        let normalized = try normalize(array)
        return normalized.transposed(2, 0, 1)
    }

    private static func prepareResizedRGB(
        decodedRGBA: DecodedRGBA,
        temporalPatchSize: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
        postResizeJPEGRoundTripQuality: Double?
    ) throws -> ResizedRGB {
        let resized = try smartResize(
            numFrames: temporalPatchSize,
            height: decodedRGBA.height,
            width: decodedRGBA.width,
            temporalFactor: temporalPatchSize,
            factor: factor,
            minPixels: minPixels,
            maxPixels: maxPixels
        )

        let resizedRGB = try resizeBicubicRGB(
            rgba: decodedRGBA.data,
            srcWidth: decodedRGBA.width,
            srcHeight: decodedRGBA.height,
            dstWidth: resized.width,
            dstHeight: resized.height
        )

        if let quality = postResizeJPEGRoundTripQuality {
            let jpegRoundTripped = try jpegRoundTripRGB(
                resizedRGB,
                width: resized.width,
                height: resized.height,
                quality: quality
            )
            return ResizedRGB(rgb: jpegRoundTripped, height: resized.height, width: resized.width)
        }

        return ResizedRGB(rgb: resizedRGB, height: resized.height, width: resized.width)
    }

    private static func jpegRoundTripRGB(_ rgb: Data, width: Int, height: Int, quality: Double) throws -> Data {
        guard width > 0, height > 0 else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        guard rgb.count == width * height * 3 else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }

        let rgba = try rgbToRGBA(rgb, width: width, height: height)
        let image = try makeRGBAImage(width: width, height: height, rgba: rgba)
        let jpegData = try encodeJPEGData(image, quality: quality)
        guard let source = CGImageSourceCreateWithData(jpegData as CFData, nil),
            let decodedImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }

        let decoded = try decodeRGBA(image: decodedImage)
        guard decoded.width == width, decoded.height == height else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        return try rgbaToRGB(decoded.data, width: width, height: height)
    }

    private static func rgbToRGBA(_ rgb: Data, width: Int, height: Int) throws -> Data {
        guard rgb.count == width * height * 3 else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        let pixelCount = width * height
        var rgba = Data(count: pixelCount * 4)
        try rgba.withUnsafeMutableBytes { dstPtr in
            try rgb.withUnsafeBytes { srcPtr in
                guard let dstBase = dstPtr.bindMemory(to: UInt8.self).baseAddress,
                    let srcBase = srcPtr.bindMemory(to: UInt8.self).baseAddress
                else { throw GLMOCRImageProcessorError.jpegRoundTripFailed }
                var srcIndex = 0
                var dstIndex = 0
                for _ in 0..<pixelCount {
                    dstBase[dstIndex] = srcBase[srcIndex]
                    dstBase[dstIndex + 1] = srcBase[srcIndex + 1]
                    dstBase[dstIndex + 2] = srcBase[srcIndex + 2]
                    dstBase[dstIndex + 3] = 255
                    srcIndex += 3
                    dstIndex += 4
                }
            }
        }
        return rgba
    }

    private static func rgbaToRGB(_ rgba: Data, width: Int, height: Int) throws -> Data {
        guard rgba.count == width * height * 4 else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        let pixelCount = width * height
        var rgb = Data(count: pixelCount * 3)
        try rgb.withUnsafeMutableBytes { dstPtr in
            try rgba.withUnsafeBytes { srcPtr in
                guard let dstBase = dstPtr.bindMemory(to: UInt8.self).baseAddress,
                    let srcBase = srcPtr.bindMemory(to: UInt8.self).baseAddress
                else { throw GLMOCRImageProcessorError.jpegRoundTripFailed }
                var srcIndex = 0
                var dstIndex = 0
                for _ in 0..<pixelCount {
                    dstBase[dstIndex] = srcBase[srcIndex]
                    dstBase[dstIndex + 1] = srcBase[srcIndex + 1]
                    dstBase[dstIndex + 2] = srcBase[srcIndex + 2]
                    srcIndex += 4
                    dstIndex += 3
                }
            }
        }
        return rgb
    }

    private static func makeRGBAImage(width: Int, height: Int, rgba: Data) throws -> CGImage {
        guard width > 0, height > 0 else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        let bytesPerRow = width * 4
        guard rgba.count == height * bytesPerRow else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        guard let provider = CGDataProvider(data: rgba as CFData),
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard
            let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        return image
    }

    private static func encodeJPEGData(_ image: CGImage, quality: Double) throws -> Data {
        let clampedQuality = max(0.0, min(quality, 1.0))
        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data,
                UTType.jpeg.identifier as CFString,
                1,
                nil
            )
        else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        let options = [kCGImageDestinationLossyCompressionQuality: clampedQuality] as CFDictionary
        CGImageDestinationAddImage(destination, image, options)
        guard CGImageDestinationFinalize(destination) else {
            throw GLMOCRImageProcessorError.jpegRoundTripFailed
        }
        return data as Data
    }

    private func normalize(_ array: MLXArray) throws -> MLXArray {
        guard let mean = normalizationMean, let std = normalizationStd else {
            throw GLMOCRImageProcessorError.invalidNormalizationConfig(
                meanCount: config.imageMean.count,
                stdCount: config.imageStd.count
            )
        }
        return (array - mean) / std
    }

    private static func batchChunkSize(imageCount: Int) -> Int {
        let cpuCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let targetChunks = max(1, cpuCount * 4)
        return max(1, min(32, (imageCount + targetChunks - 1) / targetChunks))
    }

    private struct ResizedRGB {
        let rgb: Data
        let height: Int
        let width: Int
    }

    private enum BatchStage1Result {
        case success(ResizedRGB)
        case failure(GLMOCRImageProcessorError)
    }

    private enum BatchImageLoadResult {
        case success(CGImage)
        case failure(GLMOCRImageProcessorError)
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

    private static func decodeRGBA(image: CGImage) throws -> DecodedRGBA {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        var data = Data(count: height * bytesPerRow)
        try data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else {
                throw GLMOCRImageProcessorError.bitmapAllocationFailed
            }
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
                throw GLMOCRImageProcessorError.colorSpaceFailed
            }

            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            guard
                let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo
                )
            else {
                throw GLMOCRImageProcessorError.bitmapAllocationFailed
            }

            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        return DecodedRGBA(data: data, width: width, height: height)
    }

    private static func cubicKernel(_ x: Double) -> Double {
        let a = -0.5
        let ax = abs(x)
        let ax2 = ax * ax
        let ax3 = ax2 * ax
        if ax <= 1 {
            return (a + 2) * ax3 - (a + 3) * ax2 + 1
        }
        if ax < 2 {
            return a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a
        }
        return 0
    }

    private static func buildResizeCoeffs(src: Int, dst: Int) -> [ResizeCoeff] {
        precondition(src > 0 && dst > 0)

        let scale = Double(src) / Double(dst)
        var coeffs: [ResizeCoeff] = []
        coeffs.reserveCapacity(dst)

        func clampIndex(_ i: Int) -> Int {
            min(max(i, 0), src - 1)
        }

        for outIndex in 0..<dst {
            let inCoord = (Double(outIndex) + 0.5) * scale - 0.5
            let base = Int(floor(inCoord))
            let frac = inCoord - Double(base)

            let i0 = clampIndex(base - 1)
            let i1 = clampIndex(base)
            let i2 = clampIndex(base + 1)
            let i3 = clampIndex(base + 2)

            var w0 = cubicKernel(-1 - frac)
            var w1 = cubicKernel(-frac)
            var w2 = cubicKernel(1 - frac)
            var w3 = cubicKernel(2 - frac)

            let sum = w0 + w1 + w2 + w3
            if sum != 0 {
                w0 /= sum
                w1 /= sum
                w2 /= sum
                w3 /= sum
            }

            coeffs.append(ResizeCoeff(i0: i0, i1: i1, i2: i2, i3: i3, w0: w0, w1: w1, w2: w2, w3: w3))
        }

        return coeffs
    }

    private static func resizeBicubicRGB(
        rgba: Data,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int
    ) throws -> Data {
        guard srcWidth > 0, srcHeight > 0, dstWidth > 0, dstHeight > 0 else {
            throw GLMOCRImageProcessorError.resizeFailed
        }
        guard rgba.count == srcWidth * srcHeight * 4 else {
            throw GLMOCRImageProcessorError.resizeFailed
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
                    throw GLMOCRImageProcessorError.resizeFailed
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

                func clampToByte(_ value: Double) -> UInt8 {
                    if value <= 0 { return 0 }
                    if value >= 255 { return 255 }
                    return UInt8(Int(value + 0.5))
                }

                let outBaseAddress = Int(bitPattern: outBase)
                let srcBaseAddress = Int(bitPattern: srcBase)

                let shouldParallelize =
                    (ProcessInfo.processInfo.activeProcessorCount >= 4
                        && (dstWidth * dstHeight) >= 1_000_000)

                if !shouldParallelize {
                    let rowCacheSize = 8
                    var rowCacheSourceY = Array(repeating: Int.min, count: rowCacheSize)
                    var rowCacheRows = Array(
                        repeating: [Double](repeating: 0, count: rowElementCount),
                        count: rowCacheSize
                    )
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
                } else {
                    let chunkHeight = 32
                    let chunkCount = (dstHeight + chunkHeight - 1) / chunkHeight

                    DispatchQueue.concurrentPerform(iterations: chunkCount) { chunkIndex in
                        guard let outBase = UnsafeMutablePointer<UInt8>(bitPattern: outBaseAddress),
                            let srcBase = UnsafePointer<UInt8>(bitPattern: srcBaseAddress)
                        else {
                            return
                        }

                        @inline(__always)
                        func readRGB(y: Int, x: Int) -> (Double, Double, Double) {
                            let idx = (y * srcWidth + x) * 4
                            return (
                                Double(srcBase[idx]),
                                Double(srcBase[idx + 1]),
                                Double(srcBase[idx + 2])
                            )
                        }

                        @inline(__always)
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

                        @inline(__always)
                        func clampToByte(_ value: Double) -> UInt8 {
                            if value <= 0 { return 0 }
                            if value >= 255 { return 255 }
                            return UInt8(Int(value + 0.5))
                        }

                        let yStart = chunkIndex * chunkHeight
                        let yEnd = min(yStart + chunkHeight, dstHeight)
                        guard yStart < yEnd else { return }

                        let rowCacheSize = 8
                        var rowCacheSourceY = Array(repeating: Int.min, count: rowCacheSize)
                        var rowCacheRows = Array(
                            repeating: [Double](repeating: 0, count: rowElementCount),
                            count: rowCacheSize
                        )
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

                        for yOut in yStart..<yEnd {
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
            }
        }

        return out
    }

    static func smartResize(
        numFrames: Int,
        height: Int,
        width: Int,
        temporalFactor: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int
    ) throws -> (height: Int, width: Int) {
        guard numFrames >= temporalFactor else {
            throw GLMOCRImageProcessorError.invalidTemporalArguments(
                numFrames: numFrames, temporalFactor: temporalFactor)
        }

        var height = height
        var width = width

        if height < factor || width < factor {
            let scale = max(Double(factor) / Double(height), Double(factor) / Double(width))
            height = Int(Double(height) * scale)
            width = Int(Double(width) * scale)
        }

        let aspectRatio = Double(max(height, width)) / Double(min(height, width))
        guard aspectRatio <= 200 else {
            throw GLMOCRImageProcessorError.invalidAspectRatio(aspectRatio)
        }

        var hBar = Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var wBar = Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
        let tBar = Int((Double(numFrames) / Double(temporalFactor)).rounded(.toNearestOrEven)) * temporalFactor

        let scaledPixels = tBar * hBar * wBar
        if scaledPixels > maxPixels {
            let beta = sqrt((Double(numFrames * height * width)) / Double(maxPixels))
            hBar = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            wBar = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if scaledPixels < minPixels {
            let beta = sqrt(Double(minPixels) / Double(numFrames * height * width))
            hBar = Int(ceil(Double(height) * beta / Double(factor))) * factor
            wBar = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (height: hBar, width: wBar)
    }
}

public enum GLMOCRImageProcessorError: Error, Sendable {
    case imageLoadFailed(String)
    case invalidImageExtent(width: Double, height: Double)
    case invalidNormalizationConfig(meanCount: Int, stdCount: Int)
    case invalidTemporalArguments(numFrames: Int, temporalFactor: Int)
    case invalidAspectRatio(Double)
    case bitmapAllocationFailed
    case colorSpaceFailed
    case resizeFailed
    case jpegRoundTripFailed
}
