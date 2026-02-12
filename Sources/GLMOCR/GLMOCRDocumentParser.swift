import CoreGraphics
import Foundation
import MLX

public typealias GLMOCRParseProgressHandler = (String) -> Void

public enum GLMOCRRegionTaskType: String, Sendable, Codable {
    case text
    case table
    case formula
    case image
    case skip
    case abandon
}

public struct GLMOCRRegionRecognitionConfig: Sendable {
    public var textPrompt: String
    public var tablePrompt: String
    public var formulaPrompt: String

    public init(
        textPrompt: String = GLMOCRPromptPresets.textRecognition,
        tablePrompt: String = GLMOCRPromptPresets.tableRecognition,
        formulaPrompt: String = GLMOCRPromptPresets.formulaRecognition
    ) {
        self.textPrompt = textPrompt
        self.tablePrompt = tablePrompt
        self.formulaPrompt = formulaPrompt
    }
}

public struct GLMOCRDocumentParserConfig: Sendable {
    public var threshold: Double
    public var layoutThresholdByClass: [Int: Double]?
    public var layoutNMS: Bool
    public var includeFormulaNumbers: Bool
    public var usePolygonMask: Bool
    public var useMaskCrop: Bool
    public var reencodeRegionCropsToJPEG: Bool
    public var jpegQuality: Double
    public var maxNewTokensPerRegion: Int
    public var maxNewTokensTextPerRegion: Int?
    public var maxNewTokensTablePerRegion: Int?
    public var maxNewTokensFormulaPerRegion: Int?
    public var maxRegionsPerPage: Int?
    public var generationParameters: GLMOCRGenerationParameters
    public var dtypeOverride: DType?
    public var prompts: GLMOCRRegionRecognitionConfig

    public init(
        threshold: Double = 0.4,
        layoutThresholdByClass: [Int: Double]? = nil,
        layoutNMS: Bool = true,
        includeFormulaNumbers: Bool = true,
        usePolygonMask: Bool = false,
        useMaskCrop: Bool = false,
        reencodeRegionCropsToJPEG: Bool = false,
        jpegQuality: Double = 0.95,
        maxNewTokensPerRegion: Int = 1024,
        maxNewTokensTextPerRegion: Int? = nil,
        maxNewTokensTablePerRegion: Int? = nil,
        maxNewTokensFormulaPerRegion: Int? = nil,
        maxRegionsPerPage: Int? = nil,
        generationParameters: GLMOCRGenerationParameters = .default,
        dtypeOverride: DType? = nil,
        prompts: GLMOCRRegionRecognitionConfig = .init()
    ) {
        self.threshold = threshold
        self.layoutThresholdByClass = layoutThresholdByClass
        self.layoutNMS = layoutNMS
        self.includeFormulaNumbers = includeFormulaNumbers
        self.usePolygonMask = usePolygonMask
        self.useMaskCrop = useMaskCrop
        self.reencodeRegionCropsToJPEG = reencodeRegionCropsToJPEG
        self.jpegQuality = jpegQuality
        self.maxNewTokensPerRegion = maxNewTokensPerRegion
        self.maxNewTokensTextPerRegion = maxNewTokensTextPerRegion
        self.maxNewTokensTablePerRegion = maxNewTokensTablePerRegion
        self.maxNewTokensFormulaPerRegion = maxNewTokensFormulaPerRegion
        self.maxRegionsPerPage = maxRegionsPerPage
        self.generationParameters = generationParameters
        self.dtypeOverride = dtypeOverride
        self.prompts = prompts
    }
}

public final class GLMOCRDocumentParser {
    private let ocr: GLMOCRPipeline
    private let layout: PPDocLayoutV3Pipeline
    private let formatter: GLMOCRResultFormatter

    public init(
        ocr: GLMOCRPipeline,
        layout: PPDocLayoutV3Pipeline,
        formatter: GLMOCRResultFormatter = .init()
    ) {
        self.ocr = ocr
        self.layout = layout
        self.formatter = formatter
    }

    public func parse(
        image: CGImage,
        config: GLMOCRDocumentParserConfig = .init(),
        progress: GLMOCRParseProgressHandler? = nil
    ) throws -> GLMOCRParseResult {
        let result = try parse(images: [image], config: config, progress: progress)
        precondition(result.jsonResult.count == 1, "internal error: expected single-page result")
        return result
    }

    public func parse(
        images: [CGImage],
        config: GLMOCRDocumentParserConfig = .init(),
        progress: GLMOCRParseProgressHandler? = nil
    ) throws -> GLMOCRParseResult {
        precondition(!images.isEmpty, "images must be non-empty")

        var pages: [[GLMOCRFormattedRegion]] = []
        pages.reserveCapacity(images.count)

        let layoutBatchSize = 8
        progress?("layout start: \(images.count) page(s), batch \(layoutBatchSize)")
        var start = 0
        while start < images.count {
            let end = min(start + layoutBatchSize, images.count)
            progress?("layout batch: pages \(start + 1)-\(end)/\(images.count)")
            let imageBatch = Array(images[start..<end])
            let detectionsBatch = try layout.detect(
                images: imageBatch,
                threshold: config.threshold,
                thresholdByClass: config.layoutThresholdByClass,
                layoutNMS: config.layoutNMS,
                includePolygons: config.usePolygonMask,
                includeMasks: config.usePolygonMask || config.useMaskCrop
            )
            precondition(detectionsBatch.count == imageBatch.count, "internal error: layout batch size mismatch")
            let regionCount = detectionsBatch.reduce(into: 0) { $0 += $1.count }
            progress?("layout batch done: pages \(start + 1)-\(end)/\(images.count), regions \(regionCount)")

            let parsedBatch = try parsePages(
                images: imageBatch,
                detectionsBatch: detectionsBatch,
                config: config,
                pageOffset: start,
                totalPageCount: images.count,
                progress: progress
            )
            precondition(parsedBatch.count == imageBatch.count, "internal error: parsed page batch size mismatch")
            pages.append(contentsOf: parsedBatch)
            progress?("parse batch done: pages \(start + 1)-\(end)/\(images.count)")

            start = end
        }

        progress?("formatting result")
        return formatter.format(pages: pages)
    }

    private func parsePages(
        images: [CGImage],
        detectionsBatch: [[PPDocLayoutV3PaddleBox]],
        config: GLMOCRDocumentParserConfig,
        pageOffset: Int,
        totalPageCount: Int,
        progress: GLMOCRParseProgressHandler?
    ) throws -> [[GLMOCRFormattedRegion]] {
        precondition(images.count == detectionsBatch.count, "internal error: image/detection batch size mismatch")

        struct PendingOCR {
            let pageIndex: Int
            let regionIndex: Int
            let crop: CGImage
            let maxNewTokens: Int
        }

        var pages: [[GLMOCRFormattedRegion]] = []
        pages.reserveCapacity(images.count)
        var byPrompt: [String: [PendingOCR]] = [:]
        byPrompt.reserveCapacity(3)

        for (pageIndex, (image, detections)) in zip(images, detectionsBatch).enumerated() {
            let globalPageIndex = pageOffset + pageIndex + 1
            if let maxRegionsPerPage = config.maxRegionsPerPage,
               maxRegionsPerPage > 0,
               detections.count > maxRegionsPerPage
            {
                throw GLMOCRDocumentParserError.tooManyRegions(
                    detected: detections.count,
                    limit: maxRegionsPerPage
                )
            }

            func refineBBoxPxForSealIfNeeded(_ bboxPx: [Int], label: String) -> [Int] {
                guard bboxPx.count == 4 else { return bboxPx }
                guard label == "seal" else { return bboxPx }

                let imageWidth = max(image.width, 1)
                let imageHeight = max(image.height, 1)
                let x1 = max(0, min(bboxPx[0], imageWidth))
                var y1 = max(0, min(bboxPx[1], imageHeight))
                var x2 = max(0, min(bboxPx[2], imageWidth))
                let y2 = max(0, min(bboxPx[3], imageHeight))

                let boxWidth = max(0, x2 - x1)
                let boxHeight = max(0, y2 - y1)
                let widthRatio = Double(boxWidth) / Double(imageWidth)
                let heightRatio = Double(boxHeight) / Double(imageHeight)

                if widthRatio >= 0.98, heightRatio >= 0.98 {
                    let trimRight = max(1, Int(round(Double(imageWidth) * 0.01)))
                    let trimTop = max(1, Int(round(Double(imageHeight) * 0.006)))
                    x2 = max(x1 + 1, x2 - trimRight)
                    y1 = min(y2 - 1, y1 + trimTop)
                }

                return [x1, y1, x2, y2]
            }

            func normalizeBBoxTo1000(_ bboxPx: [Int]) -> [Int] {
                guard bboxPx.count == 4 else { return bboxPx }
                let width = Double(image.width)
                let height = Double(image.height)
                guard width > 0, height > 0 else { return bboxPx }
                let x1 = Int(Double(bboxPx[0]) / width * 1000.0)
                let y1 = Int(Double(bboxPx[1]) / height * 1000.0)
                let x2 = Int(Double(bboxPx[2]) / width * 1000.0)
                let y2 = Int(Double(bboxPx[3]) / height * 1000.0)
                return [x1, y1, x2, y2]
            }

            func denormalizeBBoxFrom1000(_ bbox2D: [Int]) -> [Int] {
                guard bbox2D.count == 4 else { return bbox2D }
                let width = Double(image.width)
                let height = Double(image.height)
                guard width > 0, height > 0 else { return bbox2D }
                let x1 = Int(Double(bbox2D[0]) * width / 1000.0)
                let y1 = Int(Double(bbox2D[1]) * height / 1000.0)
                let x2 = Int(Double(bbox2D[2]) * width / 1000.0)
                let y2 = Int(Double(bbox2D[3]) * height / 1000.0)
                return [x1, y1, x2, y2]
            }

            func horizontalOverlapRatio(_ a: [Int], _ b: [Int]) -> Double {
                guard a.count == 4, b.count == 4 else { return 0 }
                let left = max(a[0], b[0])
                let right = min(a[2], b[2])
                let overlap = max(0, right - left)
                let base = max(1, min(a[2] - a[0], b[2] - b[0]))
                return Double(overlap) / Double(base)
            }

            func isFigureTitleAbove(
                imageBBox: [Int],
                candidateBBox: [Int],
                maxGapPx: Int
            ) -> Bool {
                guard imageBBox.count == 4, candidateBBox.count == 4 else { return false }
                let verticalGap = imageBBox[1] - candidateBBox[3]
                guard verticalGap >= 0, verticalGap <= maxGapPx else { return false }
                return horizontalOverlapRatio(imageBBox, candidateBBox) >= 0.5
            }

            func isFigureTitleBelow(
                imageBBox: [Int],
                candidateBBox: [Int],
                maxGapPx: Int
            ) -> Bool {
                guard imageBBox.count == 4, candidateBBox.count == 4 else { return false }
                let verticalGap = candidateBBox[1] - imageBBox[3]
                guard verticalGap >= 0, verticalGap <= maxGapPx else { return false }
                return horizontalOverlapRatio(imageBBox, candidateBBox) >= 0.5
            }

            var tasks = detections.map {
                taskType(forNativeLabel: $0.label, includeFormulaNumbers: config.includeFormulaNumbers)
            }
            var nativeLabelOverrides: [String?] = Array(repeating: nil, count: detections.count)

            let figureTitleIndices = detections.indices.filter { detections[$0].label == "figure_title" }
            if !figureTitleIndices.isEmpty {
                let maxGapPx = max(16, Int(round(Double(image.height) * 0.02)))
                for detectionIndex in detections.indices where tasks[detectionIndex] == .image {
                    guard detections[detectionIndex].label == "image" else { continue }
                    let imageBBox = detections[detectionIndex].coordinate
                    guard imageBBox.count == 4 else { continue }

                    var hasFigureTitleAbove = false
                    var hasFigureTitleBelow = false
                    for titleIndex in figureTitleIndices where titleIndex != detectionIndex {
                        let titleBBox = detections[titleIndex].coordinate
                        if isFigureTitleAbove(imageBBox: imageBBox, candidateBBox: titleBBox, maxGapPx: maxGapPx) {
                            hasFigureTitleAbove = true
                        }
                        if isFigureTitleBelow(imageBBox: imageBBox, candidateBBox: titleBBox, maxGapPx: maxGapPx) {
                            hasFigureTitleBelow = true
                        }
                        if hasFigureTitleAbove && hasFigureTitleBelow {
                            break
                        }
                    }

                    if hasFigureTitleAbove && hasFigureTitleBelow {
                        tasks[detectionIndex] = .text
                        nativeLabelOverrides[detectionIndex] = "algorithm"
                    }
                }
            }

            var regions: [GLMOCRFormattedRegion] = []
            regions.reserveCapacity(detections.count)
            var pendingOCRCount = 0

            for (detectionIndex, box) in detections.enumerated() {
                let task = tasks[detectionIndex]
                let nativeLabel = nativeLabelOverrides[detectionIndex] ?? box.label
                let bboxPx = refineBBoxPxForSealIfNeeded(box.coordinate, label: nativeLabel)
                let bbox2D = normalizeBBoxTo1000(bboxPx)
                let cropBBoxPx = denormalizeBBoxFrom1000(bbox2D)

                switch task {
                case .abandon:
                    continue
                case .skip, .image:
                    regions.append(
                        GLMOCRFormattedRegion(
                            index: 0,
                            label: "image",
                            bbox2D: bbox2D,
                            content: nil,
                            nativeLabel: nativeLabel
                        )
                    )
                case .text, .table, .formula:
                    pendingOCRCount += 1
                    let prompt = prompt(for: task, prompts: config.prompts)
                    let configuredMaxNewTokens: Int = switch task {
                    case .text: config.maxNewTokensTextPerRegion ?? config.maxNewTokensPerRegion
                    case .table: config.maxNewTokensTablePerRegion ?? config.maxNewTokensPerRegion
                    case .formula: config.maxNewTokensFormulaPerRegion ?? config.maxNewTokensPerRegion
                    default: config.maxNewTokensPerRegion
                    }
                    let maxNewTokensForRegion = Self.effectiveMaxNewTokensForRegion(
                        task: task,
                        nativeLabel: nativeLabel,
                        bbox2D: bbox2D,
                        configuredMaxNewTokens: configuredMaxNewTokens
                    )
                    let polygonPx: [[Int]]? = {
                        guard config.usePolygonMask else { return nil }
                        guard box.polygonPoints.count >= 3 else { return nil }
                        let width = Double(image.width)
                        let height = Double(image.height)
                        return box.polygonPoints.compactMap { pt -> [Int]? in
                            guard pt.count >= 2 else { return nil }
                            guard width > 0, height > 0 else {
                                return [Int(pt[0]), Int(pt[1])]
                            }
                            let xNorm = Int(Double(pt[0]) / width * 1000.0)
                            let yNorm = Int(Double(pt[1]) / height * 1000.0)
                            let xPx = Int(Double(xNorm) * width / 1000.0)
                            let yPx = Int(Double(yNorm) * height / 1000.0)
                            return [xPx, yPx]
                        }
                    }()

                    let cropped = try crop(
                        image: image,
                        bboxPx: cropBBoxPx,
                        polygonPx: polygonPx,
                        mask: config.useMaskCrop ? box.mask : nil
                    )

                    regions.append(
                        GLMOCRFormattedRegion(
                            index: 0,
                            label: task.rawValue,
                            bbox2D: bbox2D,
                            content: nil,
                            nativeLabel: nativeLabel
                        )
                    )

                    byPrompt[prompt, default: []].append(
                        PendingOCR(
                            pageIndex: pageIndex,
                            regionIndex: regions.count - 1,
                            crop: cropped,
                            maxNewTokens: maxNewTokensForRegion
                        )
                    )
                }
            }

            pages.append(regions)
            progress?("page \(globalPageIndex)/\(totalPageCount): regions \(regions.count), ocr \(pendingOCRCount)")
        }

        if !byPrompt.isEmpty {
            let prompts = Array(byPrompt.keys)
            for prompt in prompts {
                guard let group = byPrompt[prompt], !group.isEmpty else { continue }
                let promptName: String = {
                    if prompt == config.prompts.textPrompt { return "text" }
                    if prompt == config.prompts.tablePrompt { return "table" }
                    if prompt == config.prompts.formulaPrompt { return "formula" }
                    return "custom"
                }()
                progress?("ocr \(promptName): start \(group.count) region(s)")
                let crops = group.map(\.crop)
                let maxNewTokensPerImage = group.map(\.maxNewTokens)
                let groupMaxNewTokens = maxNewTokensPerImage.max() ?? config.maxNewTokensPerRegion
                let outputs = try ocr.recognizeBatch(
                    images: crops,
                    prompt: prompt,
                    maxNewTokens: groupMaxNewTokens,
                    generationParameters: config.generationParameters,
                    maxNewTokensPerImage: maxNewTokensPerImage,
                    skipSpecialTokens: true,
                    dtypeOverride: config.dtypeOverride,
                    postResizeJPEGRoundTripQuality: config.reencodeRegionCropsToJPEG ? config.jpegQuality : nil
                )
                precondition(outputs.count == group.count, "internal error: OCR batch size mismatch")
                progress?("ocr \(promptName): done \(group.count) region(s)")

                for (item, text) in zip(group, outputs) {
                    pages[item.pageIndex][item.regionIndex].content = text
                }

                byPrompt[prompt] = nil
            }
        }

        for pageIndex in 0..<pages.count {
            for i in 0..<pages[pageIndex].count {
                pages[pageIndex][i].index = i
            }
        }

        return pages
    }

    static func effectiveMaxNewTokensForRegion(
        task: GLMOCRRegionTaskType,
        nativeLabel: String,
        bbox2D: [Int],
        configuredMaxNewTokens: Int
    ) -> Int {
        let base = max(1, configuredMaxNewTokens)
        guard task == .text else { return base }

        var resolved = base
        switch nativeLabel {
        case "doc_title", "paragraph_title", "figure_title":
            resolved = min(resolved, 96)
        default:
            break
        }

        guard bbox2D.count == 4 else { return resolved }
        let width = max(0, bbox2D[2] - bbox2D[0])
        let height = max(0, bbox2D[3] - bbox2D[1])
        let area = width * height

        if height <= 20 {
            resolved = min(resolved, 96)
        } else if height <= 40 {
            resolved = min(resolved, 160)
        } else if height <= 64 {
            resolved = min(resolved, 256)
        } else if height <= 96 {
            resolved = min(resolved, 384)
        }

        if area <= 5_000 {
            resolved = min(resolved, 96)
        } else if area <= 9_000 {
            resolved = min(resolved, 160)
        }

        return max(1, resolved)
    }

    private func taskType(forNativeLabel label: String, includeFormulaNumbers: Bool) -> GLMOCRRegionTaskType {
        switch label {
        case "table":
            return .table
        case "display_formula", "inline_formula", "formula":
            return .formula
        case "chart", "image":
            return .image
        case "formula_number":
            return includeFormulaNumbers ? .text : .abandon
        case "header", "footer", "number", "footnote", "aside_text", "reference", "footer_image", "header_image":
            return .abandon
        default:
            return .text
        }
    }

    private func prompt(for task: GLMOCRRegionTaskType, prompts: GLMOCRRegionRecognitionConfig) -> String {
        switch task {
        case .text: prompts.textPrompt
        case .table: prompts.tablePrompt
        case .formula: prompts.formulaPrompt
        default: prompts.textPrompt
        }
    }

    private func crop(
        image: CGImage,
        bboxPx: [Int],
        polygonPx: [[Int]]?,
        mask: PPDocLayoutV3Mask?
    ) throws -> CGImage {
        guard bboxPx.count == 4 else {
            throw GLMOCRDocumentParserError.invalidBBox(bboxPx)
        }
        let width = image.width
        let height = image.height
        var x1 = bboxPx[0]
        var y1 = bboxPx[1]
        var x2 = bboxPx[2]
        var y2 = bboxPx[3]

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 || y2 <= y1 {
            throw GLMOCRDocumentParserError.invalidBBox(bboxPx)
        }

        let cropRect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        guard let cg = image.cropping(to: cropRect) else {
            throw GLMOCRDocumentParserError.cropFailed(bboxPx)
        }

        if let mask,
           let masked = Self.apply(
               mask: mask,
               toCrop: cg,
               inImageSize: (width: width, height: height),
               cropOriginPx: (x: x1, y: y1)
           )
        {
            return masked
        }

        guard let polygonPx, polygonPx.count >= 3 else {
            return cg
        }

        let cropWidth = cg.width
        let cropHeight = cg.height
        guard cropWidth > 0, cropHeight > 0 else { return cg }

        let points: [(Int, Int)] = polygonPx.compactMap { pt in
            guard pt.count >= 2 else { return nil }
            let px = pt[0] - x1
            let py = pt[1] - y1
            return (px, py)
        }
        guard points.count >= 3 else { return cg }

        let bytesPerPixel = 4
        let bytesPerRow = cropWidth * bytesPerPixel
        var data = Data(count: cropHeight * bytesPerRow)
        var didRender = false
        data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return }

            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            guard let context = CGContext(
                data: baseAddress,
                width: cropWidth,
                height: cropHeight,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else { return }

            context.setAllowsAntialiasing(false)
            context.setShouldAntialias(false)
            context.interpolationQuality = .none
            context.draw(cg, in: CGRect(x: 0, y: 0, width: cropWidth, height: cropHeight))
            didRender = true
        }
        guard didRender else { return cg }

        data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.bindMemory(to: UInt8.self).baseAddress else { return }
            let buffer = UnsafeMutableBufferPointer(start: baseAddress, count: cropHeight * bytesPerRow)
            Self.applyPolygonMask(width: cropWidth, height: cropHeight, rgba: buffer, points: points)
        }

        return Self.makeRGBAImage(width: cropWidth, height: cropHeight, bytes: data) ?? cg
    }

    internal static func applyPolygonMask(
        width: Int,
        height: Int,
        rgba: UnsafeMutableBufferPointer<UInt8>,
        points: [(Int, Int)],
        fill: UInt8 = 255
    ) {
        let pixelCount = width * height
        guard width > 0, height > 0 else { return }
        guard rgba.count >= pixelCount * 4 else { return }

        let mask = rasterizePolygonMask(width: width, height: height, points: points)
        guard mask.count == pixelCount else { return }

        for i in 0..<pixelCount {
            if mask[i] == 0 {
                let offset = i * 4
                rgba[offset] = fill
                rgba[offset + 1] = fill
                rgba[offset + 2] = fill
                rgba[offset + 3] = 255
            }
        }
    }

    internal static func rasterizePolygonMask(
        width: Int,
        height: Int,
        points: [(Int, Int)]
    ) -> [UInt8] {
        guard width > 0, height > 0 else { return [] }
        var mask = [UInt8](repeating: 0, count: width * height)
        guard points.count >= 3 else { return mask }

        var minY = Int.max
        var maxY = Int.min
        for (_, y) in points {
            minY = min(minY, y)
            maxY = max(maxY, y)
        }

        if minY > maxY { return mask }
        let yStart = max(0, min(minY, height - 1))
        let yEnd = max(0, min(maxY, height - 1))
        if yStart > yEnd { return mask }

        let count = points.count

        func setMask(x: Int, y: Int) {
            guard x >= 0, x < width, y >= 0, y < height else { return }
            mask[y * width + x] = 1
        }

        func drawLine(x0: Int, y0: Int, x1: Int, y1: Int) {
            var x0 = x0
            var y0 = y0
            let dx = abs(x1 - x0)
            let sx = x0 < x1 ? 1 : -1
            let dy = -abs(y1 - y0)
            let sy = y0 < y1 ? 1 : -1
            var err = dx + dy
            while true {
                setMask(x: x0, y: y0)
                if x0 == x1 && y0 == y1 { break }
                let e2 = 2 * err
                if e2 >= dy {
                    err += dy
                    x0 += sx
                }
                if e2 <= dx {
                    err += dx
                    y0 += sy
                }
            }
        }

        for i in 0..<count {
            let j = (i + 1) % count
            let p0 = points[i]
            let p1 = points[j]
            drawLine(x0: p0.0, y0: p0.1, x1: p1.0, y1: p1.1)
        }

        var intersections = [Double]()
        intersections.reserveCapacity(count)

        for y in yStart...yEnd {
            intersections.removeAll(keepingCapacity: true)

            for i in 0..<count {
                let j = (i + 1) % count
                let (x0, y0) = points[i]
                let (x1, y1) = points[j]
                if y0 == y1 { continue }

                if y0 < y1 {
                    if y < y0 || y >= y1 { continue }
                } else {
                    if y < y1 || y >= y0 { continue }
                }

                let dy = Double(y1 - y0)
                let dx = Double(x1 - x0)
                let t = Double(y - y0) / dy
                let x = Double(x0) + t * dx
                intersections.append(x)
            }

            if intersections.count < 2 { continue }
            intersections.sort()

            var idx = 0
            while idx + 1 < intersections.count {
                let xStart = intersections[idx]
                let xEnd = intersections[idx + 1]
                var start = Int(ceil(min(xStart, xEnd)))
                var end = Int(floor(max(xStart, xEnd)))
                if start < 0 { start = 0 }
                if end >= width { end = width - 1 }
                if end >= start {
                    let rowBase = y * width
                    for x in start...end {
                        mask[rowBase + x] = 1
                    }
                }
                idx += 2
            }
        }

        return mask
    }

    internal static func apply(
        mask: PPDocLayoutV3Mask,
        toCrop crop: CGImage,
        inImageSize imageSize: (width: Int, height: Int),
        cropOriginPx: (x: Int, y: Int)
    ) -> CGImage? {
        let cropWidth = crop.width
        let cropHeight = crop.height
        guard let cropMask = Self.resizedMaskForCrop(
            mask: mask,
            cropSize: (width: cropWidth, height: cropHeight),
            inImageSize: imageSize,
            cropOriginPx: cropOriginPx
        ) else { return nil }

        let bytesPerPixel = 4
        let bytesPerRow = cropWidth * bytesPerPixel
        var data = Data(count: cropHeight * bytesPerRow)
        var didRender = false
        data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return }
            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            guard let context = CGContext(
                data: baseAddress,
                width: cropWidth,
                height: cropHeight,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else { return }

            context.interpolationQuality = .none
            context.draw(crop, in: CGRect(x: 0, y: 0, width: cropWidth, height: cropHeight))
            didRender = true
        }
        guard didRender else { return nil }

        data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.bindMemory(to: UInt8.self).baseAddress else { return }
            let buffer = UnsafeMutableBufferPointer(start: baseAddress, count: cropHeight * bytesPerRow)
            Self.applyMaskCrop(width: cropWidth, height: cropHeight, rgba: buffer, mask: cropMask)
        }

        return Self.makeRGBAImage(width: cropWidth, height: cropHeight, bytes: data)
    }

    internal static func resizedMaskForCrop(
        mask: PPDocLayoutV3Mask,
        cropSize: (width: Int, height: Int),
        inImageSize imageSize: (width: Int, height: Int),
        cropOriginPx: (x: Int, y: Int)
    ) -> [UInt8]? {
        let cropWidth = cropSize.width
        let cropHeight = cropSize.height
        guard cropWidth > 0, cropHeight > 0 else { return nil }
        guard mask.width > 0, mask.height > 0 else { return nil }
        guard mask.data.count == mask.width * mask.height else { return nil }

        func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(v, hi)) }
        func roundToEvenInt(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

        let scaleW = Double(mask.width) / Double(max(imageSize.width, 1))
        let scaleH = Double(mask.height) / Double(max(imageSize.height, 1))

        let x1 = cropOriginPx.x
        let y1 = cropOriginPx.y
        let x2 = x1 + cropWidth
        let y2 = y1 + cropHeight

        let xStart = clamp(roundToEvenInt(Double(x1) * scaleW), 0, mask.width)
        let xEnd = clamp(roundToEvenInt(Double(x2) * scaleW), 0, mask.width)
        let yStart = clamp(roundToEvenInt(Double(y1) * scaleH), 0, mask.height)
        let yEnd = clamp(roundToEvenInt(Double(y2) * scaleH), 0, mask.height)

        let xs = min(xStart, xEnd)
        let xe = max(xStart, xEnd)
        let ys = min(yStart, yEnd)
        let ye = max(yStart, yEnd)

        let cropMaskW = xe - xs
        let cropMaskH = ye - ys
        guard cropMaskW > 0, cropMaskH > 0 else { return nil }

        
        var out = [UInt8](repeating: 0, count: cropWidth * cropHeight)
        for y in 0..<cropHeight {
            let srcYOffset = Int(Double(y) * Double(cropMaskH) / Double(cropHeight))
            let srcY = ys + min(srcYOffset, cropMaskH - 1)
            let maskRowBase = srcY * mask.width
            let outRowBase = y * cropWidth
            for x in 0..<cropWidth {
                let srcXOffset = Int(Double(x) * Double(cropMaskW) / Double(cropWidth))
                let srcX = xs + min(srcXOffset, cropMaskW - 1)
                out[outRowBase + x] = mask.data[maskRowBase + srcX] != 0 ? 1 : 0
            }
        }
        return out
    }

    internal static func applyMaskCrop(
        width: Int,
        height: Int,
        rgba: UnsafeMutableBufferPointer<UInt8>,
        mask: [UInt8],
        fill: UInt8 = 255
    ) {
        let pixelCount = width * height
        guard width > 0, height > 0 else { return }
        guard rgba.count >= pixelCount * 4 else { return }
        guard mask.count == pixelCount else { return }

        for i in 0..<pixelCount where mask[i] == 0 {
            let offset = i * 4
            rgba[offset] = fill
            rgba[offset + 1] = fill
            rgba[offset + 2] = fill
            rgba[offset + 3] = 255
        }
    }

    private static func makeRGBAImage(width: Int, height: Int, bytes: Data) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        guard bytes.count == height * bytesPerRow else { return nil }
        guard let provider = CGDataProvider(data: bytes as CFData) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        return CGImage(
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
    }

}

public enum GLMOCRDocumentParserError: Error, Sendable {
    case invalidBBox([Int])
    case cropFailed([Int])
    case tooManyRegions(detected: Int, limit: Int)
}
