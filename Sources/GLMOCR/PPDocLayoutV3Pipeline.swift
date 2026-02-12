import CoreGraphics
import Foundation
import MLX

public final class PPDocLayoutV3Pipeline {
    public let modelURL: URL
    public let config: PPDocLayoutV3Config

    private let model: PPDocLayoutV3ForObjectDetection
    private var imageProcessor = PPDocLayoutV3ImageProcessor()

    private static let defaultMergeModeByClass: [Int: PPDocLayoutV3MergeMode] = [
        0: .large,
        1: .large,
        2: .large,
        3: .large,
        4: .large,
        5: .large,
        6: .large,
        7: .large,
        8: .large,
        9: .large,
        10: .large,
        11: .large,
        12: .large,
        13: .large,
        14: .large,
        15: .large,
        16: .large,
        17: .large,
        18: .small,
        19: .large,
        20: .large,
        21: .large,
        22: .large,
        23: .large,
        24: .large,
    ]

    private static let glmocrId2Label: [Int: String] = [
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "display_formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer_image",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header_image",
        14: "image",
        15: "inline_formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "vertical_text",
        24: "vision_footnote",
    ]

    public init(modelURL: URL, dtype: DType? = nil) throws {
        self.modelURL = modelURL

        let loaded = try PPDocLayoutV3ModelLoader.load(from: modelURL, dtype: dtype)
        self.model = loaded.model
        self.config = loaded.config
    }

    public func detect(
        images: [CGImage],
        threshold: Double = 0.4,
        thresholdByClass: [Int: Double]? = nil,
        layoutNMS: Bool = true,
        includePolygons: Bool = false,
        includeMasks: Bool = false
    ) throws -> [[PPDocLayoutV3PaddleBox]] {
        precondition(!images.isEmpty, "images must be non-empty")

        let preprocessed = try imageProcessor.process(images)
        let outputs = model(pixelValues: preprocessed.pixelValues)

        let raw = try postProcess(
            outputs: outputs,
            threshold: threshold,
            thresholdByClass: thresholdByClass,
            imageSizes: preprocessed.originalSizes,
            includePolygons: includePolygons,
            includeMasks: includeMasks
        )

        var out: [[PPDocLayoutV3PaddleBox]] = []
        out.reserveCapacity(raw.count)

        for (rawItem, imageSize) in zip(raw, preprocessed.originalSizes) {
            out.append(
                PPDocLayoutV3LayoutPostprocess.apply(
                    raw: rawItem,
                    id2label: Self.glmocrId2Label,
                    imageSize: (width: imageSize.width, height: imageSize.height),
                    layoutNMS: layoutNMS,
                    layoutUnclipRatio: (width: 1.0, height: 1.0),
                    layoutMergeMode: nil,
                    mergeModeByClass: Self.defaultMergeModeByClass
                )
            )
        }

        return out
    }

    public func detect(
        image: CGImage,
        threshold: Double = 0.4,
        thresholdByClass: [Int: Double]? = nil,
        layoutNMS: Bool = true,
        includePolygons: Bool = false,
        includeMasks: Bool = false
    ) throws -> [PPDocLayoutV3PaddleBox] {
        let outputs = try detect(
            images: [image],
            threshold: threshold,
            thresholdByClass: thresholdByClass,
            layoutNMS: layoutNMS,
            includePolygons: includePolygons,
            includeMasks: includeMasks
        )
        precondition(outputs.count == 1)
        return outputs[0]
    }

    private func postProcess(
        outputs: PPDocLayoutV3ForObjectDetectionOutput,
        threshold: Double,
        thresholdByClass: [Int: Double]?,
        imageSizes: [(width: Int, height: Int)],
        includePolygons: Bool,
        includeMasks: Bool
    ) throws -> [PPDocLayoutV3RawDetections] {
        let logits = outputs.logits
        let predBoxes = outputs.predBoxes

        let minNormW = Float(1.0 / 200.0)
        let minNormH = Float(1.0 / 200.0)
        let wh = predBoxes[0..., 0..., 2..<4]
        let validMask = (wh[0..., 0..., 0] .> minNormW) & (wh[0..., 0..., 1] .> minNormH)
        let validMaskExpanded = validMask.expandedDimensions(axis: -1)
        let filteredLogits = which(validMaskExpanded, logits, MLXArray(Float(-100.0), dtype: logits.dtype))

        let batch = filteredLogits.dim(0)
        precondition(imageSizes.count == batch, "imageSizes batch mismatch")
        let numQueries = logits.dim(1)
        let numClasses = logits.dim(2)

        precondition(predBoxes.dim(0) == batch)
        precondition(predBoxes.dim(1) == numQueries)
        precondition(predBoxes.dim(2) == 4)

        let centers = predBoxes[0..., 0..., 0..<2]
        let dims = predBoxes[0..., 0..., 2..<4]
        let half: Float = 0.5
        let topLeft = centers - dims * half
        let bottomRight = centers + dims * half
        var boxesXYXY = concatenated([topLeft, bottomRight], axis: -1)
        var scaleFlat: [Float] = []
        scaleFlat.reserveCapacity(batch * 4)
        for imageSize in imageSizes {
            scaleFlat.append(Float(imageSize.width))
            scaleFlat.append(Float(imageSize.height))
            scaleFlat.append(Float(imageSize.width))
            scaleFlat.append(Float(imageSize.height))
        }
        let scale = MLXArray(scaleFlat).reshaped(batch, 1, 4).asType(boxesXYXY.dtype)
        boxesXYXY = boxesXYXY * scale

        let scores = sigmoid(filteredLogits)
        let scoresFlat = scores.reshaped(batch, numQueries * numClasses)
        let sortedFlat = argSort(-scoresFlat, axis: -1)
        let topkFlat = sortedFlat[0..., 0..<numQueries]

        let topScores = takeAlong(scoresFlat, topkFlat, axis: 1)
        let (queryIndex, labels) = divmod(topkFlat, numClasses)

        let boxIndices = queryIndex.reshaped(batch, numQueries, 1)
        let boxIndicesBroadcast = broadcast(boxIndices, to: [batch, numQueries, 4])
        let selectedBoxes = takeAlong(boxesXYXY, boxIndicesBroadcast, axis: 1)

        let orderSeq = computeOrderSeq(orderLogits: outputs.orderLogits)
        let selectedOrder = takeAlong(orderSeq, queryIndex, axis: 1)

        try checkedEval(topScores, labels, selectedBoxes, selectedOrder)

        let topScoresCPU = topScores.asArray(Float.self)
        let labelsCPU = labels.asArray(Int32.self)
        let boxesCPU = selectedBoxes.asArray(Float.self)
        let orderCPU = selectedOrder.asArray(Int32.self)

        let defaultScoreThreshold = Float(threshold)
        let thresholdsByLabel: [Float]? = {
            guard let thresholdByClass else { return nil }
            var thresholds = Array(repeating: defaultScoreThreshold, count: numClasses)
            for (label, value) in thresholdByClass where label >= 0 && label < numClasses {
                thresholds[label] = Float(value)
            }
            return thresholds
        }()

        func clampInt(_ value: Int, _ lo: Int, _ hi: Int) -> Int {
            max(lo, min(value, hi))
        }

        let needMasks = includeMasks || includePolygons
        let (maskH, maskW): (Int, Int) = needMasks
            ? (outputs.outMasks.dim(2), outputs.outMasks.dim(3))
            : (0, 0)
        let maskPlaneSize = maskH * maskW
        let queryIndexCPU = needMasks ? queryIndex.asArray(Int32.self) : []

        var out: [PPDocLayoutV3RawDetections] = []
        out.reserveCapacity(batch)

        for b in 0..<batch {
            let imageSize = imageSizes[b]
            let base = b * numQueries

            var keepIndices: [Int] = []
            keepIndices.reserveCapacity(numQueries)
            for i in 0..<numQueries {
                let label = Int(labelsCPU[base + i])
                let scoreThreshold: Float
                if let thresholdsByLabel, label >= 0 && label < thresholdsByLabel.count {
                    scoreThreshold = thresholdsByLabel[label]
                } else {
                    scoreThreshold = defaultScoreThreshold
                }
                if topScoresCPU[base + i] >= scoreThreshold {
                    keepIndices.append(i)
                }
            }

            var masksCPU: [UInt8]? = nil
            if needMasks, !keepIndices.isEmpty {
                var keptQueryIndices: [Int32] = []
                keptQueryIndices.reserveCapacity(keepIndices.count)
                for i in keepIndices {
                    keptQueryIndices.append(queryIndexCPU[base + i])
                }

                let masksB = outputs.outMasks[b ..< b + 1, 0..., 0..., 0...]
                let keepCount = keepIndices.count
                let maskIndices = MLXArray(keptQueryIndices).reshaped(1, keepCount, 1, 1)
                let maskIndicesBroadcast = broadcast(maskIndices, to: [1, keepCount, maskH, maskW])
                let selectedMasks = takeAlong(masksB, maskIndicesBroadcast, axis: 1)
                let selectedMasksBinary = (sigmoid(selectedMasks) .> Float(threshold)).asType(.uint8)
                try checkedEval(selectedMasksBinary)
                masksCPU = selectedMasksBinary.asArray(UInt8.self)
            }

            var outScores: [Double] = []
            var outLabels: [Int] = []
            var outBoxes: [[Double]] = []
            var outOrder: [Int] = []
            var outPolygons: [[[Double]]]? = includePolygons ? [] : nil
            var outMasks: [PPDocLayoutV3Mask]? = needMasks ? [] : nil
            outScores.reserveCapacity(keepIndices.count)
            outLabels.reserveCapacity(keepIndices.count)
            outBoxes.reserveCapacity(keepIndices.count)
            outOrder.reserveCapacity(keepIndices.count)
            if includePolygons {
                outPolygons?.reserveCapacity(keepIndices.count)
            }
            if needMasks {
                outMasks?.reserveCapacity(keepIndices.count)
            }

            for (keptMaskIndex, i) in keepIndices.enumerated() {
                let score = Double(topScoresCPU[base + i])
                let label = Int(labelsCPU[base + i])
                let order = Int(orderCPU[base + i])

                let boxBase = (base + i) * 4
                let x1f = Double(boxesCPU[boxBase])
                let y1f = Double(boxesCPU[boxBase + 1])
                let x2f = Double(boxesCPU[boxBase + 2])
                let y2f = Double(boxesCPU[boxBase + 3])

                let x1 = clampInt(Int(x1f), 0, imageSize.width)
                let y1 = clampInt(Int(y1f), 0, imageSize.height)
                let x2 = clampInt(Int(x2f), 0, imageSize.width)
                let y2 = clampInt(Int(y2f), 0, imageSize.height)

                let boxCoordinates = [Double(x1), Double(y1), Double(x2), Double(y2)]
                let rectPolygon: [[Double]]? = includePolygons ? [
                    [Double(x1), Double(y1)],
                    [Double(x2), Double(y1)],
                    [Double(x2), Double(y2)],
                    [Double(x1), Double(y2)],
                ] : nil

                let polygon: [[Double]]?
                let maskValue: PPDocLayoutV3Mask?
                if needMasks {
                    if let masksCPU {
                        let maskStart = keptMaskIndex * maskPlaneSize
                        let maskEnd = maskStart + maskPlaneSize
                        let maskData = Array(masksCPU[maskStart..<maskEnd])
                        let mask = PPDocLayoutV3Mask(width: maskW, height: maskH, data: maskData)
                        maskValue = mask
                        if includePolygons {
                            let poly = PPDocLayoutV3MaskPolygonExtractor.extractPolygonPoints(
                                boxPx: (x1: x1, y1: y1, x2: x2, y2: y2),
                                mask: mask,
                                imageSize: imageSize
                            )
                            if let poly, poly.count >= 4 {
                                polygon = poly
                            } else {
                                polygon = rectPolygon
                            }
                        } else {
                            polygon = nil
                        }
                    } else {
                        polygon = includePolygons ? rectPolygon : nil
                        maskValue = nil
                    }
                } else {
                    polygon = nil
                    maskValue = nil
                }

                outScores.append(score)
                outLabels.append(label)
                outBoxes.append(boxCoordinates)
                outOrder.append(order)
                if needMasks, let maskValue {
                    outMasks?.append(maskValue)
                }
                if includePolygons, let rectPolygon {
                    outPolygons?.append(polygon ?? rectPolygon)
                }
            }

            out.append(
                PPDocLayoutV3RawDetections(
                    scores: outScores,
                    labels: outLabels,
                    boxes: outBoxes,
                    orderSeq: outOrder,
                    polygonPoints: outPolygons,
                    masks: outMasks
                )
            )
        }

        return out
    }

    private func computeOrderSeq(orderLogits: MLXArray) -> MLXArray {
        let orderScores = sigmoid(orderLogits)
        let upperVotes = triu(orderScores, k: 1).sum(axes: [1])
        let one: Float = 1
        let lowerVotes = tril(one - orderScores.transposed(0, 2, 1), k: -1).sum(axes: [1])
        let orderVotes = upperVotes + lowerVotes
        let orderPointers = argSort(orderVotes, axis: -1)
        return argSort(orderPointers, axis: -1)
    }
}
