import Foundation

public enum PPDocLayoutV3MergeMode: String, Sendable, Codable {
    case union
    case large
    case small
}

public struct PPDocLayoutV3Mask: Sendable, Codable {
    public var width: Int
    public var height: Int
    public var data: [UInt8]

    public init(width: Int, height: Int, data: [UInt8]) {
        self.width = width
        self.height = height
        self.data = data
    }
}

public struct PPDocLayoutV3RawDetections: Sendable, Codable {
    public var scores: [Double]
    public var labels: [Int]
    public var boxes: [[Double]]
    public var orderSeq: [Int]
    public var polygonPoints: [[[Double]]]?
    public var masks: [PPDocLayoutV3Mask]?

    public init(
        scores: [Double],
        labels: [Int],
        boxes: [[Double]],
        orderSeq: [Int],
        polygonPoints: [[[Double]]]? = nil,
        masks: [PPDocLayoutV3Mask]? = nil
    ) {
        self.scores = scores
        self.labels = labels
        self.boxes = boxes
        self.orderSeq = orderSeq
        self.polygonPoints = polygonPoints
        self.masks = masks
    }
}

public struct PPDocLayoutV3PaddleBox: Sendable, Codable, Equatable {
    public var clsId: Int
    public var label: String
    public var score: Double
    public var coordinate: [Int]
    public var order: Int?
    public var polygonPoints: [[Double]]
    public var mask: PPDocLayoutV3Mask?

    public init(
        clsId: Int,
        label: String,
        score: Double,
        coordinate: [Int],
        order: Int?,
        polygonPoints: [[Double]],
        mask: PPDocLayoutV3Mask? = nil
    ) {
        self.clsId = clsId
        self.label = label
        self.score = score
        self.coordinate = coordinate
        self.order = order
        self.polygonPoints = polygonPoints
        self.mask = mask
    }

    enum CodingKeys: String, CodingKey {
        case clsId
        case label
        case score
        case coordinate
        case order
        case polygonPoints
    }

    public static func == (lhs: PPDocLayoutV3PaddleBox, rhs: PPDocLayoutV3PaddleBox) -> Bool {
        lhs.clsId == rhs.clsId
            && lhs.label == rhs.label
            && lhs.score == rhs.score
            && lhs.coordinate == rhs.coordinate
            && lhs.order == rhs.order
            && lhs.polygonPoints == rhs.polygonPoints
    }
}

public enum PPDocLayoutV3LayoutPostprocess {
    private struct BoxRecord {
        var clsId: Int
        var score: Double
        var x1: Double
        var y1: Double
        var x2: Double
        var y2: Double
        var order: Int
        var polygonPoints: [[Double]]?
        var mask: PPDocLayoutV3Mask?
    }

    public static func apply(
        raw: PPDocLayoutV3RawDetections,
        id2label: [Int: String],
        imageSize: (width: Int, height: Int),
        layoutNMS: Bool = true,
        layoutUnclipRatio: (width: Double, height: Double)? = nil,
        layoutMergeMode: PPDocLayoutV3MergeMode? = nil,
        mergeModeByClass: [Int: PPDocLayoutV3MergeMode]? = nil
    ) -> [PPDocLayoutV3PaddleBox] {
        let count = min(raw.scores.count, raw.labels.count, raw.boxes.count, raw.orderSeq.count)
        guard count > 0 else { return [] }

        var records: [BoxRecord] = []
        records.reserveCapacity(count)
        for i in 0..<count {
            let box = raw.boxes[i]
            guard box.count >= 4 else { continue }
            let polygon: [[Double]]? = {
                guard let polygons = raw.polygonPoints, i < polygons.count else { return nil }
                return polygons[i]
            }()
            let mask: PPDocLayoutV3Mask? = {
                guard let masks = raw.masks, i < masks.count else { return nil }
                return masks[i]
            }()
            records.append(
                BoxRecord(
                    clsId: raw.labels[i],
                    score: raw.scores[i],
                    x1: box[0],
                    y1: box[1],
                    x2: box[2],
                    y2: box[3],
                    order: raw.orderSeq[i],
                    polygonPoints: polygon,
                    mask: mask
                )
            )
        }

        guard !records.isEmpty else { return [] }

        if layoutNMS {
            let keep = nms(records: records, iouSame: 0.6, iouDiff: 0.98)
            records = keep.map { records[$0] }
        }

        records = filterLargeImages(records: records, id2label: id2label, imageSize: imageSize)

        if let layoutMergeMode, layoutMergeMode != .union {
            records = mergeByContainment(
                records: records,
                id2label: id2label,
                mode: layoutMergeMode
            )
        } else if let mergeModeByClass, !mergeModeByClass.isEmpty {
            records = mergeByContainment(
                records: records,
                id2label: id2label,
                modeByClass: mergeModeByClass
            )
        }

        guard !records.isEmpty else { return [] }

        records.sort { $0.order < $1.order }

        if let layoutUnclipRatio {
            records = unclip(records: records, ratio: layoutUnclipRatio)
        }

        return records.map { record in
            let labelName = id2label[record.clsId] ?? "class_\(record.clsId)"
            let order = record.order > 0 ? record.order : nil
            let x1 = record.x1
            let y1 = record.y1
            let x2 = record.x2
            let y2 = record.y2
            let polygon: [[Double]] = {
                if let polygon = record.polygonPoints, polygon.count >= 3 {
                    return polygon
                }
                return [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]
            }()
            return PPDocLayoutV3PaddleBox(
                clsId: record.clsId,
                label: labelName,
                score: record.score,
                coordinate: [
                    Int(x1),
                    Int(y1),
                    Int(x2),
                    Int(y2),
                ],
                order: order,
                polygonPoints: polygon,
                mask: record.mask
            )
        }
    }

    private static func iou(_ a: BoxRecord, _ b: BoxRecord) -> Double {
        let x1i = max(a.x1, b.x1)
        let y1i = max(a.y1, b.y1)
        let x2i = min(a.x2, b.x2)
        let y2i = min(a.y2, b.y2)

        let interArea = max(0.0, x2i - x1i + 1.0) * max(0.0, y2i - y1i + 1.0)
        let aArea = (a.x2 - a.x1 + 1.0) * (a.y2 - a.y1 + 1.0)
        let bArea = (b.x2 - b.x1 + 1.0) * (b.y2 - b.y1 + 1.0)
        let denom = aArea + bArea - interArea
        guard denom > 0 else { return 0 }
        return interArea / denom
    }

    private static func nms(records: [BoxRecord], iouSame: Double, iouDiff: Double) -> [Int] {
        let sorted = records.enumerated().sorted { $0.element.score > $1.element.score }
        var indices = sorted.map(\.offset)
        var selected: [Int] = []
        selected.reserveCapacity(indices.count)

        while let current = indices.first {
            selected.append(current)
            indices.removeFirst()

            let currentBox = records[current]
            indices = indices.filter { idx in
                let other = records[idx]
                let threshold = (other.clsId == currentBox.clsId) ? iouSame : iouDiff
                return iou(currentBox, other) < threshold
            }
        }

        return selected
    }

    private static func filterLargeImages(
        records: [BoxRecord],
        id2label: [Int: String],
        imageSize: (width: Int, height: Int)
    ) -> [BoxRecord] {
        guard records.count > 1 else { return records }

        let width = Double(imageSize.width)
        let height = Double(imageSize.height)
        let areaThres = (width > height) ? 0.82 : 0.93

        let imageClassId = id2label.first(where: { $0.value == "image" })?.key
        guard let imageClassId else { return records }

        let imageArea = width * height

        return records.filter { record in
            guard record.clsId == imageClassId else { return true }
            let xmin = max(0.0, record.x1)
            let ymin = max(0.0, record.y1)
            let xmax = min(width, record.x2)
            let ymax = min(height, record.y2)
            let boxArea = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
            return boxArea <= areaThres * imageArea
        }
    }

    private static func mergeByContainment(
        records: [BoxRecord],
        id2label: [Int: String],
        mode: PPDocLayoutV3MergeMode
    ) -> [BoxRecord] {
        let preserveLabels: Set<String> = ["image", "seal", "chart"]
        let preserveClassIds = Set(id2label.compactMap { preserveLabels.contains($0.value) ? $0.key : nil })
        let (containsOther, containedByOther) = containmentFlags(records: records, preserveClassIds: preserveClassIds)

        switch mode {
        case .union:
            return records
        case .large:
            return zip(records, containedByOther).filter { !$0.1 }.map(\.0)
        case .small:
            return zip(zip(records, containsOther), containedByOther)
                .filter { (pair, contained) in
                    let (record, contains) = pair
                    _ = record
                    return (!contains) || contained
                }
                .map { ($0.0.0) }
        }
    }

    private static func mergeByContainment(
        records: [BoxRecord],
        id2label: [Int: String],
        modeByClass: [Int: PPDocLayoutV3MergeMode]
    ) -> [BoxRecord] {
        let preserveLabels: Set<String> = ["image", "seal", "chart"]
        let preserveClassIds = Set(id2label.compactMap { preserveLabels.contains($0.value) ? $0.key : nil })

        var keep = Array(repeating: true, count: records.count)
        for (categoryIndex, mode) in modeByClass {
            guard mode != .union else { continue }
            let (containsOther, containedByOther) = containmentFlags(
                records: records,
                preserveClassIds: preserveClassIds,
                categoryIndex: categoryIndex,
                mode: mode
            )
            for i in records.indices {
                switch mode {
                case .union:
                    break
                case .large:
                    keep[i] = keep[i] && !containedByOther[i]
                case .small:
                    keep[i] = keep[i] && (!containsOther[i] || containedByOther[i])
                }
            }
        }

        return zip(records, keep).filter(\.1).map(\.0)
    }

    private static func containmentFlags(
        records: [BoxRecord],
        preserveClassIds: Set<Int>,
        categoryIndex: Int? = nil,
        mode: PPDocLayoutV3MergeMode? = nil
    ) -> (containsOther: [Bool], containedByOther: [Bool]) {
        let n = records.count
        var containsOther = Array(repeating: false, count: n)
        var containedByOther = Array(repeating: false, count: n)

        for i in 0..<n {
            if preserveClassIds.contains(records[i].clsId) {
                continue
            }
            for j in 0..<n where i != j {
                if let categoryIndex, let mode {
                    switch mode {
                    case .large:
                        if records[j].clsId == categoryIndex, isContained(inner: records[i], outer: records[j]) {
                            containedByOther[i] = true
                            containsOther[j] = true
                        }
                    case .small:
                        if records[i].clsId == categoryIndex, isContained(inner: records[i], outer: records[j]) {
                            containedByOther[i] = true
                            containsOther[j] = true
                        }
                    case .union:
                        break
                    }
                } else {
                    if isContained(inner: records[i], outer: records[j]) {
                        containedByOther[i] = true
                        containsOther[j] = true
                    }
                }
            }
        }

        return (containsOther, containedByOther)
    }

    private static func isContained(inner: BoxRecord, outer: BoxRecord) -> Bool {
        let innerArea = max(0.0, (inner.x2 - inner.x1) * (inner.y2 - inner.y1))
        guard innerArea > 0 else { return false }

        let xi1 = max(inner.x1, outer.x1)
        let yi1 = max(inner.y1, outer.y1)
        let xi2 = min(inner.x2, outer.x2)
        let yi2 = min(inner.y2, outer.y2)
        let interWidth = max(0.0, xi2 - xi1)
        let interHeight = max(0.0, yi2 - yi1)
        let intersectArea = interWidth * interHeight
        let ratio = intersectArea / innerArea
        return ratio >= 0.8
    }

    private static func unclip(records: [BoxRecord], ratio: (width: Double, height: Double)) -> [BoxRecord] {
        guard ratio.width > 0, ratio.height > 0 else { return records }
        if ratio.width == 1.0, ratio.height == 1.0 {
            return records
        }
        return records.map { record in
            let width = record.x2 - record.x1
            let height = record.y2 - record.y1
            let newW = width * ratio.width
            let newH = height * ratio.height
            let centerX = record.x1 + width / 2.0
            let centerY = record.y1 + height / 2.0
            var expanded = record
            expanded.x1 = centerX - newW / 2.0
            expanded.y1 = centerY - newH / 2.0
            expanded.x2 = centerX + newW / 2.0
            expanded.y2 = centerY + newH / 2.0
            expanded.polygonPoints = nil
            expanded.mask = nil
            return expanded
        }
    }
}
