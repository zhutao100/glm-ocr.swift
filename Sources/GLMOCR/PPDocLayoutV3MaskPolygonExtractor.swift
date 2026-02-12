import Foundation

struct PPDocLayoutV3MaskPolygonExtractor {
    private struct IntPoint: Hashable {
        var x: Int
        var y: Int
    }

    private struct IntDirection: Equatable {
        var dx: Int
        var dy: Int
    }

    private struct ContourState: Hashable {
        var point: IntPoint
        var backtrack: IntPoint
    }

    private struct DPoint: Equatable {
        var x: Double
        var y: Double
    }

    static func extractPolygonPoints(
        boxPx: (x1: Int, y1: Int, x2: Int, y2: Int),
        mask: PPDocLayoutV3Mask,
        imageSize: (width: Int, height: Int),
        epsilonRatio: Double = 0.004
    ) -> [[Double]]? {
        let boxW = boxPx.x2 - boxPx.x1
        let boxH = boxPx.y2 - boxPx.y1
        guard boxW > 0, boxH > 0 else { return nil }
        guard mask.width > 0, mask.height > 0 else { return nil }
        guard mask.data.count == mask.width * mask.height else { return nil }

        func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(v, hi)) }
        func roundToEvenInt(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

        let scaleW = Double(mask.width) / Double(max(imageSize.width, 1))
        let scaleH = Double(mask.height) / Double(max(imageSize.height, 1))

        let xStart = clamp(roundToEvenInt(Double(boxPx.x1) * scaleW), 0, mask.width)
        let xEnd = clamp(roundToEvenInt(Double(boxPx.x2) * scaleW), 0, mask.width)
        let yStart = clamp(roundToEvenInt(Double(boxPx.y1) * scaleH), 0, mask.height)
        let yEnd = clamp(roundToEvenInt(Double(boxPx.y2) * scaleH), 0, mask.height)

        let xs = min(xStart, xEnd)
        let xe = max(xStart, xEnd)
        let ys = min(yStart, yEnd)
        let ye = max(yStart, yEnd)

        let cropMaskW = xe - xs
        let cropMaskH = ye - ys
        guard cropMaskW > 0, cropMaskH > 0 else { return nil }

        var cropped = [UInt8](repeating: 0, count: cropMaskW * cropMaskH)
        for y in 0..<cropMaskH {
            let srcRow = (ys + y) * mask.width + xs
            let dstRow = y * cropMaskW
            for x in 0..<cropMaskW {
                cropped[dstRow + x] = mask.data[srcRow + x]
            }
        }

        
        var resized = [UInt8](repeating: 0, count: boxW * boxH)
        for y in 0..<boxH {
            let srcY = Int(Double(y) * Double(cropMaskH) / Double(boxH))
            let srcRow = min(max(srcY, 0), cropMaskH - 1) * cropMaskW
            let dstRow = y * boxW
            for x in 0..<boxW {
                let srcX = Int(Double(x) * Double(cropMaskW) / Double(boxW))
                let v = cropped[srcRow + min(max(srcX, 0), cropMaskW - 1)]
                resized[dstRow + x] = v
            }
        }

        guard let start = largestComponentBoundaryStart(mask: resized, width: boxW, height: boxH) else {
            return nil
        }
        let rawContour = traceContour(mask: resized, width: boxW, height: boxH, start: start)
        guard !rawContour.isEmpty else { return nil }

        let contour = chainApproxSimple(rotateContourStartOpenCVLike(ensureOpenCVExternalContourOrientation(rawContour)))
        guard !contour.isEmpty else { return nil }

        let arc = arcLength(contour, closed: true)
        let epsilon = epsilonRatio * arc
        let simplified = approxPolyDPClosed(contour, epsilon: epsilon)
        let polygon = extractCustomVertices(simplified)

        return polygon.map { pt in
            [pt.x + Double(boxPx.x1), pt.y + Double(boxPx.y1)]
        }
    }

    private static func largestComponentBoundaryStart(mask: [UInt8], width: Int, height: Int) -> IntPoint? {
        guard width > 0, height > 0 else { return nil }
        guard mask.count == width * height else { return nil }

        func idx(_ x: Int, _ y: Int) -> Int { y * width + x }
        func isInside(_ x: Int, _ y: Int) -> Bool { x >= 0 && x < width && y >= 0 && y < height }
        func isBoundary(_ x: Int, _ y: Int) -> Bool {
            guard mask[idx(x, y)] != 0 else { return false }
            let neighbors4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            for (dx, dy) in neighbors4 {
                let nx = x + dx
                let ny = y + dy
                if !isInside(nx, ny) { return true }
                if mask[idx(nx, ny)] == 0 { return true }
            }
            return false
        }

        var visited = [UInt8](repeating: 0, count: width * height)
        var bestArea = 0
        var bestStart: IntPoint?

        let neighbors8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

        for y in 0..<height {
            for x in 0..<width {
                let startIdx = idx(x, y)
                guard mask[startIdx] != 0, visited[startIdx] == 0 else { continue }

                var queue: [Int] = [startIdx]
                visited[startIdx] = 1
                var head = 0
                var area = 0
                var componentBoundary: IntPoint?

                while head < queue.count {
                    let cur = queue[head]
                    head += 1
                    area += 1
                    let cx = cur % width
                    let cy = cur / width

                    if isBoundary(cx, cy) {
                        let p = IntPoint(x: cx, y: cy)
                        if let existing = componentBoundary {
                            if p.y < existing.y || (p.y == existing.y && p.x < existing.x) {
                                componentBoundary = p
                            }
                        } else {
                            componentBoundary = p
                        }
                    }

                    for (dx, dy) in neighbors8 {
                        let nx = cx + dx
                        let ny = cy + dy
                        guard isInside(nx, ny) else { continue }
                        let n = idx(nx, ny)
                        guard mask[n] != 0, visited[n] == 0 else { continue }
                        visited[n] = 1
                        queue.append(n)
                    }
                }

                if area > bestArea, let componentBoundary {
                    bestArea = area
                    bestStart = componentBoundary
                }
            }
        }

        return bestStart
    }

    private static func traceContour(mask: [UInt8], width: Int, height: Int, start: IntPoint) -> [DPoint] {
        func isInside(_ x: Int, _ y: Int) -> Bool { x >= 0 && x < width && y >= 0 && y < height }
        func at(_ x: Int, _ y: Int) -> UInt8 {
            guard isInside(x, y) else { return 0 }
            return mask[y * width + x]
        }

        let dirs: [(dx: Int, dy: Int)] = [
            (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
        ]

        var contour: [DPoint] = []
        contour.reserveCapacity(256)

        var p = start
        var b = IntPoint(x: start.x - 1, y: start.y)
        let b0 = b
        contour.append(DPoint(x: Double(p.x), y: Double(p.y)))

        let maxSteps = max(64, width * height * 2)
        var steps = 0
        var visitedStates: Set<ContourState> = [ContourState(point: p, backtrack: b)]

        while steps < maxSteps {
            steps += 1

            let dx = b.x - p.x
            let dy = b.y - p.y
            let bIndex: Int = {
                for (i, d) in dirs.enumerated() where d.dx == dx && d.dy == dy { return i }
                return 4 
            }()

            var found: (next: IntPoint, newBacktrack: IntPoint)?
            for i in 1...8 {
                let k = (bIndex + i) % 8
                let nx = p.x + dirs[k].dx
                let ny = p.y + dirs[k].dy
                if at(nx, ny) != 0 {
                    let prevDir = (k + 7) % 8
                    let nb = IntPoint(x: p.x + dirs[prevDir].dx, y: p.y + dirs[prevDir].dy)
                    found = (IntPoint(x: nx, y: ny), nb)
                    break
                }
            }

            guard let found else { break }
            p = found.next
            b = found.newBacktrack
            contour.append(DPoint(x: Double(p.x), y: Double(p.y)))

            if p == start && b == b0 {
                break
            }

            let state = ContourState(point: p, backtrack: b)
            if !visitedStates.insert(state).inserted {
                break
            }
        }

        
        if contour.count >= 2, contour.first == contour.last {
            contour.removeLast()
        }
        return contour
    }

    private static func chainApproxSimple(_ contour: [DPoint]) -> [DPoint] {
        let deduped = removeConsecutiveDuplicates(contour)
        guard deduped.count >= 2 else { return deduped }

        if deduped.count == 2 {
            return deduped
        }

        var output: [DPoint] = [deduped[0]]
        output.reserveCapacity(deduped.count)

        var previousDirection = direction(from: deduped[0], to: deduped[1])
        for i in 1..<deduped.count {
            let current = deduped[i]
            let next = deduped[(i + 1) % deduped.count]
            let currentDirection = direction(from: current, to: next)
            if currentDirection != previousDirection {
                output.append(current)
            }
            previousDirection = currentDirection
        }

        return removeConsecutiveDuplicates(output)
    }

    private static func removeConsecutiveDuplicates(_ points: [DPoint]) -> [DPoint] {
        guard !points.isEmpty else { return [] }
        var out: [DPoint] = []
        out.reserveCapacity(points.count)
        for p in points {
            if let last = out.last, last == p { continue }
            out.append(p)
        }
        if out.count >= 2, out.first == out.last {
            out.removeLast()
        }
        return out
    }

    private static func direction(from a: DPoint, to b: DPoint) -> IntDirection {
        IntDirection(
            dx: signum(b.x - a.x),
            dy: signum(b.y - a.y)
        )
    }

    private static func signum(_ value: Double) -> Int {
        if value > 0 { return 1 }
        if value < 0 { return -1 }
        return 0
    }

    private static func ensureOpenCVExternalContourOrientation(_ points: [DPoint]) -> [DPoint] {
        guard points.count >= 3 else { return points }
        
        
        if signedArea(points) > 0 {
            return Array(points.reversed())
        }
        return points
    }

    private static func rotateContourStartOpenCVLike(_ points: [DPoint]) -> [DPoint] {
        guard points.count >= 2 else { return points }
        var bestIndex = 0
        var best = points[0]
        for i in 1..<points.count {
            let p = points[i]
            if p.y < best.y || (p.y == best.y && p.x < best.x) {
                bestIndex = i
                best = p
            }
        }
        guard bestIndex != 0 else { return points }
        return Array(points[bestIndex...] + points[..<bestIndex])
    }

    private static func signedArea(_ points: [DPoint]) -> Double {
        guard points.count >= 3 else { return 0 }
        var area = 0.0
        for i in 0..<points.count {
            let j = (i + 1) % points.count
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        }
        return area * 0.5
    }

    private static func approxPolyDPClosed(_ points: [DPoint], epsilon: Double) -> [DPoint] {
        return approxPolyDPClosedOpenCV(points, epsilon: max(0, epsilon))
    }

    
    private static func approxPolyDPClosedOpenCV(_ points: [DPoint], epsilon: Double) -> [DPoint] {
        let count0 = points.count
        guard count0 > 0 else { return [] }
        if count0 <= 2 { return points }

        struct Slice {
            var start: Int
            var end: Int
        }

        var eps = epsilon * epsilon
        var stack: [Slice] = []
        stack.reserveCapacity(count0)
        var dst: [DPoint] = []
        dst.reserveCapacity(count0)

        var slice = Slice(start: 0, end: 0)
        var rightSlice = Slice(start: 0, end: 0)

        var pos = 0
        var initIters = 3
        var leEps = false
        var startPt = points[0]
        var endPt = points[0]
        var pt = points[0]

        func readSrc(_ posIn: inout Int) -> DPoint {
            let out = points[posIn]
            posIn += 1
            if posIn >= count0 { posIn = 0 }
            return out
        }

        func readDst(_ posIn: inout Int, count: Int) -> DPoint {
            let out = dst[posIn]
            posIn += 1
            if posIn >= count { posIn = 0 }
            return out
        }

        
        rightSlice.start = 0
        for _ in 0..<initIters {
            var maxDist = 0.0
            pos = (pos + rightSlice.start) % count0
            startPt = readSrc(&pos)
            for j in 1..<count0 {
                pt = readSrc(&pos)
                let dx = pt.x - startPt.x
                let dy = pt.y - startPt.y
                let dist = dx * dx + dy * dy
                if dist > maxDist {
                    maxDist = dist
                    rightSlice.start = j
                }
            }
            leEps = maxDist <= eps
        }

        
        if !leEps {
            let start = pos % count0
            slice.start = start
            rightSlice.end = start
            let newStart = (rightSlice.start + slice.start) % count0
            rightSlice.start = newStart
            slice.end = newStart
            stack.append(rightSlice)
            stack.append(slice)
        } else {
            dst.append(startPt)
        }

        
        while let current = stack.popLast() {
            slice = current
            endPt = points[slice.end]
            pos = slice.start
            startPt = readSrc(&pos)

            if pos != slice.end {
                let dx = endPt.x - startPt.x
                let dy = endPt.y - startPt.y
                let segmentLen2 = dx * dx + dy * dy
                if segmentLen2 > 0 {
                    var maxDist2MulSegmentLen2 = 0.0
                    while pos != slice.end {
                        pt = readSrc(&pos)
                        let px = pt.x - startPt.x
                        let py = pt.y - startPt.y
                        let projection = px * dx + py * dy
                        let dist2MulSegmentLen2: Double
                        if projection < 0 {
                            dist2MulSegmentLen2 = (px * px + py * py) * segmentLen2
                        } else if projection > segmentLen2 {
                            let ex = pt.x - endPt.x
                            let ey = pt.y - endPt.y
                            dist2MulSegmentLen2 = (ex * ex + ey * ey) * segmentLen2
                        } else {
                            let dist = py * dx - px * dy
                            dist2MulSegmentLen2 = dist * dist
                        }
                        if dist2MulSegmentLen2 > maxDist2MulSegmentLen2 {
                            maxDist2MulSegmentLen2 = dist2MulSegmentLen2
                            rightSlice.start = (pos + count0 - 1) % count0
                        }
                    }
                    leEps = maxDist2MulSegmentLen2 <= eps * segmentLen2
                } else {
                    leEps = true
                }
            } else {
                leEps = true
                startPt = points[slice.start]
            }

            if leEps {
                dst.append(startPt)
            } else {
                rightSlice.end = slice.end
                slice.end = rightSlice.start
                stack.append(rightSlice)
                stack.append(slice)
            }
        }

        let count = dst.count
        if count <= 2 { return dst }

        var posDst = count - 1
        startPt = readDst(&posDst, count: count)
        var wpos = posDst
        pt = readDst(&posDst, count: count)

        var newCount = count
        var i = 0
        while i < count && newCount > 2 {
            endPt = readDst(&posDst, count: count)
            let dx = endPt.x - startPt.x
            let dy = endPt.y - startPt.y
            let dist = abs((pt.x - startPt.x) * dy - (pt.y - startPt.y) * dx)
            let successiveInnerProduct =
                (pt.x - startPt.x) * (endPt.x - pt.x) +
                (pt.y - startPt.y) * (endPt.y - pt.y)

            if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0 && dy != 0 && successiveInnerProduct >= 0 {
                newCount -= 1
                dst[wpos] = endPt
                startPt = endPt
                wpos += 1
                if wpos >= count { wpos = 0 }
                pt = readDst(&posDst, count: count)
                i += 1
                i += 1
                continue
            }

            dst[wpos] = pt
            startPt = pt
            wpos += 1
            if wpos >= count { wpos = 0 }
            pt = endPt
            i += 1
        }

        if newCount < count {
            dst = Array(dst[0..<newCount])
        }
        return dst
    }

    private static func arcLength(_ points: [DPoint], closed: Bool) -> Double {
        guard points.count >= 2 else { return 0 }
        var sum = 0.0
        for i in 1..<points.count {
            let dx = points[i].x - points[i - 1].x
            let dy = points[i].y - points[i - 1].y
            sum += (dx * dx + dy * dy).squareRoot()
        }
        if closed, let first = points.first, let last = points.last {
            let dx = first.x - last.x
            let dy = first.y - last.y
            sum += (dx * dx + dy * dy).squareRoot()
        }
        return sum
    }

    private static func rdpClosed(_ points: [DPoint], epsilon: Double) -> [DPoint] {
        guard points.count >= 3 else { return points }
        if points.count == 3 { return points }

        
        let first = points[0]
        var splitIndex = 0
        var best = -Double.infinity
        for i in 1..<points.count {
            let dx = points[i].x - first.x
            let dy = points[i].y - first.y
            let d2 = dx * dx + dy * dy
            if d2 > best {
                best = d2
                splitIndex = i
            }
        }
        if splitIndex == 0 || splitIndex == points.count - 1 {
            splitIndex = points.count / 2
        }

        let seq1 = Array(points[0...splitIndex])
        var seq2 = Array(points[splitIndex...])
        seq2.append(first)

        let s1 = rdpOpen(seq1, epsilon: epsilon)
        let s2 = rdpOpen(seq2, epsilon: epsilon)

        var out: [DPoint] = []
        out.reserveCapacity(s1.count + s2.count)
        if s1.count >= 2 {
            out.append(contentsOf: s1.dropLast())
        }
        if s2.count >= 2 {
            out.append(contentsOf: s2.dropLast())
        }

        
        var deduped: [DPoint] = []
        deduped.reserveCapacity(out.count)
        for p in out {
            if let last = deduped.last, last == p { continue }
            deduped.append(p)
        }
        return deduped
    }

    private static func rdpOpen(_ points: [DPoint], epsilon: Double) -> [DPoint] {
        guard points.count >= 3 else { return points }
        let start = points[0]
        let end = points[points.count - 1]

        var index = -1
        var maxDist = 0.0
        for i in 1..<(points.count - 1) {
            let d = perpendicularDistance(points[i], start, end)
            if d > maxDist {
                maxDist = d
                index = i
            }
        }

        if maxDist > epsilon, index >= 0 {
            let left = rdpOpen(Array(points[0...index]), epsilon: epsilon)
            let right = rdpOpen(Array(points[index...]), epsilon: epsilon)
            if left.isEmpty { return right }
            if right.isEmpty { return left }
            return left.dropLast() + right
        }

        return [start, end]
    }

    private static func perpendicularDistance(_ p: DPoint, _ a: DPoint, _ b: DPoint) -> Double {
        let vx = b.x - a.x
        let vy = b.y - a.y
        let wx = p.x - a.x
        let wy = p.y - a.y

        let denom = vx * vx + vy * vy
        if denom == 0 {
            return (wx * wx + wy * wy).squareRoot()
        }
        let t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        let projX = a.x + t * vx
        let projY = a.y + t * vy
        let dx = p.x - projX
        let dy = p.y - projY
        return (dx * dx + dy * dy).squareRoot()
    }

    private static func extractCustomVertices(_ polygon: [DPoint], sharpAngleThresh: Double = 45.0) -> [DPoint] {
        let n = polygon.count
        guard n > 0 else { return [] }

        var result: [DPoint] = []
        result.reserveCapacity(n)

        for i in 0..<n {
            let previousPoint = polygon[(i - 1 + n) % n]
            let currentPoint = polygon[i]
            let nextPoint = polygon[(i + 1) % n]

            let vector1x = previousPoint.x - currentPoint.x
            let vector1y = previousPoint.y - currentPoint.y
            let vector2x = nextPoint.x - currentPoint.x
            let vector2y = nextPoint.y - currentPoint.y

            let crossProductValue = (vector1y * vector2x) - (vector1x * vector2y)
            if crossProductValue < 0 {
                let norm1 = (vector1x * vector1x + vector1y * vector1y).squareRoot()
                let norm2 = (vector2x * vector2x + vector2y * vector2y).squareRoot()
                let denom = norm1 * norm2

                var angle = Double.nan
                if denom > 0 {
                    let dot = vector1x * vector2x + vector1y * vector2y
                    let angleCos = max(-1.0, min(1.0, dot / denom))
                    angle = acos(angleCos) * 180.0 / .pi
                }

                if angle.isFinite, abs(angle - sharpAngleThresh) < 1.0, norm1 > 0, norm2 > 0 {
                    var directionX = vector1x / norm1 + vector2x / norm2
                    var directionY = vector1y / norm1 + vector2y / norm2
                    let directionNorm = (directionX * directionX + directionY * directionY).squareRoot()

                    if directionNorm > 0 {
                        directionX /= directionNorm
                        directionY /= directionNorm
                        let stepSize = (norm1 + norm2) / 2.0
                        let newPoint = DPoint(
                            x: currentPoint.x + directionX * stepSize,
                            y: currentPoint.y + directionY * stepSize
                        )
                        result.append(newPoint)
                    } else {
                        result.append(currentPoint)
                    }
                } else {
                    result.append(currentPoint)
                }
            }
        }

        return result
    }
}
