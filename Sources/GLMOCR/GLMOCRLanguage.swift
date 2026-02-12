import Foundation
import MLX

public enum GLMOCRLanguage {
    private enum ModalityType: Int8 {
        case text = 0
        case image = 1
        case video = 2
    }

    public static func getRopeIndex(
        inputTokenIds: [[Int]],
        attentionMask: [[Int]]? = nil,
        imageGridTHW: [(t: Int, h: Int, w: Int)]? = nil,
        videoGridTHW: [(t: Int, h: Int, w: Int)]? = nil,
        config: GLMOCRModelConfig
    ) -> (positionIds: MLXArray, ropeDeltas: MLXArray) {
        let batch = inputTokenIds.count
        precondition(batch > 0, "inputTokenIds must be non-empty (batch)")

        let seqLen = inputTokenIds[0].count
        precondition(seqLen > 0, "inputTokenIds must be non-empty (seqLen)")
        precondition(inputTokenIds.allSatisfy { $0.count == seqLen }, "inputTokenIds must be rectangular [B, L]")

        if let attentionMask {
            precondition(attentionMask.count == batch, "attentionMask batch mismatch")
            precondition(attentionMask.allSatisfy { $0.count == seqLen }, "attentionMask must be rectangular [B, L]")
            precondition(
                attentionMask.allSatisfy { row in row.allSatisfy { $0 == 0 || $0 == 1 } },
                "attentionMask values must be 0/1"
            )
        }

        let hasVision = (imageGridTHW != nil || videoGridTHW != nil)

        var positions = Array(repeating: Int32(1), count: 3 * batch * seqLen)
        var ropeDeltas = Array(repeating: Int32(0), count: batch)

        @inline(__always)
        func flatIndex(channel: Int, batchIdx: Int, tokenIdx: Int) -> Int {
            ((channel * batch) + batchIdx) * seqLen + tokenIdx
        }

        func setAllChannels(batchIdx: Int, tokenIdx: Int, value: Int32) {
            positions[flatIndex(channel: 0, batchIdx: batchIdx, tokenIdx: tokenIdx)] = value
            positions[flatIndex(channel: 1, batchIdx: batchIdx, tokenIdx: tokenIdx)] = value
            positions[flatIndex(channel: 2, batchIdx: batchIdx, tokenIdx: tokenIdx)] = value
        }

        if !hasVision {
            if let attentionMask {
                for b in 0..<batch {
                    var nextPos: Int32 = 0
                    for j in 0..<seqLen {
                        guard attentionMask[b][j] == 1 else { continue }
                        setAllChannels(batchIdx: b, tokenIdx: j, value: nextPos)
                        nextPos += 1
                    }

                    let maxPosition: Int32 = max(1, nextPos - 1)
                    ropeDeltas[b] = maxPosition + 1 - Int32(seqLen)
                }
            } else {
                for b in 0..<batch {
                    for j in 0..<seqLen {
                        setAllChannels(batchIdx: b, tokenIdx: j, value: Int32(j))
                    }
                }
            }

            return (
                positionIds: MLXArray(positions).reshaped(3, batch, seqLen),
                ropeDeltas: MLXArray(ropeDeltas).reshaped(batch, 1)
            )
        }

        let spatialMergeSize = config.visionConfig.spatialMergeSize
        let imageTokenId = config.imageTokenId
        let videoTokenId = config.videoTokenId
        let videoStartTokenId = config.videoStartTokenId
        let videoEndTokenId = config.videoEndTokenId

        var imageIndex = 0
        var videoIndex = 0
        var videoGroupIndex = 0

        for b in 0..<batch {
            let maskRow = attentionMask?[b] ?? Array(repeating: 1, count: seqLen)

            var unmaskedTokens: [Int] = []
            var unmaskedPositions: [Int] = []
            unmaskedTokens.reserveCapacity(seqLen)
            unmaskedPositions.reserveCapacity(seqLen)
            for j in 0..<seqLen where maskRow[j] == 1 {
                unmaskedTokens.append(inputTokenIds[b][j])
                unmaskedPositions.append(j)
            }

            var modalities: [ModalityType] = []
            modalities.reserveCapacity(unmaskedTokens.count)

            var inVideoRegion = false
            for token in unmaskedTokens {
                if token == videoStartTokenId {
                    inVideoRegion = true
                } else if token == videoEndTokenId {
                    inVideoRegion = false
                }

                if token == imageTokenId && !inVideoRegion {
                    modalities.append(.image)
                } else if (token == imageTokenId && inVideoRegion) || token == videoTokenId {
                    modalities.append(.video)
                } else {
                    modalities.append(.text)
                }
            }

            var groups: [(type: ModalityType, start: Int, end: Int)] = []
            groups.reserveCapacity(8)
            var st = 0
            while st < modalities.count {
                let t = modalities[st]
                var ed = st + 1
                while ed < modalities.count, modalities[ed] == t { ed += 1 }
                groups.append((type: t, start: st, end: ed))
                st = ed
            }

            var ch0: [Int32] = []
            var ch1: [Int32] = []
            var ch2: [Int32] = []
            ch0.reserveCapacity(unmaskedTokens.count)
            ch1.reserveCapacity(unmaskedTokens.count)
            ch2.reserveCapacity(unmaskedTokens.count)

            var currentBase: Int32 = 0
            var videoFrameNum = 1

            for group in groups {
                let groupLen = group.end - group.start
                let stIdx = currentBase

                switch group.type {
                case .text:
                    for i in 0..<groupLen {
                        let v = stIdx + Int32(i)
                        ch0.append(v)
                        ch1.append(v)
                        ch2.append(v)
                    }
                    currentBase += Int32(groupLen)
                    videoFrameNum = 1

                case .image:
                    guard let imageGridTHW else {
                        preconditionFailure("imageGridTHW is required when image tokens are present")
                    }
                    precondition(imageIndex < imageGridTHW.count, "imageGridTHW underflow (imageIndex=\(imageIndex))")
                    let grid = imageGridTHW[imageIndex]
                    imageIndex += 1
                    videoFrameNum = 1

                    let llmGridT = grid.t
                    let llmGridH = grid.h / spatialMergeSize
                    let llmGridW = grid.w / spatialMergeSize
                    precondition(llmGridT > 0 && llmGridH > 0 && llmGridW > 0, "Invalid image grid \(grid)")
                    let expected = llmGridT * llmGridH * llmGridW
                    precondition(
                        expected == groupLen,
                        "Image token count mismatch (expected=\(expected) got=\(groupLen))"
                    )

                    for ti in 0..<llmGridT {
                        let tVal = stIdx + Int32(ti)
                        for hi in 0..<llmGridH {
                            let hVal = stIdx + Int32(hi)
                            for wi in 0..<llmGridW {
                                ch0.append(tVal)
                                ch1.append(hVal)
                                ch2.append(stIdx + Int32(wi))
                            }
                        }
                    }

                    currentBase = stIdx + Int32(max(llmGridT, max(llmGridH, llmGridW)))

                case .video:
                    guard let videoGridTHW else {
                        preconditionFailure("videoGridTHW is required when video tokens are present")
                    }
                    precondition(videoIndex < videoGridTHW.count, "videoGridTHW underflow (videoIndex=\(videoIndex))")
                    let grid = videoGridTHW[videoIndex]

                    let llmGridT = videoFrameNum
                    let llmGridH = grid.h / spatialMergeSize
                    let llmGridW = grid.w / spatialMergeSize
                    precondition(llmGridT > 0 && llmGridH > 0 && llmGridW > 0, "Invalid video grid \(grid)")
                    let expected = llmGridT * llmGridH * llmGridW
                    precondition(
                        expected == groupLen,
                        "Video token count mismatch (expected=\(expected) got=\(groupLen) frame=\(videoFrameNum))"
                    )

                    for ti in 0..<llmGridT {
                        let tVal = stIdx + Int32(ti)
                        for hi in 0..<llmGridH {
                            let hVal = stIdx + Int32(hi)
                            for wi in 0..<llmGridW {
                                ch0.append(tVal)
                                ch1.append(hVal)
                                ch2.append(stIdx + Int32(wi))
                            }
                        }
                    }

                    currentBase = stIdx + Int32(max(llmGridT, max(llmGridH, llmGridW)))

                    videoGroupIndex += 1
                    if videoGroupIndex >= grid.t {
                        videoIndex += 1
                        videoGroupIndex = 0
                    }
                    videoFrameNum += 1
                }
            }

            precondition(
                ch0.count == unmaskedTokens.count && ch1.count == unmaskedTokens.count && ch2.count == unmaskedTokens.count,
                "Internal error: position length mismatch"
            )
            precondition(unmaskedPositions.count == unmaskedTokens.count, "Internal error: mask/token mismatch")

            for k in 0..<unmaskedPositions.count {
                let pos = unmaskedPositions[k]
                positions[flatIndex(channel: 0, batchIdx: b, tokenIdx: pos)] = ch0[k]
                positions[flatIndex(channel: 1, batchIdx: b, tokenIdx: pos)] = ch1[k]
                positions[flatIndex(channel: 2, batchIdx: b, tokenIdx: pos)] = ch2[k]
            }

            let maxPos = max(ch0.max() ?? 0, max(ch1.max() ?? 0, ch2.max() ?? 0))
            ropeDeltas[b] = maxPos + 1 - Int32(seqLen)
        }

        if let imageGridTHW {
            precondition(imageIndex == imageGridTHW.count, "imageGridTHW not fully consumed (used=\(imageIndex) total=\(imageGridTHW.count))")
        }
        if let videoGridTHW {
            precondition(videoGroupIndex == 0, "videoGridTHW mismatch (unfinished video group)")
            precondition(videoIndex == videoGridTHW.count, "videoGridTHW not fully consumed (used=\(videoIndex) total=\(videoGridTHW.count))")
        }

        return (
            positionIds: MLXArray(positions).reshaped(3, batch, seqLen),
            ropeDeltas: MLXArray(ropeDeltas).reshaped(batch, 1)
        )
    }

    public static func decodePositionIds(batch: Int, length: Int, cacheOffset: Int, ropeDeltas: MLXArray) -> MLXArray {
        var delta = MLXArray(Int32(cacheOffset)) + ropeDeltas.asType(.int32)
        if delta.ndim == 0 {
            delta = delta.expandedDimensions(axis: 0)
        }
        if delta.dim(0) < batch {
            delta = tiled(delta, repetitions: [batch / delta.dim(0)])
        } else if delta.dim(0) > batch {
            delta = delta[0 ..< batch]
        }

        if length == 1 {
            let pos = delta.reshaped(batch, 1)
            return tiled(pos[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }

        var base = MLXArray(0 ..< length).reshaped(1, length).asType(.int32)
        base = broadcast(base, to: [batch, length])
        let pos = base + delta.reshaped(batch, 1)
        return tiled(pos[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
    }
}
