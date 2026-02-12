import CoreGraphics
import XCTest
@testable import GLMOCR

final class GLMOCRMaskCropTests: XCTestCase {
    func testResizedMaskForCropUsesTopLeftYDownCoordinates() {
        let imageSize = (width: 8, height: 8)
        var maskData = [UInt8](repeating: 0, count: imageSize.width * imageSize.height)
        for y in 0..<4 {
            let row = y * imageSize.width
            for x in 0..<4 {
                maskData[row + x] = 1
            }
        }
        let mask = PPDocLayoutV3Mask(width: imageSize.width, height: imageSize.height, data: maskData)

        let topLeftCrop = GLMOCRDocumentParser.resizedMaskForCrop(
            mask: mask,
            cropSize: (width: 4, height: 4),
            inImageSize: imageSize,
            cropOriginPx: (x: 0, y: 0)
        )
        XCTAssertEqual(topLeftCrop, [UInt8](repeating: 1, count: 16))

        let bottomLeftCrop = GLMOCRDocumentParser.resizedMaskForCrop(
            mask: mask,
            cropSize: (width: 4, height: 4),
            inImageSize: imageSize,
            cropOriginPx: (x: 0, y: 4)
        )
        XCTAssertEqual(bottomLeftCrop, [UInt8](repeating: 0, count: 16))
    }

    func testApplyMaskCropWhitesBackgroundOutsideMask() {
        let width = 2
        let height = 2
        let cropMask: [UInt8] = [
            1, 0,
            0, 1,
        ]

        var rgba: [UInt8] = [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255,
        ]

        rgba.withUnsafeMutableBufferPointer { buffer in
            GLMOCRDocumentParser.applyMaskCrop(
                width: width,
                height: height,
                rgba: buffer,
                mask: cropMask
            )
        }

        XCTAssertEqual(Array(rgba[0..<4]), [10, 20, 30, 255])
        XCTAssertEqual(Array(rgba[4..<8]), [255, 255, 255, 255])
        XCTAssertEqual(Array(rgba[8..<12]), [255, 255, 255, 255])
        XCTAssertEqual(Array(rgba[12..<16]), [100, 110, 120, 255])
    }

    func testApplyMaskCropImagePreservesTopRowOrientation() throws {
        let width = 2
        let height = 2
        let rgba: [UInt8] = [
            10, 0, 0, 255,
            20, 0, 0, 255,
            0, 0, 30, 255,
            0, 0, 40, 255,
        ]
        let crop = try makeRGBAImage(width: width, height: height, rgba: rgba)

        let mask = PPDocLayoutV3Mask(
            width: width,
            height: height,
            data: [
                1, 1,
                0, 0,
            ]
        )

        guard let masked = GLMOCRDocumentParser.apply(
            mask: mask,
            toCrop: crop,
            inImageSize: (width: width, height: height),
            cropOriginPx: (x: 0, y: 0)
        ) else {
            XCTFail("Expected masked image")
            return
        }

        let out = try decodeRGBA(masked)
        XCTAssertEqual(Array(out[0..<4]), [10, 0, 0, 255])
        XCTAssertEqual(Array(out[4..<8]), [20, 0, 0, 255])
        XCTAssertEqual(Array(out[8..<12]), [255, 255, 255, 255])
        XCTAssertEqual(Array(out[12..<16]), [255, 255, 255, 255])
    }

    private func makeRGBAImage(width: Int, height: Int, rgba: [UInt8]) throws -> CGImage {
        XCTAssertEqual(rgba.count, width * height * 4)
        let bytesPerRow = width * 4
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else {
            XCTFail("Unable to create data provider")
            throw NSError(domain: "GLMOCRMaskCropTests", code: 1)
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            XCTFail("Unable to create color space")
            throw NSError(domain: "GLMOCRMaskCropTests", code: 2)
        }

        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let image = CGImage(
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
        ) else {
            XCTFail("Unable to create image")
            throw NSError(domain: "GLMOCRMaskCropTests", code: 3)
        }
        return image
    }

    private func decodeRGBA(_ image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        let bytesPerRow = width * 4
        var data = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            XCTFail("Unable to create color space")
            throw NSError(domain: "GLMOCRMaskCropTests", code: 4)
        }

        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            XCTFail("Unable to create context")
            throw NSError(domain: "GLMOCRMaskCropTests", code: 5)
        }

        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }
}
