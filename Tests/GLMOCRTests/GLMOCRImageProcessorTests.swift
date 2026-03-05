import CoreImage
import XCTest

@testable import GLMOCR

final class GLMOCRImageProcessorTests: XCTestCase {
    func testProcessPatchifyShape() throws {
        let config = GLMOCRPreprocessorConfig(
            size: .init(shortestEdge: 112 * 112, longestEdge: 28 * 28 * 15000),
            doRescale: true,
            patchSize: 14,
            temporalPatchSize: 2,
            mergeSize: 2,
            imageMean: [0.48145466, 0.4578275, 0.40821073],
            imageStd: [0.26862954, 0.26130258, 0.27577711],
            imageProcessorType: nil,
            processorClass: nil
        )
        let processor = GLMOCRImageProcessor(config: config)

        let image = CIImage(color: CIColor(red: 1, green: 1, blue: 1))
            .cropped(to: CGRect(x: 0, y: 0, width: 112, height: 112))

        let processed = try processor.process(image)
        XCTAssertEqual(processed.imageGridTHW.t, 1)
        XCTAssertEqual(processed.imageGridTHW.h, 8)
        XCTAssertEqual(processed.imageGridTHW.w, 8)
        XCTAssertEqual(processed.pixelValues.shape, [64, 1176])
    }

    func testProcessBatchMatchesSingleProcessForCGImages() throws {
        let config = GLMOCRPreprocessorConfig(
            size: .init(shortestEdge: 112 * 112, longestEdge: 28 * 28 * 15000),
            doRescale: true,
            patchSize: 14,
            temporalPatchSize: 2,
            mergeSize: 2,
            imageMean: [0.48145466, 0.4578275, 0.40821073],
            imageStd: [0.26862954, 0.26130258, 0.27577711],
            imageProcessorType: nil,
            processorClass: nil
        )
        let processor = GLMOCRImageProcessor(config: config)

        let image1 = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 120, height: 96))
        let image2 = CIImage(color: CIColor(red: 0, green: 1, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 96, height: 120))
        let context = CIContext()

        guard let cgImage1 = context.createCGImage(image1, from: image1.extent),
            let cgImage2 = context.createCGImage(image2, from: image2.extent)
        else {
            XCTFail("Failed to build test CGImages")
            return
        }

        let single0 = try processor.process(cgImage1)
        let single1 = try processor.process(cgImage2)
        let batch = try processor.process([cgImage1, cgImage2])

        XCTAssertEqual(batch.count, 2)
        let singles = [single0, single1]
        for idx in 0..<batch.count {
            XCTAssertEqual(batch[idx].imageGridTHW.t, singles[idx].imageGridTHW.t)
            XCTAssertEqual(batch[idx].imageGridTHW.h, singles[idx].imageGridTHW.h)
            XCTAssertEqual(batch[idx].imageGridTHW.w, singles[idx].imageGridTHW.w)
            XCTAssertEqual(batch[idx].resizedHeight, singles[idx].resizedHeight)
            XCTAssertEqual(batch[idx].resizedWidth, singles[idx].resizedWidth)
            XCTAssertEqual(batch[idx].pixelValues.shape, singles[idx].pixelValues.shape)
            XCTAssertEqual(batch[idx].pixelValues.asArray(Float32.self), singles[idx].pixelValues.asArray(Float32.self))
        }
    }
}
