import ArgumentParser
import CoreGraphics
import Foundation
import GLMOCR
import Hub
import ImageIO
import MLX

@main
struct GLMOCRCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "glm-ocr",
        abstract: "GLM-OCR (zai-org/GLM-OCR) port to Swift/MLX.",
        subcommands: [OCR.self, Parse.self],
        defaultSubcommand: OCR.self
    )

    enum PromptPreset: String, CaseIterable, ExpressibleByArgument {
        case `default` = "default"
        case recognition = "recognition"

        init?(argument: String) {
            switch argument.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case Self.default.rawValue:
                self = .default
            case Self.recognition.rawValue, "text-recognition":
                self = .recognition
            default:
                return nil
            }
        }

        var promptText: String {
            switch self {
            case .default:
                return GLMOCRPromptPresets.default
            case .recognition:
                return GLMOCRPromptPresets.textRecognition
            }
        }
    }

    struct ModelOptions: ParsableArguments {
        @Option(help: "Hugging Face model id to download/use, or a local model directory (skips download).")
        var model: String = "zai-org/GLM-OCR"

        @Option(help: "Hugging Face revision (branch/tag/commit) (remote models only).")
        var revision: String = "main"

        @Option(help: "Hub download base directory (defaults to Hugging Face hub cache).")
        var downloadBase: String?

        @Option(help: "Override model weights dtype (auto|float32|float16|bfloat16). (auto defaults to float16)")
        var dtype: String = "auto"

        @Option(
            name: .customLong("cache-limit"),
            help: "GPU memory cache limit in MB (default: 2048; 0 disables caching)."
        )
        var cacheLimitMB: Int = 2048
    }

    struct DecodingOptions: ParsableArguments {
        @Option(help: "Max new tokens for generation.")
        var maxNewTokens: Int = 4096

        @Option(help: "Temperature (0 = greedy).")
        var temperature: Float?

        @Option(name: [.customLong("top-p")], help: "Override generation top_p.")
        var topP: Float?

        @Option(name: [.customLong("top-k")], help: "Override generation top_k.")
        var topK: Int?

        @Option(name: [.customLong("repetition-penalty")], help: "Override generation repetition_penalty.")
        var repetitionPenalty: Float?

        @Option(help: "Random seed for sampling (only used when temperature > 0).")
        var seed: UInt64?

        func generationParameters(defaultPreset: GLMOCRGenerationParameters) -> GLMOCRGenerationParameters {
            var parameters = defaultPreset
            if let temperature {
                parameters.temperature = temperature
            }
            if let topP {
                parameters.topP = topP
            }
            if let topK {
                parameters.topK = topK
            }
            if let repetitionPenalty {
                parameters.repetitionPenalty = repetitionPenalty
            }
            if let seed {
                parameters.seed = seed
            }
            return parameters
        }
    }

    struct OCR: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "ocr",
            abstract: "Run OCR on one image or a batch of images."
        )

        @OptionGroup var modelOptions: ModelOptions
        @OptionGroup var decoding: DecodingOptions

        @Option(
            name: [.customLong("input")],
            parsing: .upToNextOption,
            help: "Path(s) to image file(s) and/or directory(ies). Repeat or pass multiple values."
        )
        var inputs: [String] = []

        @Option(help: "Prompt preset (default|recognition). Ignored if --prompt is provided.")
        var promptPreset: PromptPreset = .recognition

        @Option(help: "Prompt text override.")
        var prompt: String?

        @Option(name: [.customLong("output-dir")], help: "OCR output directory for batch output.")
        var outputDir: String = "outputs/ocr"

        @Option(
            name: [.customLong("post-resize-jpeg-quality")],
            help: "JPEG quality (0...1) used when re-encoding images after smart resize (default: disabled)."
        )
        var postResizeJpegQuality: Double?

        @Option(name: [.customLong("batch-size")], help: "Maximum number of images per processing batch (0 = all).")
        var batchSize: Int = 0

        @Flag(
            name: [.customLong("skip-special-tokens")],
            inversion: .prefixedNo,
            help: "Skip tokenizer special tokens in OCR output."
        )
        var skipSpecialTokens: Bool = true

        mutating func run() async throws {
            func eprintln(_ message: String) {
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }

            if batchSize < 0 {
                throw ValidationError("--batch-size must be >= 0.")
            }
            if let postResizeJpegQuality, postResizeJpegQuality < 0.0 || postResizeJpegQuality > 1.0 {
                throw ValidationError("--post-resize-jpeg-quality must be between 0 and 1.")
            }
            try await GLMOCRCLI.withExecutionDevice {
                let resolvedURLs = try GLMOCRCLI.resolveImageURLs(inputs: inputs)
                let imageURLs = resolvedURLs.filter { $0.pathExtension.lowercased() != "pdf" }
                if imageURLs.count != resolvedURLs.count {
                    eprintln("Skipping PDF inputs for ocr (use parse for PDFs).")
                }
                if imageURLs.isEmpty {
                    throw ValidationError("No supported image inputs found.")
                }
                let pipeline = try await GLMOCRCLI.makeOCRPipeline(modelOptions: modelOptions, eprintln: eprintln)

                let dtypeOverride = try parseDTypeOverride(modelOptions.dtype)
                let generationParameters = decoding.generationParameters(defaultPreset: .greedy)
                let resolvedPrompt = GLMOCRCLI.resolvePrompt(override: prompt, preset: promptPreset)

                if imageURLs.count == 1, batchSize == 0 {
                    let outputs = try pipeline.recognizeBatch(
                        imagePaths: [imageURLs[0].path],
                        prompt: resolvedPrompt,
                        maxNewTokens: decoding.maxNewTokens,
                        generationParameters: generationParameters,
                        skipSpecialTokens: skipSpecialTokens,
                        dtypeOverride: dtypeOverride,
                        postResizeJPEGRoundTripQuality: postResizeJpegQuality
                    )
                    precondition(outputs.count == 1)
                    print(outputs[0])
                    return
                }

                try GLMOCRCLI.runBatchOCR(
                    pipeline: pipeline,
                    imageURLs: imageURLs,
                    prompt: resolvedPrompt,
                    maxNewTokens: decoding.maxNewTokens,
                    generationParameters: generationParameters,
                    dtypeOverride: dtypeOverride,
                    skipSpecialTokens: skipSpecialTokens,
                    outputDir: outputDir,
                    postResizeJpegQuality: postResizeJpegQuality,
                    batchSize: batchSize,
                    eprintln: eprintln
                )
            }
        }
    }

    struct Parse: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "parse",
            abstract: "Run layout + OCR parse and output default markdown/json results."
        )

        @OptionGroup var modelOptions: ModelOptions
        @OptionGroup var decoding: DecodingOptions

        @Option(
            name: [.customLong("input")],
            parsing: .upToNextOption,
            help: "Path(s) to image file(s) and/or directory(ies). Repeat or pass multiple values."
        )
        var inputs: [String] = []

        @Option(
            name: [.customLong("output-dir")], help: "Parse output directory containing <stem>/{result.md,result.json}."
        )
        var outputDir: String = "outputs/parse"

        @Option(
            name: [.customLong("jpeg-quality")],
            help: "JPEG quality (0...1) used when re-encoding region crops (default: 0.95)."
        )
        var jpegQuality: Double = 0.95

        @Option(name: [.customLong("batch-size")], help: "Maximum number of images per processing batch (0 = all).")
        var batchSize: Int = 0

        @Option(
            name: [.customLong("pdf-max-pages")],
            help: "For PDF inputs, render at most this many pages. Omit for all pages."
        )
        var pdfMaxPages: Int?

        @Option(
            name: [.customLong("max-new-tokens-text")],
            help: "Per-region max new tokens for text OCR (default: min(--max-new-tokens, 2048))."
        )
        var maxNewTokensText: Int?

        @Option(
            name: [.customLong("max-new-tokens-table")],
            help: "Per-region max new tokens for table OCR (default: --max-new-tokens)."
        )
        var maxNewTokensTable: Int?

        @Option(
            name: [.customLong("max-new-tokens-formula")],
            help: "Per-region max new tokens for formula OCR (default: min(--max-new-tokens, 1024))."
        )
        var maxNewTokensFormula: Int?

        @Flag(
            name: [.customLong("include-formula-numbers")],
            help: "OCR formula number regions and merge them into formulas.")
        var includeFormulaNumbers: Bool = false

        @Option(
            name: [.customLong("layout-threshold")],
            help: "Layout confidence threshold (default: 0.4)."
        )
        var layoutThreshold: Double = 0.4

        @Option(
            name: [.customLong("layout-threshold-class-1")],
            help: "Layout confidence threshold override for class 1 (default: 0.10)."
        )
        var layoutThresholdClass1: Double = 0.10

        @Option(
            name: [.customLong("layout-threshold-class-7")],
            help: "Layout confidence threshold override for class 7 (default: 0.30)."
        )
        var layoutThresholdClass7: Double = 0.30

        @Option(
            name: [.customLong("layout-threshold-class-14")],
            help: "Layout confidence threshold override for class 14 (default: 0.30)."
        )
        var layoutThresholdClass14: Double = 0.30

        mutating func run() async throws {
            func eprintln(_ message: String) {
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }

            func validateUnitInterval(_ value: Double, optionName: String) throws {
                if value < 0.0 || value > 1.0 {
                    throw ValidationError("\(optionName) must be between 0 and 1.")
                }
            }

            if batchSize < 0 {
                throw ValidationError("--batch-size must be >= 0.")
            }
            if decoding.maxNewTokens <= 0 {
                throw ValidationError("--max-new-tokens must be > 0.")
            }
            if let pdfMaxPages, pdfMaxPages <= 0 {
                throw ValidationError("--pdf-max-pages must be > 0 when provided.")
            }
            if let maxNewTokensText, maxNewTokensText <= 0 {
                throw ValidationError("--max-new-tokens-text must be > 0 when provided.")
            }
            if let maxNewTokensTable, maxNewTokensTable <= 0 {
                throw ValidationError("--max-new-tokens-table must be > 0 when provided.")
            }
            if let maxNewTokensFormula, maxNewTokensFormula <= 0 {
                throw ValidationError("--max-new-tokens-formula must be > 0 when provided.")
            }
            try validateUnitInterval(jpegQuality, optionName: "--jpeg-quality")
            try validateUnitInterval(layoutThreshold, optionName: "--layout-threshold")
            try validateUnitInterval(layoutThresholdClass1, optionName: "--layout-threshold-class-1")
            try validateUnitInterval(layoutThresholdClass7, optionName: "--layout-threshold-class-7")
            try validateUnitInterval(layoutThresholdClass14, optionName: "--layout-threshold-class-14")
            try await GLMOCRCLI.withExecutionDevice {
                let imageURLs = try GLMOCRCLI.resolveImageURLs(inputs: inputs)
                let pipeline = try await GLMOCRCLI.makeOCRPipeline(modelOptions: modelOptions, eprintln: eprintln)
                let dtypeOverride = try parseDTypeOverride(modelOptions.dtype)
                let parseDeterministicParameters = GLMOCRGenerationParameters(
                    temperature: 0,
                    topP: 1,
                    topK: 0,
                    repetitionPenalty: 1.1,
                    seed: nil
                )
                let generationParameters = decoding.generationParameters(defaultPreset: parseDeterministicParameters)

                let layoutURL = try await GLMOCRCLI.resolveModelDirectory(
                    model: "PaddlePaddle/PP-DocLayoutV3_safetensors",
                    revision: "main",
                    downloadBase: modelOptions.downloadBase,
                    globs: ["*.safetensors", "*.json"],
                    eprintln: eprintln
                )
                let layoutPipeline = try PPDocLayoutV3Pipeline(modelURL: layoutURL)
                let parser = GLMOCRDocumentParser(ocr: pipeline, layout: layoutPipeline)

                let regionPrompts = GLMOCRRegionRecognitionConfig(
                    textPrompt: GLMOCRPromptPresets.textRecognition,
                    tablePrompt: GLMOCRPromptPresets.tableRecognition,
                    formulaPrompt: GLMOCRPromptPresets.formulaRecognition
                )
                let defaultTextMaxTokens = min(decoding.maxNewTokens, 2048)
                let defaultTableMaxTokens = decoding.maxNewTokens
                let defaultFormulaMaxTokens = min(decoding.maxNewTokens, 1024)

                let fm = FileManager.default
                let outURL = URL(fileURLWithPath: (outputDir as NSString).expandingTildeInPath).standardizedFileURL
                try fm.createDirectory(at: outURL, withIntermediateDirectories: true)

                let parseConfig = GLMOCRDocumentParserConfig(
                    threshold: layoutThreshold,
                    layoutThresholdByClass: [
                        1: layoutThresholdClass1,
                        7: layoutThresholdClass7,
                        14: layoutThresholdClass14,
                    ],
                    layoutNMS: true,
                    includeFormulaNumbers: includeFormulaNumbers,
                    usePolygonMask: false,
                    useMaskCrop: false,
                    reencodeRegionCropsToJPEG: true,
                    jpegQuality: jpegQuality,
                    maxNewTokensPerRegion: decoding.maxNewTokens,
                    maxNewTokensTextPerRegion: maxNewTokensText ?? defaultTextMaxTokens,
                    maxNewTokensTablePerRegion: maxNewTokensTable ?? defaultTableMaxTokens,
                    maxNewTokensFormulaPerRegion: maxNewTokensFormula ?? defaultFormulaMaxTokens,
                    maxRegionsPerPage: 256,
                    generationParameters: generationParameters,
                    dtypeOverride: dtypeOverride,
                    prompts: regionPrompts
                )

                let effectiveBatchSize = batchSize > 0 ? batchSize : imageURLs.count

                var start = 0
                while start < imageURLs.count {
                    let end = min(start + effectiveBatchSize, imageURLs.count)
                    let chunk = Array(imageURLs[start..<end])

                    try autoreleasepool {
                        for url in chunk {
                            eprintln("==> \(url.lastPathComponent)")

                            let pages = try loadCGImages(at: url.path, pdfMaxPages: pdfMaxPages)
                            let result = try parser.parse(images: pages, config: parseConfig) { message in
                                eprintln("  \(message)")
                            }

                            let stem = url.deletingPathExtension().lastPathComponent
                            let itemDir = outURL.appendingPathComponent(stem, isDirectory: true)
                            let outMD = itemDir.appendingPathComponent("result.md")
                            let outJSON = itemDir.appendingPathComponent("result.json")

                            try fm.createDirectory(at: itemDir, withIntermediateDirectories: true)
                            try result.markdownResult.write(to: outMD, atomically: true, encoding: .utf8)
                            let jsonText = renderParseJSON(result.jsonResult)
                            try jsonText.write(to: outJSON, atomically: true, encoding: .utf8)
                            eprintln("wrote \(stem)")
                        }
                    }

                    start = end
                }

                eprintln("Wrote parse results for \(imageURLs.count) image(s) to \(outputDir)")
            }
        }
    }

    static func resolvePrompt(override: String?, preset: PromptPreset) -> String {
        let trimmed = override?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmed.isEmpty {
            return trimmed
        }
        return preset.promptText
    }

    static func withExecutionDevice<R>(
        operation: () async throws -> R
    ) async throws -> R {
        try await Device.withDefaultDevice(.gpu) {
            try await operation()
        }
    }

    static func resolveImageURLs(inputs: [String]) throws -> [URL] {
        let fm = FileManager.default
        let allowedExtensions = Set(["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "pdf"])

        let trimmedInputs =
            inputs
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !trimmedInputs.isEmpty else {
            throw ValidationError("Provide at least one --input.")
        }

        var resolved: [URL] = []
        for rawPath in trimmedInputs {
            let expanded = (rawPath as NSString).expandingTildeInPath
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: expanded, isDirectory: &isDirectory) else {
                throw ValidationError("Input path does not exist: \(displayRawPath(rawPath))")
            }

            if isDirectory.boolValue {
                let dirURL = URL(fileURLWithPath: expanded).standardizedFileURL
                let contents = try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil)
                let imageURLs =
                    contents
                    .filter { allowedExtensions.contains($0.pathExtension.lowercased()) }
                    .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                guard !imageURLs.isEmpty else {
                    throw ValidationError("No images found under: \(displayRawPath(rawPath))")
                }
                resolved.append(contentsOf: imageURLs)
            } else {
                try validateImageFile(pathToCheck: expanded, displayPath: rawPath, allowedExtensions: allowedExtensions)
                resolved.append(URL(fileURLWithPath: expanded).standardizedFileURL)
            }
        }

        return resolved
    }

    static func validateImageFile(pathToCheck: String, displayPath: String, allowedExtensions: Set<String>) throws {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: pathToCheck, isDirectory: &isDirectory) else {
            throw ValidationError("Image file does not exist: \(displayRawPath(displayPath))")
        }
        guard !isDirectory.boolValue else {
            throw ValidationError("Expected an image file, got directory: \(displayRawPath(displayPath))")
        }

        let ext = URL(fileURLWithPath: pathToCheck).pathExtension.lowercased()
        guard allowedExtensions.contains(ext) else {
            throw ValidationError("Unsupported image extension: \(displayRawPath(displayPath))")
        }
    }

    static func makeOCRPipeline(
        modelOptions: ModelOptions,
        eprintln: @escaping (String) -> Void
    ) async throws -> GLMOCRPipeline {
        try configureMLXCacheLimit(cacheLimitMB: modelOptions.cacheLimitMB, eprintln: eprintln)

        let isLocalModelDir = (try? resolveLocalModelDirectory(modelOptions.model)) != nil
        let modelURL = try await resolveModelDirectory(
            model: modelOptions.model,
            revision: modelOptions.revision,
            downloadBase: modelOptions.downloadBase,
            globs: ["*.safetensors", "*.json", "*.jinja", "tokenizer.*"],
            eprintln: eprintln
        )

        if isLocalModelDir {
            eprintln("Model: local (\(modelURL.lastPathComponent))")
        } else {
            eprintln("Model: \(modelOptions.model)@\(modelOptions.revision)")
        }
        eprintln("Device: gpu")

        let pipeline = try await GLMOCRPipeline(modelURL: modelURL, strictTokenizer: true)

        if let dtypeOverride = try parseDTypeOverride(modelOptions.dtype) {
            try pipeline.loadModel(dtype: dtypeOverride)
        }

        return pipeline
    }

    static func configureMLXCacheLimit(
        cacheLimitMB: Int,
        eprintln: @escaping (String) -> Void
    ) throws {
        guard cacheLimitMB >= 0 else {
            throw ValidationError("--cache-limit must be >= 0.")
        }

        let (bytes, overflow) = cacheLimitMB.multipliedReportingOverflow(by: 1024 * 1024)
        guard !overflow else {
            throw ValidationError("--cache-limit is too large.")
        }

        Memory.cacheLimit = bytes
        if cacheLimitMB != 2048 {
            eprintln("GPU cache limit: \(cacheLimitMB)MB")
        }
    }

    static func resolveModelDirectory(
        model: String,
        revision: String,
        downloadBase: String?,
        globs: [String],
        eprintln: @escaping (String) -> Void
    ) async throws -> URL {
        if let localURL = try resolveLocalModelDirectory(model) {
            return localURL
        }

        let downloadBaseURL = resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBase)
        try? FileManager.default.createDirectory(at: downloadBaseURL, withIntermediateDirectories: true)

        let hub = HubApi(downloadBase: downloadBaseURL, useOfflineMode: false)
        var lastCompleted: Int64 = -1

        return try await hub.snapshot(from: model, revision: revision, matching: globs) { progress in
            if progress.completedUnitCount != lastCompleted {
                lastCompleted = progress.completedUnitCount
                let total = max(progress.totalUnitCount, 1)
                eprintln("Downloading \(model) (\(lastCompleted)/\(total) files)...")
            }
        }
    }

    static func runBatchOCR(
        pipeline: GLMOCRPipeline,
        imageURLs: [URL],
        prompt: String,
        maxNewTokens: Int,
        generationParameters: GLMOCRGenerationParameters,
        dtypeOverride: MLX.DType?,
        skipSpecialTokens: Bool,
        outputDir: String,
        postResizeJpegQuality: Double?,
        batchSize: Int,
        eprintln: (String) -> Void
    ) throws {
        let fm = FileManager.default
        let outURL = URL(fileURLWithPath: (outputDir as NSString).expandingTildeInPath).standardizedFileURL
        try fm.createDirectory(at: outURL, withIntermediateDirectories: true)

        let effectiveBatchSize = batchSize > 0 ? batchSize : imageURLs.count

        var start = 0
        while start < imageURLs.count {
            let end = min(start + effectiveBatchSize, imageURLs.count)
            let chunk = Array(imageURLs[start..<end])

            try autoreleasepool {
                for url in chunk {
                    eprintln("==> \(url.lastPathComponent)")
                }

                let batchOutputs = try pipeline.recognizeBatch(
                    imagePaths: chunk.map(\.path),
                    prompt: prompt,
                    maxNewTokens: maxNewTokens,
                    generationParameters: generationParameters,
                    skipSpecialTokens: skipSpecialTokens,
                    dtypeOverride: dtypeOverride,
                    postResizeJPEGRoundTripQuality: postResizeJpegQuality
                )
                for (url, outputText) in zip(chunk, batchOutputs) {
                    let outPath = outURL.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".txt")
                    try outputText.write(to: outPath, atomically: true, encoding: .utf8)
                    eprintln("wrote \(outPath.lastPathComponent)")
                }
            }

            start = end
        }
    }

    static func resolveHuggingFaceHubCacheDirectory(downloadBasePath: String?) -> URL {
        func normalize(_ value: String?) -> String? {
            let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let trimmed, !trimmed.isEmpty else { return nil }
            return trimmed
        }

        if let downloadBasePath = normalize(downloadBasePath) {
            return URL(fileURLWithPath: downloadBasePath).standardizedFileURL
        }

        let env = ProcessInfo.processInfo.environment
        if let hubCache = normalize(env["HF_HUB_CACHE"]) {
            return URL(fileURLWithPath: hubCache).standardizedFileURL
        }
        if let hfHome = normalize(env["HF_HOME"]) {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub").standardizedFileURL
        }

        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub").standardizedFileURL
    }

    static func resolveLocalModelDirectory(_ path: String) throws -> URL? {
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory) else {
            return nil
        }
        guard isDirectory.boolValue else {
            throw ValidationError("Expected a directory at \(displayRawPath(path)).")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }

    static func displayRawPath(_ rawPath: String) -> String {
        rawPath.hasPrefix("/") ? URL(fileURLWithPath: rawPath).lastPathComponent : rawPath
    }

    static func renderParseJSON(_ pages: [[GLMOCRFormattedRegion]]) -> String {
        if pages.isEmpty { return "[]" }

        let indent0 = ""
        let indent2 = "  "
        let indent4 = "    "
        let indent6 = "      "
        let indent8 = "        "

        var out = ""
        out.append("[\n")
        for (pageIndex, page) in pages.enumerated() {
            out.append(indent2)
            out.append("[\n")
            for (regionIndex, region) in page.enumerated() {
                out.append(indent4)
                out.append("{\n")

                out.append(indent6)
                out.append("\"index\": ")
                out.append(String(region.index))
                out.append(",\n")

                out.append(indent6)
                out.append("\"label\": \"")
                out.append(jsonEscaped(region.label))
                out.append("\",\n")

                out.append(indent6)
                out.append("\"bbox_2d\": ")
                if let bbox = region.bbox2D {
                    out.append("[\n")
                    for (i, v) in bbox.enumerated() {
                        out.append(indent8)
                        out.append(String(v))
                        if i + 1 < bbox.count { out.append(",") }
                        out.append("\n")
                    }
                    out.append(indent6)
                    out.append("]")
                } else {
                    out.append("null")
                }
                out.append(",\n")

                out.append(indent6)
                out.append("\"content\": ")
                if let content = region.content {
                    out.append("\"")
                    out.append(jsonEscaped(content))
                    out.append("\"")
                } else {
                    out.append("null")
                }
                out.append(",\n")

                out.append(indent6)
                out.append("\"native_label\": \"")
                out.append(jsonEscaped(region.nativeLabel))
                out.append("\"\n")

                out.append(indent4)
                out.append("}")
                if regionIndex + 1 < page.count { out.append(",") }
                out.append("\n")
            }
            out.append(indent2)
            out.append("]")
            if pageIndex + 1 < pages.count { out.append(",") }
            out.append("\n")
        }
        out.append(indent0)
        out.append("]")
        return out
    }

    static func jsonEscaped(_ value: String) -> String {
        var out = ""
        out.reserveCapacity(value.utf8.count)

        for scalar in value.unicodeScalars {
            let v = scalar.value
            switch v {
            case 0x22:
                out.append("\\\"")
            case 0x5C:
                out.append("\\\\")
            case 0x08:
                out.append("\\b")
            case 0x0C:
                out.append("\\f")
            case 0x0A:
                out.append("\\n")
            case 0x0D:
                out.append("\\r")
            case 0x09:
                out.append("\\t")
            case 0x00...0x1F:
                out.append(String(format: "\\u%04X", v))
            default:
                out.append(String(scalar))
            }
        }

        return out
    }
}

private func loadCGImage(at path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw ValidationError("Failed to decode image \(url.lastPathComponent).")
    }
    return image
}

private func loadCGImages(at path: String, pdfMaxPages: Int? = nil) throws -> [CGImage] {
    let url = URL(fileURLWithPath: path)
    if url.pathExtension.lowercased() == "pdf" {
        return try loadCGImagesFromPDF(at: url, dpi: 200, maxPages: pdfMaxPages)
    }
    return [try loadCGImage(at: path)]
}

private func loadCGImagesFromPDF(at url: URL, dpi: CGFloat, maxPages: Int?) throws -> [CGImage] {
    func eprintln(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    guard let document = CGPDFDocument(url as CFURL) else {
        throw ValidationError("Failed to decode PDF \(url.lastPathComponent).")
    }
    let pageCount = document.numberOfPages
    guard pageCount > 0 else {
        throw ValidationError("PDF has no pages: \(url.lastPathComponent).")
    }

    let pagesToRender = maxPages.map { min(max($0, 0), pageCount) } ?? pageCount

    if pagesToRender > 1 {
        let suffix = pagesToRender == pageCount ? "" : " (limited to \(pagesToRender))"
        eprintln("Rendering \(url.lastPathComponent) (\(pageCount) pages)\(suffix) at \(Int(max(dpi, 1)))dpi...")
    }

    let scale = max(dpi, 1) / 72.0
    var pages: [CGImage] = []
    pages.reserveCapacity(pagesToRender)

    for pageIndex in 1...pagesToRender {
        guard let page = document.page(at: pageIndex) else { continue }
        let pageBox = page.getBoxRect(.mediaBox)
        let width = max(Int(ceil(pageBox.width * scale)), 1)
        let height = max(Int(ceil(pageBox.height * scale)), 1)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw ValidationError("Failed to create sRGB color space.")
        }

        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            throw ValidationError("Failed to create PDF render context.")
        }

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        context.saveGState()
        let normalizedRotation = ((page.rotationAngle % 360) + 360) % 360
        if normalizedRotation == 0 {
            context.scaleBy(x: scale, y: scale)
        } else {
            let targetRect = CGRect(x: 0, y: 0, width: width, height: height)
            let transform = page.getDrawingTransform(.mediaBox, rect: targetRect, rotate: 0, preserveAspectRatio: true)
            context.concatenate(transform)
        }
        context.drawPDFPage(page)
        context.restoreGState()

        guard let image = context.makeImage() else {
            throw ValidationError("Failed to render PDF page \(pageIndex) of \(url.lastPathComponent).")
        }
        pages.append(image)

        if pagesToRender > 1,
            pageIndex == 1 || pageIndex == pagesToRender || pageIndex.isMultiple(of: 5)
        {
            eprintln("rendered page \(pageIndex)/\(pagesToRender)")
        }
    }

    guard !pages.isEmpty else {
        throw ValidationError("Failed to render PDF \(url.lastPathComponent).")
    }
    return pages
}

private func parseDTypeOverride(_ value: String) throws -> MLX.DType? {
    let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    switch normalized {
    case "", "auto":
        return nil
    case "float32", "f32":
        return .float32
    case "float16", "f16":
        return .float16
    case "bfloat16", "bf16":
        return .bfloat16
    default:
        throw ValidationError("Unsupported --dtype \(value). Use auto|float32|float16|bfloat16.")
    }
}
