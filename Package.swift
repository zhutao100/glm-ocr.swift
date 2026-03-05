// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "GLMOCR",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "GLMOCR", targets: ["GLMOCR"]),
        .executable(name: "glm-ocr", targets: ["GLMOCRCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.1.6")),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "1.5.0")),
    ],
    targets: [
        .target(
            name: "GLMOCR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "GLMOCRCLI",
            dependencies: [
                "GLMOCR",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "GLMOCRTests",
            dependencies: [
                "GLMOCR"
            ],
            resources: [
                .process("Fixtures")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
