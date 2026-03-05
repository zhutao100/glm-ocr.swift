# GLM-OCR (Swift)

Swift/MLX CLI for `zai-org/GLM-OCR`.

## Build

```bash
xcodebuild -scheme GLMOCR-Package -destination "platform=macOS" -configuration Release -derivedDataPath build build
export PATH="$PWD/build/Build/Products/Release:$PATH"
```

## CLI usage

By default, the CLI uses `--model zai-org/GLM-OCR`.

Run OCR on one image (prints to stdout):

```bash
glm-ocr ocr --input /path/to/image.png
```

Run OCR on multiple images (writes `<stem>.txt` under `--output-dir`):

```bash
glm-ocr ocr --output-dir outputs/ocr --input a.png --input b.png
```

Use a local model directory (instead of downloading from Hugging Face):

```bash
glm-ocr ocr --model /path/to/GLM-OCR --input /path/to/image.png
```

Override the GPU cache limit (MB):

```bash
glm-ocr ocr --cache-limit 1024 --input /path/to/image.png
```

Run layout + parse (writes `<output-dir>/<stem>/{result.md,result.json}` by default):

```bash
glm-ocr parse --input /path/to/image.png --output-dir outputs/parse
```

Use `<stem>.{md,json}` file names (needed for `tools/example_eval`):

```bash
glm-ocr parse --input /path/to/image.png --output-dir outputs/parse --output-naming stem
```

Parse a directory of images:

```bash
glm-ocr parse --input /path/to/images --output-dir outputs/parse
```

Continue on per-input errors (useful for batch runs):

```bash
glm-ocr parse --input /path/to/images --output-dir outputs/parse --continue-on-error
```

Tune parse layout thresholds:

```bash
glm-ocr parse --input /path/to/image.png --output-dir outputs/parse \
  --layout-threshold 0.4 \
  --layout-threshold-class-1 0.10 \
  --layout-threshold-class-7 0.30 \
  --layout-threshold-class-14 0.30
```

## Debug build (optional)

```bash
xcodebuild -scheme GLMOCR-Package -destination "platform=macOS" -configuration Debug -derivedDataPath build build
export PATH="$PWD/build/Build/Products/Debug:$PATH"
```

## Help

```bash
glm-ocr --help
glm-ocr help ocr
glm-ocr help parse
```

## Examples regression loop

### Recommended (agentic loop: refresh → eval → record → compare):

```bash
scripts/verify_example_eval.sh
```

This updates the persistent record under `examples/eval_records/latest/` (see `examples/eval_records/README.md`).

### Manual (refresh + eval only):

Generate `examples/result/` (also writes `examples/result/.run_examples_meta.json`):

```bash
scripts/run_examples.sh
```

Evaluate `examples/result/` against `examples/{reference_result,golden_result}/`:

```bash
uv run --project tools/example_eval example-eval evaluate --repo-root .
```

Record the run output into `examples/eval_records/latest/`:

```bash
python3 scripts/example_eval_record.py --repo-root .
```
