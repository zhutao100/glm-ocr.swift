#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate `examples/result/` by running the Swift CLI against `examples/source/`.

Defaults:
  - builds and runs with `-c release`
  - ensures `mlx.metallib` exists next to the built executable
  - writes `examples/result/<stem>/{<stem>.md,<stem>.json}`
  - continues on per-input errors (exits non-zero if any failed)

Usage:
  scripts/run_examples.sh [-c debug|release] [--no-clean]
EOF
}

config="release"
clean=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configuration)
      config="${2:-}"; shift 2 ;;
    --no-clean)
      clean=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$config" != "debug" && "$config" != "release" ]]; then
  echo "Invalid configuration: $config (expected 'debug' or 'release')" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root_dir/examples/source"
result_dir="$root_dir/examples/result"

if [[ ! -d "$source_dir" ]]; then
  echo "Missing examples source directory: $source_dir" >&2
  exit 1
fi

cd "$root_dir"
swift build -c "$config" --product glm-ocr
bin_path="$(swift build -c "$config" --product glm-ocr --show-bin-path)"
exe="$bin_path/glm-ocr"

if [[ ! -x "$exe" ]]; then
  echo "Built executable not found: $exe" >&2
  exit 1
fi

metallib="$bin_path/mlx.metallib"
if [[ ! -f "$metallib" ]]; then
  echo "Missing mlx.metallib for -c $config; building..." >&2
  "$root_dir/scripts/build_mlx_metallib.sh" -c "$config"
fi

if [[ ! -f "$metallib" ]]; then
  echo "mlx.metallib still missing after build: $metallib" >&2
  exit 1
fi

if [[ "$clean" -eq 1 ]]; then
  rm -rf "$result_dir"
fi
mkdir -p "$result_dir"

set +e
"$exe" parse \
  --input "$source_dir" \
  --output-dir "$result_dir" \
  --output-naming stem \
  --continue-on-error
parse_status=$?
set -e

shopt -s nullglob
for pdf_path in "$source_dir"/*.pdf; do
  pdf_name="$(basename "$pdf_path")"
  if [[ "$pdf_name" == ._* ]]; then
    continue
  fi
  pdf_stem="${pdf_name%.pdf}"
  mkdir -p "$result_dir/$pdf_stem/imgs"
done

if [[ "$parse_status" -ne 0 ]]; then
  echo "Completed with errors (exit $parse_status). See stderr logs above." >&2
fi
echo "OK: wrote $result_dir"
exit "$parse_status"
