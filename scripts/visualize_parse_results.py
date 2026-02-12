#!/usr/bin/env python3
"""Visualize a single `glm-ocr parse` result as HTML overlays.

Expected inputs:
- source image file
- parse result JSON file (result.json)
- optional markdown file (result.md)
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one glm-ocr parse output with image-region overlays."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to source image file used by parse.",
    )
    parser.add_argument(
        "--result-json",
        required=True,
        help="Path to parse JSON result file (result.json).",
    )
    parser.add_argument(
        "--result-md",
        default=None,
        help="Optional path to parse markdown result (result.md).",
    )
    parser.add_argument(
        "--out-html",
        default=None,
        help="Output HTML path (default: <result-json-dir>/visualization.html).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated HTML after rendering.",
    )
    return parser.parse_args()


def load_parse_regions(result_json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(result_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("result.json root must be a list.")

    flattened: list[dict[str, Any]] = []
    for page_idx, page in enumerate(data):
        if not isinstance(page, list):
            raise ValueError(f"result.json page {page_idx} is not a list.")

        for position, region in enumerate(page):
            if not isinstance(region, dict):
                raise ValueError(f"result.json page {page_idx} region {position} is not an object.")

            bbox = region.get("bbox_2d")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            except (TypeError, ValueError):
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            label = str(region.get("label", "unknown"))
            native_label = str(region.get("native_label", ""))
            content = str(region.get("content", ""))
            region_index = region.get("index")
            if isinstance(region_index, int):
                index_value = region_index
            else:
                index_value = position

            flattened.append(
                {
                    "id": len(flattened),
                    "page": page_idx,
                    "index": index_value,
                    "label": label,
                    "native_label": native_label,
                    "content": content,
                    "bbox": [x1, y1, x2, y2],
                    "color": color_for_label(label),
                }
            )

    return flattened


def color_for_label(label: str) -> str:
    normalized = label.lower()
    if normalized in {"text", "paragraph", "paragraph_title", "title"}:
        return "#2563eb"
    if normalized in {"table"}:
        return "#16a34a"
    if normalized in {"formula", "display_formula", "inline_formula"}:
        return "#d97706"
    if normalized in {"image", "chart", "figure"}:
        return "#7c3aed"
    return "#dc2626"


def render_stem_html(
    stem: str,
    image_path: Path,
    image_src: str,
    result_md_path: Path,
    regions: list[dict[str, Any]],
    out_path: Path,
) -> None:
    markdown_text = result_md_path.read_text(encoding="utf-8") if result_md_path.is_file() else ""
    region_rows = "\n".join(render_region_row(region) for region in regions)
    regions_json = json.dumps(regions, ensure_ascii=False)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GLM-OCR Parse Visualizer - __TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #e2e8f0;
      --active: #fef3c7;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background: var(--bg);
    }
    .page {
      display: grid;
      grid-template-columns: minmax(480px, 58vw) minmax(460px, 42vw);
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .panel header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .title {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.3;
    }
    .meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }
    .image-wrap {
      padding: 14px;
      overflow: auto;
      flex: 1;
    }
    .canvas {
      position: relative;
      display: inline-block;
      max-width: 100%;
    }
    .canvas img {
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .overlay {
      position: absolute;
      left: 0;
      top: 0;
      pointer-events: none;
    }
    .box {
      position: absolute;
      border: 2px solid var(--box-color, #dc2626);
      background: color-mix(in srgb, var(--box-color, #dc2626) 14%, transparent);
      border-radius: 4px;
      pointer-events: auto;
      cursor: pointer;
    }
    .box.active {
      border-width: 3px;
      box-shadow: 0 0 0 2px #11182722;
    }
    .list-wrap {
      overflow: auto;
      flex: 1;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px;
      font-weight: 700;
    }
    tbody td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
    }
    tbody tr.active {
      background: var(--active);
    }
    .mono { font-family: var(--mono); }
    .content {
      white-space: pre-wrap;
      word-break: break-word;
      max-width: 460px;
    }
    .md-wrap {
      border-top: 1px solid var(--line);
      background: #fafcff;
      padding: 12px 14px;
    }
    .md-wrap h2 {
      margin: 0 0 8px 0;
      font-size: 13px;
    }
    .md-wrap pre {
      margin: 0;
      max-height: 220px;
      overflow: auto;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    @media (max-width: 1200px) {
      .page { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="panel">
      <header>
        <h1 class="title">__TITLE__</h1>
        <div class="meta">__IMAGE_PATH__</div>
        <div class="meta" id="coord-mode"></div>
      </header>
      <div class="image-wrap">
        <div class="canvas" id="canvas">
          <img id="source-image" src="__IMAGE_SRC__" alt="Source image" />
          <div id="overlay" class="overlay"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <header>
        <h2 class="title">Detected Regions (__REGION_COUNT__)</h2>
        <div class="meta">Click a row or box to inspect the same region.</div>
      </header>
      <div class="list-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Label</th>
              <th>Native</th>
              <th>BBox</th>
              <th>Content</th>
            </tr>
          </thead>
          <tbody>
            __REGION_ROWS__
          </tbody>
        </table>
      </div>
      <div class="md-wrap">
        <h2>result.md</h2>
        <pre>__MARKDOWN__</pre>
      </div>
    </section>
  </main>

  <script>
    const regions = __REGIONS_JSON__;
    const image = document.getElementById('source-image');
    const overlay = document.getElementById('overlay');
    const coordMode = document.getElementById('coord-mode');

    function computeTransform(naturalWidth, naturalHeight) {
      // `glm-ocr parse` emits bbox_2d in normalized 0..1000 coordinates.
      // Keep a raw fallback for compatibility with other inputs.
      const looksNormalized1000 = regions.every((region) => (
        Array.isArray(region.bbox) &&
        region.bbox.length === 4 &&
        region.bbox.every((value) => Number.isFinite(value) && value >= -2 && value <= 1002)
      ));
      if (looksNormalized1000) {
        return { mode: 'normalized-1000', scaleX: naturalWidth / 1000.0, scaleY: naturalHeight / 1000.0 };
      }
      return { mode: 'raw', scaleX: 1, scaleY: 1 };
    }

    function mapBBox(bbox, transform, naturalWidth, naturalHeight) {
      const x1 = bbox[0] * transform.scaleX;
      const y1 = bbox[1] * transform.scaleY;
      const x2 = bbox[2] * transform.scaleX;
      const y2 = bbox[3] * transform.scaleY;
      return [
        Math.max(0, Math.min(naturalWidth, x1)),
        Math.max(0, Math.min(naturalHeight, y1)),
        Math.max(0, Math.min(naturalWidth, x2)),
        Math.max(0, Math.min(naturalHeight, y2)),
      ];
    }

    function renderBoxes() {
      const displayWidth = image.clientWidth;
      const displayHeight = image.clientHeight;
      const naturalWidth = image.naturalWidth || displayWidth;
      const naturalHeight = image.naturalHeight || displayHeight;
      const scaleX = displayWidth / naturalWidth;
      const scaleY = displayHeight / naturalHeight;
      const transform = computeTransform(naturalWidth, naturalHeight);

      overlay.style.width = `${displayWidth}px`;
      overlay.style.height = `${displayHeight}px`;
      overlay.innerHTML = '';
      coordMode.textContent = `bbox mode: ${transform.mode}`;

      for (const region of regions) {
        const [x1, y1, x2, y2] = mapBBox(region.bbox, transform, naturalWidth, naturalHeight);
        const box = document.createElement('div');
        box.className = 'box';
        box.dataset.regionId = String(region.id);
        box.style.setProperty('--box-color', region.color || '#dc2626');
        box.style.left = `${Math.round(x1 * scaleX)}px`;
        box.style.top = `${Math.round(y1 * scaleY)}px`;
        box.style.width = `${Math.max(1, Math.round((x2 - x1) * scaleX))}px`;
        box.style.height = `${Math.max(1, Math.round((y2 - y1) * scaleY))}px`;
        box.addEventListener('click', () => setActive(region.id, true));
        overlay.appendChild(box);
      }
    }

    function clearActive() {
      document.querySelectorAll('.box.active').forEach((el) => el.classList.remove('active'));
      document.querySelectorAll('tr.active').forEach((el) => el.classList.remove('active'));
    }

    function setActive(regionId, scroll) {
      clearActive();
      const box = document.querySelector(`.box[data-region-id="${regionId}"]`);
      const row = document.querySelector(`tr[data-region-id="${regionId}"]`);
      if (box) box.classList.add('active');
      if (row) {
        row.classList.add('active');
        if (scroll) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }

    for (const row of document.querySelectorAll('tr[data-region-id]')) {
      row.addEventListener('click', () => setActive(Number(row.dataset.regionId), false));
    }

    image.addEventListener('load', () => {
      renderBoxes();
      if (regions.length > 0) {
        setActive(regions[0].id, false);
      }
    });
    window.addEventListener('resize', renderBoxes);
  </script>
</body>
</html>
"""

    html_output = (
        template.replace("__TITLE__", html.escape(stem))
        .replace("__IMAGE_PATH__", html.escape(str(image_path)))
        .replace("__IMAGE_SRC__", html.escape(image_src))
        .replace("__REGION_COUNT__", str(len(regions)))
        .replace("__REGION_ROWS__", region_rows)
        .replace("__MARKDOWN__", html.escape(markdown_text))
        .replace("__REGIONS_JSON__", regions_json)
    )

    out_path.write_text(html_output, encoding="utf-8")


def render_region_row(region: dict[str, Any]) -> str:
    region_id = region["id"]
    label = html.escape(str(region.get("label", "")))
    native_label = html.escape(str(region.get("native_label", "")))
    bbox = region.get("bbox", [0, 0, 0, 0])
    bbox_text = html.escape(json.dumps(bbox, ensure_ascii=False))
    content = html.escape(str(region.get("content", "")))
    color = html.escape(str(region.get("color", "#dc2626")))
    page = int(region.get("page", 0))
    index = int(region.get("index", region_id))

    return (
        f'<tr data-region-id="{region_id}">'
        f'<td class="mono">{page}:{index}</td>'
        f'<td><span class="mono" style="color: {color};">{label}</span></td>'
        f'<td class="mono">{native_label}</td>'
        f'<td class="mono">{bbox_text}</td>'
        f'<td class="content">{content}</td>'
        "</tr>"
    )


def open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


def main() -> int:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    result_json_path = Path(args.result_json).expanduser().resolve()
    result_md_path = (
        Path(args.result_md).expanduser().resolve()
        if args.result_md
        else result_json_path.with_name("result.md")
    )
    out_html_path = (
        Path(args.out_html).expanduser().resolve()
        if args.out_html
        else result_json_path.with_name("visualization.html")
    )

    if not image_path.is_file():
        print(f"error: --image is not a file: {image_path}", file=sys.stderr)
        return 2
    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        print(f"error: unsupported image extension for --image: {image_path}", file=sys.stderr)
        return 2
    if not result_json_path.is_file():
        print(f"error: --result-json is not a file: {result_json_path}", file=sys.stderr)
        return 2

    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = out_html_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    stem = result_json_path.parent.name or image_path.stem
    copied_image_name = f"{stem}{image_path.suffix.lower()}"
    copied_image_path = assets_dir / copied_image_name
    shutil.copy2(image_path, copied_image_path)
    image_src = (Path("assets") / copied_image_name).as_posix()

    try:
        regions = load_parse_regions(result_json_path)
        render_stem_html(
            stem=stem,
            image_path=image_path,
            image_src=image_src,
            result_md_path=result_md_path,
            regions=regions,
            out_path=out_html_path,
        )
    except Exception as exc:
        print(f"error: failed to build visualization: {exc}", file=sys.stderr)
        return 1

    print(f"ok: wrote {out_html_path} ({len(regions)} regions)")
    if result_md_path.is_file():
        print(f"md: {result_md_path}")
    else:
        print("md: not found (continuing without result.md)")

    if args.open:
        open_path(out_html_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
