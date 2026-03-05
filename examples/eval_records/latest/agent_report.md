# Example evaluation (agent report)

- generated_at: `2026-03-06T04:21:05+00:00`
- git: `3e16b85`
- git_head_sha: `3e16b850cd7d576484fac60464ddf737b5b1f6b3`
- git_dirty: `False`
- mean_final_overall: `0.9353`

## Scores

| Example | Final | Δ vs baseline | Parity | Result→Golden | Ref→Golden | Rules |
|---|---:|---:|---:|---:|---:|---:|
| `GLM-4.5V_Page_1` | 0.8570 | +0.0000 | 0.8570 | None | None | 0/0 |
| `GLM-4.5V_Pages_1_2_3` | 0.8667 | +0.0000 | 0.8739 | 0.8786 | 0.9230 | 0/7 |
| `code` | 0.9457 | +0.0000 | 0.9450 | 0.7244 | 0.7270 | 0/0 |
| `handwritten` | 1.0000 | +0.0000 | 0.9870 | 0.9706 | 0.9550 | 0/1 |
| `page` | 0.8396 | +0.0000 | 0.8467 | 0.5639 | 0.6078 | 0/0 |
| `paper` | 0.9853 | +0.0000 | 0.9853 | 0.7058 | 0.7090 | 0/0 |
| `seal` | 0.9894 | +0.0000 | 0.9894 | 0.9808 | 0.9808 | 0/0 |
| `table` | 0.9987 | +0.0000 | 0.9987 | 1.0000 | 1.0000 | 0/0 |

## Focus

### `page`

- final_overall: `0.8396`
- delta_vs_baseline: `+0.0000`
- result_md: `examples/result/page/page.md`
- result_json: `examples/result/page/page.json`
- reference_md: `examples/reference_result/page/page.md`
- reference_json: `examples/reference_result/page/page.json`
- eval_report_md: `examples/eval_records/latest/examples/page/report.md`
- eval_report_json: `examples/eval_records/latest/examples/page/report.json`
- golden_md: `examples/golden_result/page/page.md`
- golden_json: `examples/golden_result/page/page.json`

**Signals**
- lowest dimension: `text_fidelity` = 0.8070
- json components < 0.98: content=0.8300, bbox=0.9238
- block_shape: 0.6897
- text blocks: missing=1, paired=28
- lowest text pairs: (idx=8, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=12, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=14, status=paired, actual=list_item, expected=heading, score=0.0000), (idx=15, status=paired, actual=heading, expected=heading, score=0.0000), (idx=19, status=paired, actual=paragraph, expected=list_item, score=0.0000)
- fix_hints: OCR JSON (block ordering, bbox rounding, content normalization), Markdown block segmentation (heading/list/paragraph splits)

### `GLM-4.5V_Page_1`

- final_overall: `0.8570`
- delta_vs_baseline: `+0.0000`
- result_md: `examples/result/GLM-4.5V_Page_1/GLM-4.5V_Page_1.md`
- result_json: `examples/result/GLM-4.5V_Page_1/GLM-4.5V_Page_1.json`
- reference_md: `examples/reference_result/GLM-4.5V_Page_1/GLM-4.5V_Page_1.md`
- reference_json: `examples/reference_result/GLM-4.5V_Page_1/GLM-4.5V_Page_1.json`
- eval_report_md: `examples/eval_records/latest/examples/GLM-4.5V_Page_1/report.md`
- eval_report_json: `examples/eval_records/latest/examples/GLM-4.5V_Page_1/report.json`
- golden: not available for this example

**Signals**
- lowest dimension: `decorative_style` = 0.7500
- style.center_wrappers: actual=0, expected=4
- json components < 0.98: content=0.8123, bbox=0.9530
- block_shape: 0.3889
- text blocks: missing=7, paired=11
- lowest text pairs: (idx=0, status=paired, actual=heading, expected=paragraph, score=0.0000), (idx=2, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=3, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=4, status=paired, actual=heading, expected=paragraph, score=0.0000), (idx=6, status=paired, actual=paragraph, expected=heading, score=0.0000)
- fix_hints: Markdown style wrappers (centering/bold/fences), OCR JSON (block ordering, bbox rounding, content normalization), Markdown block segmentation (heading/list/paragraph splits)

### `GLM-4.5V_Pages_1_2_3`

- final_overall: `0.8667`
- delta_vs_baseline: `+0.0000`
- result_md: `examples/result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.md`
- result_json: `examples/result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.json`
- reference_md: `examples/reference_result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.md`
- reference_json: `examples/reference_result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.json`
- eval_report_md: `examples/eval_records/latest/examples/GLM-4.5V_Pages_1_2_3/report.md`
- eval_report_json: `examples/eval_records/latest/examples/GLM-4.5V_Pages_1_2_3/report.json`
- golden_md: `examples/golden_result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.md`
- golden_json: `examples/golden_result/GLM-4.5V_Pages_1_2_3/GLM-4.5V_Pages_1_2_3.json`

**Signals**
- lowest dimension: `decorative_style` = 0.6875
- style.center_wrappers: actual=0, expected=4
- json components < 0.98: content=0.9334, bbox=0.9530
- block_shape: 0.3250
- text blocks: missing=8, paired=32
- lowest text pairs: (idx=0, status=paired, actual=heading, expected=paragraph, score=0.0000), (idx=2, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=3, status=paired, actual=paragraph, expected=paragraph, score=0.0000), (idx=4, status=paired, actual=heading, expected=paragraph, score=0.0000), (idx=6, status=paired, actual=paragraph, expected=heading, score=0.0000)
- fix_hints: Markdown style wrappers (centering/bold/fences), OCR JSON (block ordering, bbox rounding, content normalization), Markdown block segmentation (heading/list/paragraph splits)
