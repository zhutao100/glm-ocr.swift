# Example evaluation (agent report)

- generated_at: `2026-03-17T18:51:41+00:00`
- git: `20c4fb0`
- git_head_sha: `20c4fb0184085388e29f0bc4a680f7c0160c9b72`
- git_dirty: `False`
- mean_final_overall: `0.9466`

## Scores

| Example | Final | Δ vs baseline | Parity | Result→Golden | Ref→Golden | Rules |
|---|---:|---:|---:|---:|---:|---:|
| `GLM-4.5V_Page_1` | 0.9067 | +0.0497 | 0.9224 | 0.8679 | 0.9308 | 0/0 |
| `GLM-4.5V_Pages_1_2_3` | 0.9557 | +0.0890 | 0.9627 | 0.8463 | 0.8743 | 0/7 |
| `code` | 0.9105 | -0.0352 | 0.9384 | 0.8382 | 0.8406 | 2/3 |
| `handwritten` | 1.0000 | +0.0000 | 0.9875 | 0.8034 | 0.7967 | 0/1 |
| `page` | 0.8254 | -0.0142 | 0.8836 | 0.4837 | 0.5148 | 1/1 |
| `paper` | 0.9864 | +0.0011 | 0.9864 | 0.7326 | 0.7343 | 0/2 |
| `seal` | 0.9894 | +0.0000 | 0.9894 | 0.9875 | 0.9875 | 0/0 |
| `table` | 0.9987 | +0.0000 | 0.9987 | 1.0000 | 1.0000 | 0/0 |

## Focus

### `page`

- final_overall: `0.8254`
- delta_vs_baseline: `-0.0142`
- result_md: `examples/result/page/page.md`
- result_json: `examples/result/page/page.json`
- reference_md: `examples/reference_result/page/page.md`
- reference_json: `examples/reference_result/page/page.json`
- eval_report_md: `examples/eval_records/latest/examples/page/report.md`
- eval_report_json: `examples/eval_records/latest/examples/page/report.json`
- golden_md: `examples/golden_result/page/page.md`
- golden_json: `examples/golden_result/page/page.json`

**Signals**
- lowest dimension: `text_fidelity` = 0.8130
- json components < 0.98: content=0.8289, bbox=0.9238
- text blocks: extra_actual=1, paired=28
- lowest text pairs: (idx=None, status=extra_actual, actual=None, expected=None, score=0.0000), (idx=None, status=paired, actual=list_item, expected=list_item, score=0.3929), (idx=None, status=paired, actual=list_item, expected=list_item, score=0.4177), (idx=None, status=paired, actual=paragraph, expected=paragraph, score=0.4238), (idx=None, status=paired, actual=paragraph, expected=paragraph, score=0.4541)
- rules failed: 1 (first: glue_strength_constant / contains / Missing required phrase: '0.2\\mathrm{N} / \\mathrm{mm}^{2}'.)
- fix_hints: OCR JSON (block ordering, bbox rounding, content normalization), Markdown block segmentation (heading/list/paragraph splits), Example-specific rules/regression

### `code`

- final_overall: `0.9105`
- delta_vs_baseline: `-0.0352`
- result_md: `examples/result/code/code.md`
- result_json: `examples/result/code/code.json`
- reference_md: `examples/reference_result/code/code.md`
- reference_json: `examples/reference_result/code/code.json`
- eval_report_md: `examples/eval_records/latest/examples/code/report.md`
- eval_report_json: `examples/eval_records/latest/examples/code/report.json`
- golden_md: `examples/golden_result/code/code.md`
- golden_json: `examples/golden_result/code/code.json`

**Signals**
- lowest dimension: `text_fidelity` = 0.9138
- style.code_languages: actual=['xml', 'xml'], expected=['html', 'html']
- json components < 0.98: bbox=0.9083, content=0.9416
- text blocks: paired=6
- lowest text pairs: (idx=None, status=paired, actual=code, expected=code, score=0.4253)
- rules failed: 2 (first: weblogic_rdbms_bean_tag / contains / Missing required phrase: '<weblogic-rdbms-bean>'.)
- fix_hints: Markdown style wrappers (centering/bold/fences), OCR JSON (block ordering, bbox rounding, content normalization), Markdown block segmentation (heading/list/paragraph splits), Example-specific rules/regression
