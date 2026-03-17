# Example evaluation summary

- fail_under: disabled
- inflation_warn_threshold: 0.15

## How to interpret the scores

- `parity_overall`: `result` vs `reference_result` (upstream parity/regression signal).
- `quality_overall`: `result_to_golden.overall` when available; otherwise `parity_overall` (absolute usefulness proxy).
- `final_overall`: parity-first score with a small golden correction (see `config/policy.yaml`).
- `final_minus_quality`: diagnostic for parity-first inflation; large values usually mean the upstream baseline is also far from golden.

| Example | Parity | Quality | Result→Golden | Ref→Golden | Final | Final-Quality | Rules | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `GLM-4.5V_Page_1` | 0.9224 | 0.8679 | 0.8679 | 0.9308 | 0.9067 | 0.0387 | 0/0 fail |  |
| `GLM-4.5V_Pages_1_2_3` | 0.9627 | 0.8463 | 0.8463 | 0.8743 | 0.9557 | 0.1094 | 0/7 fail |  |
| `code` | 0.9384 | 0.8382 | 0.8382 | 0.8406 | 0.9105 | 0.0722 | 2/3 fail |  |
| `handwritten` | 0.9875 | 0.8034 | 0.8034 | 0.7967 | 1.0 | 0.1966 | 0/1 fail | inflation |
| `page` | 0.8836 | 0.4837 | 0.4837 | 0.5148 | 0.8254 | 0.3417 | 1/1 fail | inflation |
| `paper` | 0.9864 | 0.7326 | 0.7326 | 0.7343 | 0.9864 | 0.2538 | 0/2 fail | inflation |
| `seal` | 0.9894 | 0.9875 | 0.9875 | 0.9875 | 0.9894 | 0.0019 | 0/0 fail |  |
| `table` | 0.9987 | 1.0 | 1.0 | 1.0 | 0.9987 | -0.0013 | 0/0 fail |  |

## Per-example notes

### `GLM-4.5V_Page_1`

- parity.text_fidelity: 0.9106
- parity.critical_structure: 0.9672
- parity.decorative_style: 0.75
- quality_overall: 0.8679
- final_overall: 0.9067
- final_minus_quality: 0.0387

### `GLM-4.5V_Pages_1_2_3`

- parity.text_fidelity: 0.968
- parity.critical_structure: 0.9841
- parity.decorative_style: 0.75
- quality_overall: 0.8463
- final_overall: 0.9557
- final_minus_quality: 0.1094
- rules:
  - [pass] page1_start: Page 1 start matched expected content.
  - [pass] page1_end: Page 1 end matched expected content.
  - [pass] page2_start: Page 2 start matched expected content.
  - [pass] page2_end: Page 2 end matched expected content.
  - [pass] page3_start: Page 3 start matched expected content.
  - [pass] page3_end: Page 3 end matched expected content.
  - [pass] page2_page3_continuation: Continuation across pages 2 -> 3 matched.

### `code`

- parity.text_fidelity: 0.9138
- parity.critical_structure: 0.979
- parity.decorative_style: 0.95
- quality_overall: 0.8382
- final_overall: 0.9105
- final_minus_quality: 0.0722
- rules:
  - [pass] local_jndi_name_tag: Found required phrase for local_jndi_name_tag.
  - [fail] weblogic_rdbms_bean_tag: Missing required phrase: '<weblogic-rdbms-bean>'.
  - [fail] key_cache_size_value: Missing required phrase: '<key-cache-size>10</key-cache-size>'.

### `handwritten`

- parity.text_fidelity: 0.9835
- parity.critical_structure: 0.9927
- parity.decorative_style: 1.0
- quality_overall: 0.8034
- final_overall: 1.0
- final_minus_quality: 0.1966
- warning: final_overall significantly exceeds quality_overall (parity-first inflation). (value=0.1966, threshold=0.15)
- rules:
  - [pass] corrected_phrase: Found required phrase for corrected_phrase.

### `page`

- parity.text_fidelity: 0.8267
- parity.critical_structure: 0.9645
- parity.decorative_style: 1.0
- quality_overall: 0.4837
- final_overall: 0.8254
- final_minus_quality: 0.3417
- warning: final_overall significantly exceeds quality_overall (parity-first inflation). (value=0.3417, threshold=0.15)
- rules:
  - [fail] glue_strength_constant: Missing required phrase: '0.2\\mathrm{N} / \\mathrm{mm}^{2}'.

### `paper`

- parity.text_fidelity: 0.9828
- parity.critical_structure: 0.9907
- parity.decorative_style: 1.0
- quality_overall: 0.7326
- final_overall: 0.9864
- final_minus_quality: 0.2538
- warning: final_overall significantly exceeds quality_overall (parity-first inflation). (value=0.2538, threshold=0.15)
- rules:
  - [pass] not_divisible_by_Q: Found required phrase for not_divisible_by_Q.
  - [pass] laplacian_operator: Found required phrase for laplacian_operator.

### `seal`

- parity.text_fidelity: 1.0
- parity.critical_structure: 0.9697
- parity.decorative_style: 1.0
- quality_overall: 0.9875
- final_overall: 0.9894
- final_minus_quality: 0.0019

### `table`

- parity.text_fidelity: 1.0
- parity.critical_structure: 0.9964
- parity.decorative_style: 1.0
- quality_overall: 1.0
- final_overall: 0.9987
- final_minus_quality: -0.0013
