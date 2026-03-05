# Example evaluation summary

- fail_under: disabled

| Example | Parity | Result→Golden | Ref→Golden | Final | Rules |
|---|---:|---:|---:|---:|---:|
| `GLM-4.5V_Page_1` | 0.857 | None | None | 0.857 | 0/0 fail |
| `GLM-4.5V_Pages_1_2_3` | 0.8739 | 0.8786 | 0.923 | 0.8667 | 0/7 fail |
| `code` | 0.945 | 0.7244 | 0.727 | 0.9457 | 0/0 fail |
| `handwritten` | 0.987 | 0.9706 | 0.955 | 1.0 | 0/1 fail |
| `page` | 0.8467 | 0.5639 | 0.6078 | 0.8396 | 0/0 fail |
| `paper` | 0.9853 | 0.7058 | 0.709 | 0.9853 | 0/0 fail |
| `seal` | 0.9894 | 0.9808 | 0.9808 | 0.9894 | 0/0 fail |
| `table` | 0.9987 | 1.0 | 1.0 | 0.9987 | 0/0 fail |

## Per-example notes

### `GLM-4.5V_Page_1`

- parity.text_fidelity: 0.9086
- parity.critical_structure: 0.7838
- parity.decorative_style: 0.75
- final_overall: 0.857

### `GLM-4.5V_Pages_1_2_3`

- parity.text_fidelity: 0.9381
- parity.critical_structure: 0.7816
- parity.decorative_style: 0.75
- final_overall: 0.8667
- rules:
  - [pass] page1_start: Page 1 start matched expected content.
  - [pass] page1_end: Page 1 end matched expected content.
  - [pass] page2_start: Page 2 start matched expected content.
  - [pass] page2_end: Page 2 end matched expected content.
  - [pass] page3_start: Page 3 start matched expected content.
  - [pass] page3_end: Page 3 end matched expected content.
  - [pass] page2_page3_continuation: Continuation across pages 2 -> 3 matched.

### `code`

- parity.text_fidelity: 0.9248
- parity.critical_structure: 0.979
- parity.decorative_style: 0.95
- final_overall: 0.9457

### `handwritten`

- parity.text_fidelity: 0.9826
- parity.critical_structure: 0.9927
- parity.decorative_style: 1.0
- final_overall: 1.0
- rules:
  - [pass] corrected_phrase: Found required phrase for corrected_phrase.

### `page`

- parity.text_fidelity: 0.8189
- parity.critical_structure: 0.8724
- parity.decorative_style: 1.0
- final_overall: 0.8396

### `paper`

- parity.text_fidelity: 0.981
- parity.critical_structure: 0.9907
- parity.decorative_style: 1.0
- final_overall: 0.9853

### `seal`

- parity.text_fidelity: 1.0
- parity.critical_structure: 0.9697
- parity.decorative_style: 1.0
- final_overall: 0.9894

### `table`

- parity.text_fidelity: 1.0
- parity.critical_structure: 0.9964
- parity.decorative_style: 1.0
- final_overall: 0.9987
