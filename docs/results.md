# VisionRAG Retrieval Evaluation

- Questions: `data\eval\questions.json`
- Split: **test** (48 questions evaluated)
- Splits in file: 12 dev / 48 test
- Metrics: top-1/3/5 page accuracy, MRR@5 (never calls Ollama)

## Overall (test split)

| Retriever | N | Top-1 (95% CI) | Top-3 (95% CI) | Top-5 (95% CI) | MRR (95% CI) | Avg query latency (s) | Load time (s) | Index build time (s) | Index size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual | 48 | 0.562 (0.417, 0.708) | 0.875 (0.771, 0.958) | 0.979 (0.938, 1.000) | 0.727 (0.639, 0.821) | 0.259 | 0.0 | 520.6 | 21.61 |
| visual_int8 | 48 | 0.562 (0.417, 0.708) | 0.875 (0.771, 0.958) | 0.979 (0.938, 1.000) | 0.727 (0.639, 0.821) | 0.293 | 0.0 | 0.1 | 7.11 |
| visual_mean_pool_64 | 48 | 0.417 (0.271, 0.542) | 0.708 (0.583, 0.833) | 0.854 (0.750, 0.938) | 0.565 (0.447, 0.671) | 0.294 | 0.0 | 0.2 | 0.85 |
| visual_mean_pool_256 | 48 | 0.583 (0.438, 0.729) | 0.854 (0.750, 0.958) | 0.958 (0.896, 1.000) | 0.724 (0.631, 0.823) | 0.287 | 0.0 | 0.6 | 3.36 |
| visual_fp16_local | 48 | 0.562 (0.417, 0.708) | 0.875 (0.771, 0.958) | 0.979 (0.938, 1.000) | 0.727 (0.639, 0.821) | 0.246 | 0.0 | 0.1 | 13.74 |
| text | 48 | 0.396 (0.250, 0.542) | 0.583 (0.458, 0.729) | 0.604 (0.479, 0.750) | 0.477 (0.351, 0.608) | 0.041 | 5.5 | 16.5 | 0.28 |
| ocr_text | 48 | 0.500 (0.354, 0.625) | 0.750 (0.625, 0.854) | 0.792 (0.667, 0.896) | 0.615 (0.497, 0.722) | 0.044 | 4.5 | 211.4 | 0.42 |
| hybrid | 48 | 0.625 (0.479, 0.750) | 0.833 (0.729, 0.938) | 0.938 (0.854, 1.000) | 0.738 (0.634, 0.837) | 0.494 | 0.0 | 731.9 | 22.03 |

*Overall CIs: bootstrap (1000 resamples, seed 42). By-type CIs: Wilson score interval. Hybrid build time = visual (520.6 s) + OCR+Text (211.4 s) = 731.9 s (computed from unrounded values); derived compressed indexes (int8, MeanPool, local) exclude the fp16 build time.*

## By type (test split) — with 95% Wilson score CIs

| Retriever | Type | N | Top-1 (95% CI) | Top-3 (95% CI) | Top-5 (95% CI) | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| visual | text | 6 | 0.667 (0.300, 0.903) | 1.000 (0.610, 1.000) | 1.000 (0.610, 1.000) | 0.806 |
| visual | table | 17 | 0.647 (0.413, 0.827) | 0.882 (0.657, 0.967) | 1.000 (0.816, 1.000) | 0.784 |
| visual | chart | 9 | 0.444 (0.189, 0.733) | 1.000 (0.701, 1.000) | 1.000 (0.701, 1.000) | 0.685 |
| visual | scanned | 16 | 0.500 (0.280, 0.720) | 0.750 (0.505, 0.898) | 0.938 (0.717, 0.989) | 0.661 |
| visual_int8 | text | 6 | 0.667 (0.300, 0.903) | 1.000 (0.610, 1.000) | 1.000 (0.610, 1.000) | 0.806 |
| visual_int8 | table | 17 | 0.647 (0.413, 0.827) | 0.882 (0.657, 0.967) | 1.000 (0.816, 1.000) | 0.784 |
| visual_int8 | chart | 9 | 0.444 (0.189, 0.733) | 1.000 (0.701, 1.000) | 1.000 (0.701, 1.000) | 0.685 |
| visual_int8 | scanned | 16 | 0.500 (0.280, 0.720) | 0.750 (0.505, 0.898) | 0.938 (0.717, 0.989) | 0.661 |
| visual_mean_pool_64 | text | 6 | 0.333 (0.097, 0.700) | 0.667 (0.300, 0.903) | 0.833 (0.436, 0.970) | 0.542 |
| visual_mean_pool_64 | table | 17 | 0.588 (0.360, 0.784) | 0.882 (0.657, 0.967) | 0.941 (0.730, 0.990) | 0.698 |
| visual_mean_pool_64 | chart | 9 | 0.556 (0.267, 0.811) | 0.889 (0.565, 0.980) | 1.000 (0.701, 1.000) | 0.726 |
| visual_mean_pool_64 | scanned | 16 | 0.188 (0.066, 0.430) | 0.438 (0.231, 0.668) | 0.688 (0.444, 0.858) | 0.341 |
| visual_mean_pool_256 | text | 6 | 0.500 (0.188, 0.812) | 0.667 (0.300, 0.903) | 1.000 (0.610, 1.000) | 0.650 |
| visual_mean_pool_256 | table | 17 | 0.647 (0.413, 0.827) | 1.000 (0.816, 1.000) | 1.000 (0.816, 1.000) | 0.804 |
| visual_mean_pool_256 | chart | 9 | 0.667 (0.354, 0.879) | 0.889 (0.565, 0.980) | 0.889 (0.565, 0.980) | 0.759 |
| visual_mean_pool_256 | scanned | 16 | 0.500 (0.280, 0.720) | 0.750 (0.505, 0.898) | 0.938 (0.717, 0.989) | 0.648 |
| visual_fp16_local | text | 6 | 0.667 (0.300, 0.903) | 1.000 (0.610, 1.000) | 1.000 (0.610, 1.000) | 0.806 |
| visual_fp16_local | table | 17 | 0.647 (0.413, 0.827) | 0.882 (0.657, 0.967) | 1.000 (0.816, 1.000) | 0.784 |
| visual_fp16_local | chart | 9 | 0.444 (0.189, 0.733) | 1.000 (0.701, 1.000) | 1.000 (0.701, 1.000) | 0.685 |
| visual_fp16_local | scanned | 16 | 0.500 (0.280, 0.720) | 0.750 (0.505, 0.898) | 0.938 (0.717, 0.989) | 0.661 |
| text | text | 6 | 0.667 (0.300, 0.903) | 0.833 (0.436, 0.970) | 0.833 (0.436, 0.970) | 0.722 |
| text | table | 17 | 0.529 (0.310, 0.738) | 0.882 (0.657, 0.967) | 0.882 (0.657, 0.967) | 0.667 |
| text | chart | 9 | 0.667 (0.354, 0.879) | 0.889 (0.565, 0.980) | 1.000 (0.701, 1.000) | 0.806 |
| text | scanned | 16 | 0.000 (0.000, 0.194) | 0.000 (0.000, 0.194) | 0.000 (0.000, 0.194) | 0.000 |
| ocr_text | text | 6 | 0.667 (0.300, 0.903) | 0.667 (0.300, 0.903) | 0.833 (0.436, 0.970) | 0.708 |
| ocr_text | table | 17 | 0.529 (0.310, 0.738) | 0.882 (0.657, 0.967) | 0.882 (0.657, 0.967) | 0.667 |
| ocr_text | chart | 9 | 0.667 (0.354, 0.879) | 0.889 (0.565, 0.980) | 1.000 (0.701, 1.000) | 0.806 |
| ocr_text | scanned | 16 | 0.312 (0.142, 0.556) | 0.562 (0.332, 0.769) | 0.562 (0.332, 0.769) | 0.417 |
| hybrid | text | 6 | 0.833 (0.436, 0.970) | 0.833 (0.436, 0.970) | 1.000 (0.610, 1.000) | 0.875 |
| hybrid | table | 17 | 0.765 (0.527, 0.904) | 0.882 (0.657, 0.967) | 0.941 (0.730, 0.990) | 0.828 |
| hybrid | chart | 9 | 0.667 (0.354, 0.879) | 1.000 (0.701, 1.000) | 1.000 (0.701, 1.000) | 0.815 |
| hybrid | scanned | 16 | 0.375 (0.185, 0.614) | 0.688 (0.444, 0.858) | 0.875 (0.640, 0.965) | 0.547 |

## Non-scanned questions (n=32)

| Retriever | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| visual | 19/32 (59.4%) | 30/32 (93.8%) | 32/32 (100%) |
| text | 19/32 (59.4%) | 28/32 (87.5%) | 29/32 (90.6%) |
| ocr_text | 19/32 (59.4%) | 27/32 (84.4%) | 29/32 (90.6%) |
| hybrid | 24/32 (75.0%) | 29/32 (90.6%) | 31/32 (96.9%) |

## Paired Comparisons vs Visual (Top-1 Rank)

| Retriever | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| visual_int8 | 0 | 0 | 48 | 1.000 |
| visual_mean_pool_64 | 12 | 5 | 31 | 0.143 |
| visual_mean_pool_256 | 5 | 6 | 37 | 1.000 |
| visual_fp16_local | 0 | 0 | 48 | 1.000 |
| text | 12 | 4 | 32 | 0.077 |
| ocr_text | 9 | 6 | 33 | 0.607 |
| hybrid | 3 | 6 | 39 | 0.508 |

## Paired Comparisons vs Visual (Top-3 Rank)

| Retriever | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| visual_int8 | 0 | 0 | 48 | 1.000 |
| visual_mean_pool_64 | 10 | 2 | 36 | 0.039 |
| visual_mean_pool_256 | 4 | 3 | 41 | 1.000 |
| visual_fp16_local | 0 | 0 | 48 | 1.000 |
| text | 16 | 2 | 30 | 0.001 |
| ocr_text | 12 | 6 | 30 | 0.238 |
| hybrid | 6 | 4 | 38 | 0.754 |

## Paired Comparisons vs Visual (Top-5 Rank)

| Retriever | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| visual_int8 | 0 | 0 | 48 | 1.000 |
| visual_mean_pool_64 | 6 | 0 | 42 | 0.031 |
| visual_mean_pool_256 | 2 | 1 | 45 | 1.000 |
| visual_fp16_local | 0 | 0 | 48 | 1.000 |
| text | 18 | 0 | 30 | <0.001 |
| ocr_text | 10 | 1 | 37 | 0.012 |
| hybrid | 2 | 0 | 46 | 0.500 |

## Paired Comparisons vs Visual on Scanned Questions (n=16)


### Top-1

| Challenger | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| ocr_text | 5 | 2 | 9 | 0.453 |
| text | 8 | 0 | 8 | 0.008 |

### Top-3

| Challenger | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| ocr_text | 7 | 4 | 5 | 0.549 |
| text | 12 | 0 | 4 | <0.001 |

### Top-5

| Challenger | Visual Win | Visual Loss | Tie | Sign-test p |
| --- | ---: | ---: | ---: | ---: |
| ocr_text | 7 | 1 | 8 | 0.070 |
| text | 15 | 0 | 1 | <0.001 |

## Per-question gold rank (test split)

| ID | Type | Gold | visual | visual_int8 | visual_mean_pool_64 | visual_mean_pool_256 | visual_fp16_local | text | ocr_text | hybrid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q001 | table | `paper1_page_0008` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q004 | chart | `paper1_page_0003` | 3 | 3 | 3 | MISS | 3 | 1 | 1 | 1 |
| q005 | table | `paper1_page_0006` | 1 | 1 | 3 | 1 | 1 | 1 | 1 | 1 |
| q006 | scanned | `BERT_paper01_page_0002` | 1 | 1 | 3 | 1 | 1 | MISS | 2 | 1 |
| q007 | scanned | `BERT_paper01_page_0003` | 2 | 2 | 3 | 3 | 2 | MISS | MISS | MISS |
| q008 | scanned | `BERT_paper01_page_0003` | 1 | 1 | MISS | 2 | 1 | MISS | MISS | 3 |
| q009 | scanned | `BERT_paper01_page_0003` | 1 | 1 | 1 | 1 | 1 | MISS | MISS | 4 |
| q010 | scanned | `BERT_paper01_page_0004` | 1 | 1 | 4 | 1 | 1 | MISS | 1 | 1 |
| q012 | scanned | `BERT_paper01_page_0005` | 4 | 4 | 4 | 3 | 4 | MISS | 1 | 2 |
| q014 | scanned | `BERT_paper01_page_0005` | 4 | 4 | 4 | 4 | 4 | MISS | 1 | 2 |
| q015 | scanned | `BERT_paper01_page_0006` | 3 | 3 | MISS | MISS | 3 | MISS | MISS | 4 |
| q017 | scanned | `BERT_paper01_page_0007` | MISS | MISS | MISS | 4 | MISS | MISS | 3 | MISS |
| q018 | scanned | `BERT_paper01_page_0007` | 1 | 1 | 5 | 2 | 1 | MISS | MISS | 1 |
| q019 | scanned | `BERT_paper01_page_0008` | 4 | 4 | MISS | 5 | 4 | MISS | 2 | 3 |
| q020 | scanned | `BERT_paper01_page_0008` | 2 | 2 | MISS | 1 | 2 | MISS | MISS | 4 |
| q022 | scanned | `BERT_paper01_page_0009` | 1 | 1 | 3 | 1 | 1 | MISS | 1 | 1 |
| q023 | scanned | `BERT_paper01_page_0009` | 1 | 1 | 1 | 1 | 1 | MISS | 3 | 1 |
| q024 | scanned | `BERT_paper01_page_0015` | 2 | 2 | 1 | 1 | 2 | MISS | MISS | 3 |
| q025 | scanned | `BERT_paper01_page_0016` | 1 | 1 | 2 | 1 | 1 | MISS | 1 | 1 |
| q026 | text | `paper1_page_0002` | 1 | 1 | 2 | 1 | 1 | 3 | 4 | 1 |
| q027 | chart | `paper1_page_0003` | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| q028 | text | `paper1_page_0005` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q029 | text | `paper1_page_0007` | 1 | 1 | MISS | 2 | 1 | 1 | 1 | 1 |
| q031 | table | `paper1_page_0009` | 1 | 1 | MISS | 2 | 1 | MISS | MISS | 2 |
| q032 | table | `paper1_page_0010` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q033 | chart | `paper1_page_0013` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q035 | chart | `paper1_page_0014` | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 |
| q036 | text | `paper2_page_0003` | 2 | 2 | 4 | 5 | 2 | MISS | MISS | 4 |
| q037 | table | `paper2_page_0001` | 2 | 2 | 1 | 3 | 2 | MISS | MISS | MISS |
| q038 | table | `paper2_page_0005` | 2 | 2 | 3 | 1 | 2 | 2 | 2 | 1 |
| q039 | chart | `paper2_page_0001` | 2 | 2 | 5 | 3 | 2 | 4 | 4 | 3 |
| q041 | text | `paper2_page_0008` | 3 | 3 | 2 | 5 | 3 | 1 | 1 | 1 |
| q042 | table | `paper2_page_0012` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q043 | table | `paper2_page_0011` | 1 | 1 | 1 | 1 | 1 | 3 | 3 | 1 |
| q044 | chart | `paper2_page_0008` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q045 | table | `paper2_page_0006` | 1 | 1 | 3 | 1 | 1 | 1 | 1 | 1 |
| q048 | table | `f_paper01_page_0001` | 4 | 4 | 3 | 2 | 4 | 3 | 3 | 4 |
| q050 | table | `f_paper01_page_0001` | 3 | 3 | 1 | 2 | 3 | 3 | 3 | 3 |
| q051 | chart | `f_paper01_page_0001` | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 |
| q052 | table | `f_paper01_page_0001` | 4 | 4 | 3 | 1 | 4 | 3 | 3 | 1 |
| q053 | table | `f_paper02_page_0001` | 2 | 2 | 5 | 3 | 2 | 2 | 2 | 1 |
| q054 | chart | `f_paper02_page_0002` | 3 | 3 | 1 | 2 | 3 | 1 | 1 | 2 |
| q055 | table | `f_paper02_page_0003` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q056 | table | `f_paper02_page_0004` | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| q057 | table | `f_paper02_page_0005` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q058 | table | `f_paper02_page_0006` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q059 | text | `f_paper02_page_0007` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q060 | chart | `f_paper02_page_0007` | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 1 |
