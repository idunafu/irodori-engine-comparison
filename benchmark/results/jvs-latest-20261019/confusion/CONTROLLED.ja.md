# 話者混同：生成条件を揃えた追加診断

元評価のMF/RF合計で誤り純増が大きい6話者＋両実装で全正解の男女各1話者を、介入出力を見る前に選択。
悪化話者：jvs042, jvs049, jvs071, jvs036, jvs061, jvs098。対照：jvs001, jvs002。
各セルは各話者5文章×5 seed。各文章の公式BF16自動生成長を全条件で固定し、同じFP32初期ノイズを注入する。
hann条件では公式torchaudio既定方式で参照を48kHzへ変換してcppへ渡す。ラウドネス・codec・TTSは各実装の処理を維持。
strictはVulkanの内部F16・cooperative matrixを無効にする追加診断。元評価の設定は変更していない。

| 方式 | 条件 | 悪化6話者Top-1 | 対照2話者Top-1 |
|---|---|---:|---:|
| MF | official_bf16_common | 68.00% | 100.00% |
| MF | cpp_f16_kaiser_common | 20.00% | 100.00% |
| MF | cpp_f16_hann_common | 69.33% | 100.00% |
| MF | cpp_fp32_hann_common | 69.33% | 100.00% |
| MF | official_fp32_common | 70.00% | 100.00% |
| MF | cpp_fp32_hann_strict | 70.00% | 100.00% |
| RF | official_bf16_common | 65.33% | 100.00% |
| RF | cpp_f16_kaiser_common | 29.33% | 100.00% |
| RF | cpp_f16_hann_common | 66.67% | 100.00% |
| RF | cpp_fp32_hann_common | 67.33% | 100.00% |
| RF | official_fp32_common | 68.00% | 100.00% |
| RF | cpp_fp32_hann_strict | 68.00% | 100.00% |

| 話者 | 方式 | 条件 | 正解数/25 | 誤認先 |
|---|---|---|---:|---|
| jvs042 | MF | official_bf16_common | 18 | jvs005:5;jvs022:2 |
| jvs049 | MF | official_bf16_common | 25 |  |
| jvs071 | MF | official_bf16_common | 23 | jvs022:2 |
| jvs036 | MF | official_bf16_common | 11 | jvs066:13;jvs053:1 |
| jvs061 | MF | official_bf16_common | 10 | jvs060:9;jvs053:5;jvs057:1 |
| jvs098 | MF | official_bf16_common | 15 | jvs032:10 |
| jvs042 | MF | cpp_f16_kaiser_common | 2 | jvs022:15;jvs005:8 |
| jvs049 | MF | cpp_f16_kaiser_common | 11 | jvs088:8;jvs022:3;jvs074:3 |
| jvs071 | MF | cpp_f16_kaiser_common | 13 | jvs022:12 |
| jvs036 | MF | cpp_f16_kaiser_common | 0 | jvs066:15;jvs082:4;jvs051:3;jvs065:2;jvs055:1 |
| jvs061 | MF | cpp_f16_kaiser_common | 0 | jvs060:17;jvs053:6;jvs057:1;jvs051:1 |
| jvs098 | MF | cpp_f16_kaiser_common | 4 | jvs032:18;jvs099:2;jvs078:1 |
| jvs042 | MF | cpp_f16_hann_common | 18 | jvs005:4;jvs022:3 |
| jvs049 | MF | cpp_f16_hann_common | 25 |  |
| jvs071 | MF | cpp_f16_hann_common | 23 | jvs022:2 |
| jvs036 | MF | cpp_f16_hann_common | 12 | jvs066:12;jvs053:1 |
| jvs061 | MF | cpp_f16_hann_common | 11 | jvs060:8;jvs053:5;jvs057:1 |
| jvs098 | MF | cpp_f16_hann_common | 15 | jvs032:10 |
| jvs042 | MF | cpp_fp32_hann_common | 18 | jvs005:4;jvs022:3 |
| jvs049 | MF | cpp_fp32_hann_common | 25 |  |
| jvs071 | MF | cpp_fp32_hann_common | 22 | jvs022:3 |
| jvs036 | MF | cpp_fp32_hann_common | 12 | jvs066:12;jvs053:1 |
| jvs061 | MF | cpp_fp32_hann_common | 11 | jvs060:8;jvs053:5;jvs057:1 |
| jvs098 | MF | cpp_fp32_hann_common | 16 | jvs032:9 |
| jvs042 | MF | official_fp32_common | 19 | jvs005:3;jvs022:3 |
| jvs049 | MF | official_fp32_common | 25 |  |
| jvs071 | MF | official_fp32_common | 22 | jvs022:3 |
| jvs036 | MF | official_fp32_common | 12 | jvs066:12;jvs053:1 |
| jvs061 | MF | official_fp32_common | 11 | jvs060:8;jvs053:5;jvs057:1 |
| jvs098 | MF | official_fp32_common | 16 | jvs032:9 |
| jvs042 | MF | cpp_fp32_hann_strict | 19 | jvs005:3;jvs022:3 |
| jvs049 | MF | cpp_fp32_hann_strict | 25 |  |
| jvs071 | MF | cpp_fp32_hann_strict | 22 | jvs022:3 |
| jvs036 | MF | cpp_fp32_hann_strict | 12 | jvs066:12;jvs053:1 |
| jvs061 | MF | cpp_fp32_hann_strict | 11 | jvs060:8;jvs053:5;jvs057:1 |
| jvs098 | MF | cpp_fp32_hann_strict | 16 | jvs032:9 |
| jvs042 | RF | official_bf16_common | 17 | jvs005:5;jvs022:3 |
| jvs049 | RF | official_bf16_common | 25 |  |
| jvs071 | RF | official_bf16_common | 25 |  |
| jvs036 | RF | official_bf16_common | 5 | jvs066:18;jvs065:2 |
| jvs061 | RF | official_bf16_common | 4 | jvs060:11;jvs053:10 |
| jvs098 | RF | official_bf16_common | 22 | jvs032:3 |
| jvs042 | RF | cpp_f16_kaiser_common | 3 | jvs022:17;jvs005:5 |
| jvs049 | RF | cpp_f16_kaiser_common | 3 | jvs088:9;jvs074:7;jvs022:5;jvs068:1 |
| jvs071 | RF | cpp_f16_kaiser_common | 22 | jvs022:3 |
| jvs036 | RF | cpp_f16_kaiser_common | 0 | jvs066:18;jvs082:3;jvs065:3;jvs051:1 |
| jvs061 | RF | cpp_f16_kaiser_common | 0 | jvs060:16;jvs053:9 |
| jvs098 | RF | cpp_f16_kaiser_common | 16 | jvs087:4;jvs032:4;jvs089:1 |
| jvs042 | RF | cpp_f16_hann_common | 17 | jvs005:5;jvs022:3 |
| jvs049 | RF | cpp_f16_hann_common | 25 |  |
| jvs071 | RF | cpp_f16_hann_common | 25 |  |
| jvs036 | RF | cpp_f16_hann_common | 5 | jvs066:18;jvs065:2 |
| jvs061 | RF | cpp_f16_hann_common | 5 | jvs053:12;jvs060:7;jvs063:1 |
| jvs098 | RF | cpp_f16_hann_common | 23 | jvs032:2 |
| jvs042 | RF | cpp_fp32_hann_common | 17 | jvs005:5;jvs022:3 |
| jvs049 | RF | cpp_fp32_hann_common | 25 |  |
| jvs071 | RF | cpp_fp32_hann_common | 25 |  |
| jvs036 | RF | cpp_fp32_hann_common | 6 | jvs066:17;jvs065:2 |
| jvs061 | RF | cpp_fp32_hann_common | 6 | jvs053:11;jvs060:7;jvs063:1 |
| jvs098 | RF | cpp_fp32_hann_common | 22 | jvs032:3 |
| jvs042 | RF | official_fp32_common | 17 | jvs005:5;jvs022:3 |
| jvs049 | RF | official_fp32_common | 25 |  |
| jvs071 | RF | official_fp32_common | 25 |  |
| jvs036 | RF | official_fp32_common | 7 | jvs066:16;jvs065:2 |
| jvs061 | RF | official_fp32_common | 6 | jvs053:12;jvs060:7 |
| jvs098 | RF | official_fp32_common | 22 | jvs032:3 |
| jvs042 | RF | cpp_fp32_hann_strict | 17 | jvs005:5;jvs022:3 |
| jvs049 | RF | cpp_fp32_hann_strict | 25 |  |
| jvs071 | RF | cpp_fp32_hann_strict | 25 |  |
| jvs036 | RF | cpp_fp32_hann_strict | 7 | jvs066:16;jvs065:2 |
| jvs061 | RF | cpp_fp32_hann_strict | 6 | jvs053:12;jvs060:7 |
| jvs098 | RF | cpp_fp32_hann_strict | 22 | jvs032:3 |

悪化例に基づく診断集合なので、この集合の集計値を100話者全体の改善率とは扱わない。
全体の混同行列と公式比較はRESULTS.ja.md、介入の個別予測はcontrolled-decisions.csvに保存。
