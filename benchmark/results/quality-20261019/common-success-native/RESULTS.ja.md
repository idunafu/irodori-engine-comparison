# 共通の生成成功文による比較

実用40文から、全5構成・全5 seedで生成できた32文を共通に採用。各構成160音声。
MFは公式BF16・ONNX FP16・speech.cpp Vulkan F16、RFは公式BF16・speech.cpp Vulkan F16。
除外はexisting:p13〜p20の8文。ASRの良し悪しによる選別は行っていない。
既存のnative結果のみを使用し、音声生成・ASRの再実行は行っていない。
値は5 seed平均。誤り率は各seedで誤り数合計／参照文字数合計。破綻候補は誤り率30%超。

| 構成 | Qwen誤り率 | Reazon誤り率 | Qwen破綻候補率 | Reazon破綻候補率 |
|---|---:|---:|---:|---:|
| official-mf-bf16 | 4.54% | 11.37% | 14.37% | 21.25% |
| onnx-mf-fp16 | 4.72% | 14.67% | 15.00% | 26.25% |
| speech-mf-f16 | 4.45% | 13.16% | 13.75% | 25.00% |
| official-rf-bf16 | 2.05% | 4.88% | 13.12% | 15.62% |
| speech-rf-f16 | 2.43% | 10.15% | 13.75% | 22.50% |

公式との差（ポイント）と文章単位の対応ありbootstrap 95%区間：

| 構成 | 指標 | 差 | 95%区間 |
|---|---|---:|---:|
| onnx-mf-fp16 | qwen_heard_error_rate | +0.18 | [-1.22, +1.57] |
| onnx-mf-fp16 | qwen_breakdown_rate | +0.63 | [+0.00, +1.88] |
| onnx-mf-fp16 | reazon_heard_error_rate | +3.30 | [+0.46, +6.48] |
| onnx-mf-fp16 | reazon_breakdown_rate | +5.00 | [+0.63, +10.00] |
| speech-mf-f16 | qwen_heard_error_rate | -0.09 | [-1.50, +1.27] |
| speech-mf-f16 | qwen_breakdown_rate | -0.62 | [-2.52, +1.25] |
| speech-mf-f16 | reazon_heard_error_rate | +1.79 | [-0.32, +4.36] |
| speech-mf-f16 | reazon_breakdown_rate | +3.75 | [-3.14, +11.88] |
| speech-rf-f16 | qwen_heard_error_rate | +0.38 | [+0.05, +0.80] |
| speech-rf-f16 | qwen_breakdown_rate | +0.63 | [+0.00, +1.88] |
| speech-rf-f16 | reazon_heard_error_rate | +5.27 | [+1.12, +11.24] |
| speech-rf-f16 | reazon_breakdown_rate | +6.88 | [+1.25, +13.75] |

文間の声の一貫性は全構成・全seedでembeddingを取得できた共通25文に限定。
JVSによる話者再現性の評価とは異なり、参照話者への類似度ではない。

| 構成 | 文間の声の一貫性（cosine） |
|---|---:|
| official-mf-bf16 | 0.921541 |
| onnx-mf-fp16 | 0.913974 |
| speech-mf-f16 | 0.924387 |
| official-rf-bf16 | 0.931521 |
| speech-rf-f16 | 0.931535 |

個別の音声に対する自動判定であり、人が確認した誤りや聴感品質の同等性は意味しない。
この32文の結果は30秒制限で拒否された8文には一般化しない。
元の主比較のASR採点はすべて成功。追加のcommon-noise診断で起きた3件の採点失敗はこの集計に含まれない。
