# 最新speech.cpp：JVS100話者・1クリップ参照

最新コミット `eacabcd891fefc2c3de3ca2479ff84e60f2679aa`、ggml `a655cf6910770404213bc338e622cd82b08ca277`。
100話者×5文章×5 seed×MF/RF＝新規5,000音声。speech.cppはF16重み・通常Vulkan。
公式CUDA BF16・旧speech.cpp F16の各5,000音声および採点結果を既存評価から再利用した。
各実装の自動長予測・native seed処理を使用。captionは空、末尾トリムと透かしは無効。
参照は既存と同じnonpara30先頭1クリップ。照合用parallel100の10音声から作った100話者centroidも再利用。

CAM++ cosine×100を類似度％として表示する。Top-1は100話者識別の正答率。

| 方式 | 実装 | 類似度 | 話者Top-1 | Whisper CER |
|---|---|---:|---:|---:|
| MF | 公式BF16（再利用） | 76.45% | 88.64% | 2.98% |
| MF | 旧speech.cpp F16（再利用） | 75.66% | 84.52% | 2.97% |
| MF | 最新speech.cpp F16 | 76.49% | 86.00% | 2.97% |
| RF | 公式BF16（再利用） | 76.84% | 88.96% | 2.87% |
| RF | 旧speech.cpp F16（再利用） | 75.92% | 84.28% | 2.87% |
| RF | 最新speech.cpp F16 | 76.74% | 85.56% | 2.86% |

最新版−比較基準の差。単位は％ポイント、95%区間は話者単位の対応bootstrap（2,000回）。

| 方式 | 基準 | 指標 | 差 | 95%区間 |
|---|---|---|---:|---:|
| MF | official-mf-bf16 | camplus_cosine | +0.041 | [-0.353, +0.408] |
| MF | official-mf-bf16 | speaker_top1 | -2.640 | [-5.560, +0.040] |
| MF | official-mf-bf16 | whisper_cer | -0.009 | [-0.104, +0.084] |
| MF | previous-mf-f16 | camplus_cosine | +0.832 | [+0.542, +1.140] |
| MF | previous-mf-f16 | speaker_top1 | +1.480 | [+0.119, +3.361] |
| MF | previous-mf-f16 | whisper_cer | +0.000 | [-0.068, +0.070] |
| RF | official-rf-bf16 | camplus_cosine | -0.096 | [-0.521, +0.352] |
| RF | official-rf-bf16 | speaker_top1 | -3.400 | [-6.440, -0.399] |
| RF | official-rf-bf16 | whisper_cer | -0.013 | [-0.067, +0.041] |
| RF | previous-rf-f16 | camplus_cosine | +0.825 | [+0.552, +1.097] |
| RF | previous-rf-f16 | speaker_top1 | +1.280 | [-0.640, +3.280] |
| RF | previous-rf-f16 | whisper_cer | -0.005 | [-0.038, +0.028] |

生成失敗・採点失敗はcoverage.jsonとfailures.csvを参照。各seedの値・母標準偏差はper-seed.csvとsummary.csv。
修正前からは49コミットとggmlの更新を含むため、差を単一修正だけの効果とは断定しない。
自動評価であり、聴感上の自然さや品質同等性は判定しない。
