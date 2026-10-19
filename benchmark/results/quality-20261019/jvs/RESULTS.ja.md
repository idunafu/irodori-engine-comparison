# Irodori-TTS 自動品質評価

評価対象: jvs.
JSUT・試聴・caption指示追従は対象外。固定版の実装を同一評価器で比較。
値は5 seedの平均と母標準偏差。生成失敗は読み評価で空出力、話者Top-1で不正解として扱う。
CAM++類似度は採点可能な音声のみ。自動指標だけで聴感上の自然さや同等性は判定しない。

| データ | 実装 | モード | 指標 | 平均 | 母標準偏差 |
|---|---|---|---|---:|---:|
| jvs | official-mf-bf16 | native | camplus_cosine | 0.810473 | 0.000598 |
| jvs | official-mf-bf16 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | official-mf-bf16 | native | speaker_top1 | 0.961600 | 0.002245 |
| jvs | official-mf-fp32 | native | camplus_cosine | 0.798795 | 0.000735 |
| jvs | official-mf-fp32 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | official-mf-fp32 | native | speaker_top1 | 0.969000 | 0.004062 |
| jvs | official-rf-bf16 | native | camplus_cosine | 0.817514 | 0.000549 |
| jvs | official-rf-bf16 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | official-rf-bf16 | native | speaker_top1 | 0.965800 | 0.002657 |
| jvs | official-rf-fp32 | native | camplus_cosine | 0.805485 | 0.001381 |
| jvs | official-rf-fp32 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | official-rf-fp32 | native | speaker_top1 | 0.967000 | 0.003317 |
| jvs | onnx-mf-fp16 | native | camplus_cosine | 0.809317 | 0.001238 |
| jvs | onnx-mf-fp16 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | onnx-mf-fp16 | native | speaker_top1 | 0.962400 | 0.002746 |
| jvs | onnx-mf-fp32 | native | camplus_cosine | 0.796656 | 0.001560 |
| jvs | onnx-mf-fp32 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | onnx-mf-fp32 | native | speaker_top1 | 0.967500 | 0.003162 |
| jvs | speech-mf-f16 | native | camplus_cosine | 0.808789 | 0.001447 |
| jvs | speech-mf-f16 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | speech-mf-f16 | native | speaker_top1 | 0.954200 | 0.001939 |
| jvs | speech-mf-fp32 | native | camplus_cosine | 0.795025 | 0.001981 |
| jvs | speech-mf-fp32 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | speech-mf-fp32 | native | speaker_top1 | 0.963000 | 0.001871 |
| jvs | speech-rf-f16 | native | camplus_cosine | 0.817341 | 0.000798 |
| jvs | speech-rf-f16 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | speech-rf-f16 | native | speaker_top1 | 0.955200 | 0.001806 |
| jvs | speech-rf-fp32 | native | camplus_cosine | 0.805489 | 0.001264 |
| jvs | speech-rf-fp32 | native | generation_failure_rate | 0.000000 | 0.000000 |
| jvs | speech-rf-fp32 | native | speaker_top1 | 0.963500 | 0.002550 |

公式版との差・95% paired cluster bootstrap区間は comparisons.csv、種別別・成功例のみの集計は summary.csv を参照。
JKYBは漢字読みの組、JVSは話者を再標本化する。実用文は文章を単位とし、異なる文章間の声の一貫性を測る。
生成失敗の分類・件数は failures.csv、採点可能件数は coverage.json と per-seed.csv に保存。
既存速度比較とコード・モデル・runtimeを共有するが、生成条件や入力集合が異なる速度値は直接結合しない。
