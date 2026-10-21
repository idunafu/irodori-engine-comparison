# speech.cpp更新後のJVS100話者評価と識別率低下の原因

2026年10月19日に固定したspeech.cpp `eacabcd891fefc2c3de3ca2479ff84e60f2679aa`、ggml `a655cf6910770404213bc338e622cd82b08ca277` の追加評価です。このページの「更新後」はこの固定版を指します。公式BF16と旧speech.cpp F16の結果は[既存のJVS評価](../quality-20261019/RESULTS.ja.md#jvs話者再現性と参照長)から再利用しました。

更新後の話者類似度は公式に近づきましたが、100話者識別Top-1はMFで2.64ポイント、RFで3.40ポイント低い点推定でした。悪化の大きい6話者の追加診断では、生成長・初期ノイズを揃えても差が残り、参照音声のリサンプルを公式相当のHannに揃えると差がほぼ消えました。Hannによる100話者全体の改善量は未測定です。

## 100話者・1クリップ参照の比較

100話者×5文章×5 seed×MF/RF。更新後の新規生成は5,000音声、公式と旧cppは各5,000音声の生成・CAM++・Whisper採点結果を再利用しました。参照はnonpara30のID順先頭1クリップ、合成対象はparallel100先頭5発話、照合centroidは続く別の10発話から作成し、全条件で共通です。captionは空、各実装の自動長予測とnative seed、末尾トリム・透かしなし。新規生成・採点の失敗はありません。

類似度はCAM++ cosine×100であり、確率や識別正答率ではありません。Top-1は100人の固定centroidで正解話者が最大スコアになった割合です。

| 方式 | 実装 | 類似度（cosine×100） | 識別Top-1 | Whisper CER |
|---|---|---:|---:|---:|
| MF | 公式BF16 | 76.45% | 88.64% | 2.98% |
| MF | 旧cpp F16 | 75.66% | 84.52% | 2.97% |
| MF | 更新後cpp F16 | 76.49% | 86.00% | 2.97% |
| RF | 公式BF16 | 76.84% | 88.96% | 2.87% |
| RF | 旧cpp F16 | 75.92% | 84.28% | 2.87% |
| RF | 更新後cpp F16 | 76.74% | 85.56% | 2.86% |

更新後−公式のTop-1差の95%区間はMF [−5.56, +0.04]、RF [−6.44, −0.40]ポイントです。100話者を単位とする対応ありbootstrap 2,000回で求めています。旧版からはMF +1.48、RF +1.28ポイント改善しましたが、49コミットとggml更新を含む比較なので、単一修正だけの効果とは扱いません。

[詳細レポート](evaluation/RESULTS.ja.md)、[seed別値](evaluation/per-seed.csv)、[平均・母標準偏差](evaluation/summary.csv)、[対応あり差と95%区間](evaluation/latest-paired-differences.csv)、[完了件数](evaluation/coverage.json)を保存しています。ONNXの再測定はこの追加評価に含みません。

## 誰を誰と間違えたか

各話者・方式につき25音声です。主な誤認先を示します。

| 方式 | 正解→誤認先 | 公式の誤認数 | 更新後cppの誤認数 |
|---|---|---:|---:|
| MF | jvs042 → jvs005 | 1 | 13 |
| MF | jvs042 → jvs022 | 3 | 11 |
| MF | jvs049 → jvs088 | 0 | 13 |
| MF | jvs071 → jvs022 | 4 | 16 |
| MF | jvs098 → jvs032 | 10 | 15 |
| RF | jvs042 → jvs022 | 1 | 19 |
| RF | jvs049 → jvs022 | 0 | 9 |
| RF | jvs049 → jvs088 | 0 | 8 |
| RF | jvs049 → jvs074 | 0 | 6 |
| RF | jvs092 → jvs055 | 14 | 23 |

MFは「公式のみ正解」157件と「cppのみ正解」91件で誤り純増66件、RFは165件と80件で純増85件でした。jvs042・jvs049の2話者が純増の約半分を占めます。平均類似度が近くても、競合話者のスコアが上がればTop-1は下がります。

参照の実音声100件と合成対象に対応する実音声500件は、同じ照合centroidで600件すべて正しく識別されました。入力の取り違えを支持する結果ではありませんが、合成音声に対する評価モデルの偏りまで否定する検証ではありません。

[全混同ペア](confusion/confusion-comparison.csv)、[話者別の比較](confusion/speaker-gaps.csv)、[15,000音声の予測・順位・スコア](confusion/decisions.csv)、[公式との対応関係](confusion/paired-outcomes.csv)、[実音声の識別結果](confusion/natural-decisions.csv)、[混同の詳細レポート](confusion/RESULTS.ja.md)を参照してください。

## 共通生成長・初期ノイズによる原因診断

悪化の大きいjvs042・049・071・036・061・098と、全件正解の対照jvs001・002を介入出力の前に固定しました。8話者×5文章×5 seed×MF/RF×6条件＝2,400音声で、全件生成・採点が完了しています。全6条件で公式BF16の生成フレーム数と同じfloat32初期ノイズを使用しました。

Hann条件では、公式torchaudio既定方式で24kHzの参照を48kHz float32 WAVへ変換してからcppへ渡します。ラウドネス正規化・codec・TTSは各実装の処理を維持しました。strict FP32条件はFP32重みに加え、Vulkan内部F16とcooperative matrixを無効にします。

| 条件 | 悪化6話者：MF Top-1 | 悪化6話者：RF Top-1 | 対照2話者：MF/RF |
|---|---:|---:|---:|
| 公式BF16 | 68.00% | 65.33% | 100.00% |
| cpp F16・現行Kaiser | 20.00% | 29.33% | 100.00% |
| cpp F16・公式相当Hann | 69.33% | 66.67% | 100.00% |
| cpp FP32重み・Hann・通常Vulkan | 69.33% | 67.33% | 100.00% |
| 公式FP32 | 70.00% | 68.00% | 100.00% |
| cpp FP32重み・Hann・内部FP32 | 70.00% | 68.00% | 100.00% |

調べた悪化例ではリサンプル方式の違いが低下の主因でした。F16重みを維持したままHannへ揃えると公式BF16との差がほぼ消え、内部演算までFP32へ揃えると全400組で公式FP32と予測話者が一致しました。集計した識別率の一致は波形の完全一致を意味しません。

悪化例を選んだ診断なので、この識別率や改善幅を100話者全体へ当てはめません。Hannの一般的な優位性、他の話者埋め込みモデルでの優位性、聴感上の自然さは検証していません。現行cppのリサンプルはKaiserBest（幅64）、公式相当Hannは幅6です。リサンプル速度の直接比較は未実施です。

[診断の詳細](confusion/CONTROLLED.ja.md)、[2,400音声の予測](confusion/controlled-decisions.csv)、[集計](confusion/controlled-summary.csv)、[話者別](confusion/controlled-per-speaker.csv)、[対応あり差と区間](confusion/controlled-differences.csv)、[対象選択](confusion/selection.json)、[共通長・ノイズの検証](confusion/validation.json)を保存しています。

## 保存物と再現方法

| フォルダー・ファイル | 内容 |
|---|---|
| evaluation/ | 100話者の比較表、seed別値、対応あり区間、coverage、測定時manifest・ビルド記録・終了状態 |
| confusion/ | 全件の混同結果、固定対象、実音声の識別、共通条件診断、manifest・終了状態 |
| [provenance.json](provenance.json) | 公開した34ファイルの保存元、元データと公開コピーのSHA-256、変換内容 |
| [validation.json](validation.json) | 公開コピー照合、CSV再集計、パス除去と条件一致の検証結果 |

実行・集計・公開用コピー作成のコードは[benchmark/quality/jvs_latest/](../../quality/jvs_latest/README.ja.md)にあります。音声、コーパス、モデル、全centroidスコア・埋め込み、SQLite、ログ、参照cache、ビルド成果物はGit管理外の`artifacts/`へ保持します。コーパスZIPも公開しません。

公開コピーのパスはリポジトリルートからの相対表記へ変換し、外部Vulkan loaderのインストール場所は`${Vulkan_LIBRARY}`へ置換しました。数値指標・音声やノイズのハッシュは変更していません。公開manifestに残るコードや入力のハッシュは測定時の元ファイルを識別する値で、パスを変換した公開manifest自体のハッシュではありません。終了済み元runのmanifestと再開状態は変更していません。

公開コードの`run.py`はVulkan loaderを自動検出するよう整理したため、測定時のコードハッシュとは異なります。コード変更後に元runへ再開しようとすると照合で拒否します。再生成は[手順](../../quality/jvs_latest/README.ja.md)に従い別の出力先へ記録してください。

公開CSVと保存物の検証はPython標準ライブラリだけで実行できます。元音声・モデル・GPUは不要です。

```bash
python3 benchmark/quality/jvs_latest/publish.py --check
```

元runから公開コピーを作り直す場合は同じコマンドの`--check`を省きます。元runの再集計はREADMEに記載した各スクリプトを使用します。公開結果は再集計から自動更新されません。
