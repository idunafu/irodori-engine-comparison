# 自動品質評価

[2026年10月19日集計の結果](../results/quality-20261019/RESULTS.ja.md)に、JKYB・実用文・JVSの主評価と、RFの読み差、1クリップ参照、入力サンプルレート、F16重みの追加診断をまとめています。読み・実用文runは生成失敗200件と追加診断のQwen採点失敗3件を記録して終了し、JVSとサンプルレートの追加評価は完了しました。全件と共通成功文の結果は分けて掲載しています。

公式CUDA版、ONNX CUDA版、speech.cpp Vulkan版を既存主比較の固定コード・重み・runtimeで評価します。MFは3実装、RFは公式版とspeech.cppを比較します。JSUT、試聴、caption指示追従の採点は含みません。

## 評価条件

実用精度は公式BF16、ONNX FP16、speech.cpp F16。公式／speech.cppのcodecはFP32、ONNXは既存主比較と同じdecoder FP16・encoder FP32です。抜粋診断はモデルとcodecをFP32にします。RF40ステップ、MF4ステップ、seed 0〜4、batch 1、候補数1、自動生成長を使用します。透かし・末尾トリム・追加無音除去は無効です。

- JKYB-Parakeet全13,536文：参照音声・captionなし。上流のkana-whisper、Whisper、読み採点器を使用します。
- JVS全100話者：5文章、1クリップ／約30秒／約60秒／120秒の参照、5 seed。ID順のparallel100先頭5発話を対象、続く10発話を固定照合centroid、nonpara30を参照にします。参照は独立encode後に連結し、30／60秒は最後のクリップを保持、120秒のみ切り詰めます。参照不足は失敗として記録します。
- 実用40文：speech-benchの20文と既存速度比較20文。TYC054と中立captionを使用し、Qwen3-ASR／ReazonSpeechの読み戻し、ERes2NetV2の文間一貫性、音高、無音、クリッピングを集計します。

全件の実用設定は389,400要求（JKYB＋JVSは388,400要求）。FP32抜粋23,500要求と共通生成長・ノイズの診断2,000要求を加え、合計414,900要求です。小規模試行はこの集合の一部を先に実行し、重複生成しません。JKYB 500文とJVS20話者の診断対象は出力を参照せずに固定します。

## 準備と起動

リポジトリルートで実行します。既存ベンチマークのモデル・ONNX graph・Vulkanライブラリが必要です。環境は `.venvs/quality` に分離し、既存venvとsubmoduleを変更しません。

```bash
bash benchmark/quality/setup.sh
bash -c 'source scripts/environment.sh; python3 benchmark/quality/build.py'
bash -c 'source scripts/environment.sh; .venvs/quality/bin/python -m pytest -q benchmark/quality/test_quality.py'
bash -c 'source scripts/environment.sh; .venvs/quality/bin/python benchmark/quality/validate.py'
bash benchmark/quality/run.sh --detach
```

データ取得から自動実行する場合は、環境・workerの構築後に次を使います。

```bash
bash benchmark/quality/run.sh --bootstrap --detach
```

bootstrapは公式JVS配布先の公開ダウンロード上限を検出すると、進捗に理由と次回時刻を保存し、6時間後に再試行します。取得できれば事前検証から全件評価へ自動で進みます。その他の準備エラーでは停止します。定期通知やエージェントによる監視は不要です。

## 状態確認・停止・再開

JVSの到着を待たず、JKYBと実用文だけを進める場合は評価対象を指定します。

```bash
bash benchmark/quality/run.sh --datasets jkyb practical --detach
```

この対象では計354,900要求を実行します。対象はrunごとにmanifestへ固定され、`--resume` は保存した対象を引き継ぎます。JVS取得後は別runとして `--datasets jvs --detach` を実行すれば、読み評価を再生成せずに残りの60,000要求を処理できます。JVSだけのrunでは、先に `prepare.py --datasets jvs` で取得済みコーパスのindexを作成します。

JVS配布物の書き起こし一覧には未収録の文も含まれるため、indexは実在する録音のID順で作成します。合成対象5発話と照合用10発話は書き起こしID順の先頭15件を必須とし、欠落時に別の文で補いません。一覧のみ存在する文は `assets/jvs-index-audit.json` に記録します。

```bash
bash benchmark/quality/run.sh --status
bash benchmark/quality/run.sh --stop
bash benchmark/quality/run.sh --resume --detach
# JVS取得待ちから再開する場合
bash benchmark/quality/run.sh --bootstrap --resume --detach
# 原因を修正して失敗した要求も再試行する場合
bash benchmark/quality/run.sh --resume --retry-failed --detach
```

既定では直近のrunを選択します。別runは `--run-id ID` で指定します。tmux名は `irodori-quality-ID`。状態ファイルは30秒おきに更新され、生成・採点の結果は要求ごとにSQLiteへコミットされます。`--stop` は子workerも終了します。単なるtmuxのdetachや端末の切断では処理は止まりません。

再開時はモデル・コード・入力・環境のmanifestと完了音声のSHA-256を検証します。変更があれば混在を防ぐため再開を拒否します。通常再開では失敗済み要求を選別再生成しません。GPUは既存の `artifacts/gpu.lock` を共有し、合成と採点を逐次処理します。同一構成のOOM・timeout等が3回連続すると停止します。

## 保存先と結果の解釈

`artifacts/quality/runs/ID/` にmanifest、固定split、SQLite、生の48kHz float32 WAV、進捗、ログを保存します。`report/` に個別JSONL、seed別・種別別・成功例のみのCSV、公式との差と95% paired cluster bootstrap区間、失敗一覧、日本語レポートを出力します。保存量は小規模試行後に見積もり、残容量に20%の余裕がなければ停止します。

読みの失敗は空出力、話者Top-1は不正解に含めます。CAM++類似度は採点可能な音声のみで、対象件数を併記します。5 seedの平均と母標準偏差を報告し、JKYBは漢字読みの組、JVSは話者を再標本化します。実用文のCERは認識器ごとに独立して報告し、speech-bench方式の上限付き誤り率と上限なしCERを併記します。

主評価の同じseed番号は実装ごとに異なる乱数列です。共通ノイズ診断では公式FP32の生成長と同一float32初期ノイズを渡します。波形相関は数値診断であり、聴感品質の採点ではありません。自動指標だけで自然さや品質同等性を断定しません。

評価器の再実行は、データ取得後に `.venvs/quality/bin/python benchmark/quality/report.py artifacts/quality/runs/ID` を使用します。JVS原音はローカルに保持し、配布対象に含めません。

Gitには評価コード・設定と、`benchmark/results/quality-20261019/` のレポート・主要CSV・検証記録を残します。音声、モデル、SQLite、詳細JSONL、ログ、派生ソース・バイナリは既存の `artifacts/` 配下に保持し、コーパスの `jvs_ver1.zip` と `wav_tsukuyomi.zip` も追跡しません。公開ファイルのローカルパスはリポジトリルートからの相対表記にし、`provenance.json` に保存元、元ファイルの `source_sha256`、公開ファイルの `sha256` を記録しています。追加診断のソース保存先と再集計上の制約は結果レポートを参照してください。

参照：[JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition)、[公式MF評価](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)、[公式JVS手順](https://huggingface.co/Aratako/Irodori-TTS-v4-Small/blob/main/VOICE_CLONING_BENCHMARK.md)、[speech-bench](https://github.com/nyosegawa/speech-bench)。具体的なrevisionはconfigと各runのmanifestに保存します。
