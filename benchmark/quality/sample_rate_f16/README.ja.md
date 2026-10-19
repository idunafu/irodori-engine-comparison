# speech.cpp F16重みの追加評価

[完了した900音声の結果と検証記録](../../results/quality-20261019/sample-rate-f16/RESULTS.ja.md)を保存しています。[品質評価全体のまとめ](../../results/quality-20261019/RESULTS.ja.md)も参照してください。

`sample_rate/` の完了済みFP32評価から、入力・生成長・初期ノイズ・照合話者centroidを再利用する。
元評価と同じMF/RF、6レート、参照5クリップ、3文章、5 seedで900音声を生成・CAM++採点する。
公式FP32・speech.cpp FP32の保存済み結果と比較する。公式BF16・ONNX FP16の追加実行は含まない。
F16モデルのcodec重み型を確認し、参照codec latentの差も保存する。

起動前に `artifacts/quality/runs/20261019-tsukuyomi-samplerate-f16` を作成し、リポジトリルートで実行する。

```bash
tmux new-session -d -s irodori-samplerate-f16 'exec env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venvs/quality/bin/python benchmark/quality/sample_rate_f16/run.py >> artifacts/quality/runs/20261019-tsukuyomi-samplerate-f16/runner.log 2>&1'
```

状態確認は出力フォルダー内の `status.json` と `runner.log`。進捗を30秒ごとに保存し、共有GPUロックを使用する。
停止は `status.json` のpidへSIGTERM。子workerも停止する。同じコマンドで途中再開し、manifestと成功済み保存物のhashを確認する。
失敗要求の再試行は `--retry-failed` 明示時のみ。OOM・timeoutはworkerを再起動し、3回連続で異常終了する。

再集計：

```bash
.venvs/quality/bin/python benchmark/quality/sample_rate_f16/run.py --report-only
```
