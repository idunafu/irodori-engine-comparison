# 入力サンプルレートの診断

[完了した3,600音声の結果と検証記録](../../results/quality-20261019/sample-rate/RESULTS.ja.md)を保存しています。[品質評価全体のまとめ](../../results/quality-20261019/RESULTS.ja.md)も参照してください。

ルートの `wav_tsukuyomi.zip`（96kHz・100発話）から全レートを直接作る。
96・48・44.1・32・24・16kHzを比較し、量子化条件はfloat32 WAVで揃える。
波形比較は元が96kHzの99発話（ID050のみ48kHzのため除外）。生成比較は参照001〜005、照合011〜020、対象文021〜023を分離し、MF/RF・5 seedを使用する。
元のベンチマーク出力・submoduleを変更しない。

起動（リポジトリルート）：

```bash
tmux new-session -d -s irodori-samplerate 'exec .venvs/quality/bin/python benchmark/quality/sample_rate/run.py > artifacts/quality/runs/20261019-tsukuyomi-samplerate/runner.log 2>&1'
```

初回起動前に出力ディレクトリを作成する。
GPUロックを共有し、進捗は30秒ごとにstatus.jsonへ保存する。通知・常時ポーリングは行わない。

状態確認：

```bash
cat artifacts/quality/runs/20261019-tsukuyomi-samplerate/status.json
tail -n 20 artifacts/quality/runs/20261019-tsukuyomi-samplerate/runner.log
```

停止（子workerも終了）：

```bash
kill -TERM "$(.venvs/quality/bin/python -c 'import json; print(json.load(open("artifacts/quality/runs/20261019-tsukuyomi-samplerate/status.json"))["pid"])')"
```

同じ起動コマンドで途中再開する。manifestおよび成功保存物のhashが違えば拒否する。
失敗済み要求を再試行する場合のみrun.pyへ `--retry-failed` を付ける。OOM・timeoutはworkerを再起動し、3回連続で失敗したら異常終了する。
実行中の二重起動はrun.lockで拒否する。

公式・speech.cpp通常Vulkan・内部F16無効Vulkan・公式へspeech.cppの前処理PCMを渡す条件の4種類。
全てFP32重み。公式48kHz参照の自動長予測と、同一の初期ノイズを条件間で共有する。
生成3,600件、参照エンコード180件、CAM++採点3,600件を保存し、最後にCSVと日本語レポートを自動生成する。
1話者なのでTop-1は集計しない。再標本化単位は参照クリップであり、話者集団への一般化はしない。
