# PyTorch / ONNX Runtime 更新 A/B（2026-10-15）

状態: 全測定完了。


## 主比較の速度差

同じ生成フレーム数に揃えた全20文章（matched）。各文章で `(新版ms / 旧版ms - 1) × 100` を計算し、その中央値を掲載。負の値は新版が高速。

| 実装 / モデル / 重み精度 | 再エンコード 差% | キャッシュあり 差% |
|---|---:|---:|
| Irodori-TTS / RF / FP32 | +2.02 | +1.36 |
| Irodori-TTS / RF / BF16 | -2.62 | -3.60 |
| Irodori-TTS / MF / FP32 | +9.75 | +8.47 |
| Irodori-TTS / MF / BF16 | +12.01 | +11.36 |
| Irodori-TTS-Optimiz / RF / FP32 | -0.46 | -0.45 |
| Irodori-TTS-Optimiz / RF / BF16 | -10.82 | -11.37 |
| irodori-onnx / MF / FP32 | +1.15 | +0.25 |
| irodori-onnx / MF / FP16 | +1.20 | -1.63 |

[生成長スケーリング](audio_scaling.png)／[テキスト長スケーリング](text_scaling.png)／[全入力CSV](per-input.csv)。

既存venv・固定エンジン・モデル・既存結果を保持し、専用venvで比較。負の差は新版が高速。

旧: Python 3.10.20、PyTorch 2.10.0+cu128、ONNX Runtime 1.23.2。
新: PyTorch 2.14.1+cu130、TorchAudio 2.11.0、TorchCodec 0.17.0、ONNX Runtime 1.31.0、ONNX 1.23.2、ONNXScript 0.7.2。
ONNX Runtimeの最新wheelはPython>=3.11のためONNX環境は3.11。PyTorch版は3.10を維持。

ONNXはRustから共有ライブラリを直接使用するため、Pythonの実行時間は測定に含まれない。

同じエンジンコミット・重み・ONNXグラフ・参照PCMを使用。再exportせずruntimeの更新を比較する。ONNX/ONNXScriptは導入バージョンを記録し、測定する推論自体には使用しない。

batch 1、seed 42、RF 40 / MF 4 steps、CPU 8 threads、TF32 off、compile off。caption/CFG/codec精度/キャッシュは主比較と同じ。
実用精度は公式／OptimizがBF16モデル＋FP32 codec、ONNXがFP16モデル・decoder＋FP32参照encoder。FP32条件はモデル・codecともFP32。
参照音声は共通のTYC054.wav（48kHz・PCM24）。captionは「自然な日本語で、落ち着いた口調で話す。」。RFはindependent CFG（text/caption 3、speaker 5、適用区間0.5〜1.0）、linear schedule。
小規模比較: p01/p19、2秒/8秒、両キャッシュ、FP32/実用精度。3ラウンド、旧新の順序を交互にし、各条件warmup 2回・通常5回・詳細5回。
拡張判定: 対応ラウンドを保った階層bootstrap 95%区間が0を跨がず、全3ラウンドが同方向。2%以上は実用差として別記。条件別検定の多重性補正なし。
拡張: 該当するエンジン・モデル・精度の新旧両方で主比較と同じ20文章、matched/auto/text_scaling/audio_scaling、両キャッシュ、warmup 2回・通常5回・詳細5回。拡張は1ラウンドの記述統計。

## 小規模A/B

| 条件 | cache | 入力 | 秒 | 旧ms | 新ms | 差% | 95%区間% | 有意差 |
|---|---|---|---:|---:|---:|---:|---|---|
| Irodori-TTS-rf-cuda-fp32 | reencode | p01 | 2 | 379.50 | 364.58 | -3.93 | -5.23〜-3.26 | True |
| Irodori-TTS-rf-cuda-fp32 | reencode | p01 | 8 | 640.75 | 642.62 | +0.29 | -0.10〜+1.26 | False |
| Irodori-TTS-rf-cuda-fp32 | reencode | p19 | 2 | 379.80 | 365.04 | -3.89 | -4.08〜-3.47 | True |
| Irodori-TTS-rf-cuda-fp32 | reencode | p19 | 8 | 641.48 | 644.48 | +0.47 | +0.05〜+1.24 | True |
| Irodori-TTS-rf-cuda-fp32 | native_cache | p01 | 2 | 333.82 | 314.74 | -5.72 | -5.86〜-5.31 | True |
| Irodori-TTS-rf-cuda-fp32 | native_cache | p01 | 8 | 598.91 | 598.30 | -0.10 | -0.59〜+0.80 | False |
| Irodori-TTS-rf-cuda-fp32 | native_cache | p19 | 2 | 334.02 | 314.88 | -5.73 | -5.88〜-5.30 | True |
| Irodori-TTS-rf-cuda-fp32 | native_cache | p19 | 8 | 599.90 | 598.85 | -0.18 | -0.64〜+0.67 | False |
| Irodori-TTS-rf-cuda-practical | reencode | p01 | 2 | 293.89 | 298.64 | +1.62 | -0.38〜+2.52 | False |
| Irodori-TTS-rf-cuda-practical | reencode | p01 | 8 | 355.51 | 351.13 | -1.23 | -1.80〜-0.30 | True |
| Irodori-TTS-rf-cuda-practical | reencode | p19 | 2 | 294.08 | 299.08 | +1.70 | +0.38〜+2.44 | True |
| Irodori-TTS-rf-cuda-practical | reencode | p19 | 8 | 356.83 | 352.16 | -1.31 | -1.97〜-0.26 | True |
| Irodori-TTS-rf-cuda-practical | native_cache | p01 | 2 | 246.97 | 246.82 | -0.06 | -0.79〜+1.41 | False |
| Irodori-TTS-rf-cuda-practical | native_cache | p01 | 8 | 307.24 | 301.09 | -2.00 | -2.54〜-1.50 | True |
| Irodori-TTS-rf-cuda-practical | native_cache | p19 | 2 | 245.19 | 247.63 | +0.99 | -1.61〜+1.67 | False |
| Irodori-TTS-rf-cuda-practical | native_cache | p19 | 8 | 307.34 | 300.37 | -2.27 | -2.65〜-1.84 | True |
| Irodori-TTS-mf-cuda-fp32 | reencode | p01 | 2 | 102.14 | 104.59 | +2.39 | +1.95〜+5.43 | True |
| Irodori-TTS-mf-cuda-fp32 | reencode | p01 | 8 | 140.48 | 149.22 | +6.22 | +4.43〜+7.55 | True |
| Irodori-TTS-mf-cuda-fp32 | reencode | p19 | 2 | 102.36 | 105.68 | +3.24 | +0.35〜+4.62 | True |
| Irodori-TTS-mf-cuda-fp32 | reencode | p19 | 8 | 141.73 | 149.39 | +5.41 | +4.17〜+6.04 | True |
| Irodori-TTS-mf-cuda-fp32 | native_cache | p01 | 2 | 56.20 | 54.47 | -3.08 | -3.89〜-2.09 | True |
| Irodori-TTS-mf-cuda-fp32 | native_cache | p01 | 8 | 95.41 | 99.96 | +4.77 | +4.42〜+5.24 | True |
| Irodori-TTS-mf-cuda-fp32 | native_cache | p19 | 2 | 56.30 | 54.57 | -3.06 | -3.33〜-2.72 | True |
| Irodori-TTS-mf-cuda-fp32 | native_cache | p19 | 8 | 95.30 | 99.86 | +4.79 | +3.75〜+5.53 | True |
| Irodori-TTS-mf-cuda-practical | reencode | p01 | 2 | 92.72 | 99.46 | +7.28 | +6.11〜+8.07 | True |
| Irodori-TTS-mf-cuda-practical | reencode | p01 | 8 | 119.54 | 132.41 | +10.77 | +10.24〜+11.49 | True |
| Irodori-TTS-mf-cuda-practical | reencode | p19 | 2 | 93.25 | 99.97 | +7.22 | +6.65〜+8.89 | True |
| Irodori-TTS-mf-cuda-practical | reencode | p19 | 8 | 119.56 | 133.73 | +11.86 | +9.85〜+14.29 | True |
| Irodori-TTS-mf-cuda-practical | native_cache | p01 | 2 | 45.28 | 48.01 | +6.02 | +4.58〜+7.87 | True |
| Irodori-TTS-mf-cuda-practical | native_cache | p01 | 8 | 72.19 | 80.75 | +11.87 | +11.07〜+13.17 | True |
| Irodori-TTS-mf-cuda-practical | native_cache | p19 | 2 | 45.18 | 47.88 | +5.98 | +4.28〜+8.05 | True |
| Irodori-TTS-mf-cuda-practical | native_cache | p19 | 8 | 72.36 | 80.90 | +11.80 | +11.00〜+12.27 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | reencode | p01 | 2 | 384.84 | 365.13 | -5.12 | -5.63〜-5.06 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | reencode | p01 | 8 | 652.81 | 643.36 | -1.45 | -1.76〜-1.10 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | reencode | p19 | 2 | 386.76 | 366.13 | -5.33 | -5.49〜-4.93 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | reencode | p19 | 8 | 654.19 | 645.33 | -1.35 | -1.90〜-1.00 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | native_cache | p01 | 2 | 342.64 | 324.43 | -5.32 | -5.44〜-5.20 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | native_cache | p01 | 8 | 614.88 | 605.09 | -1.59 | -1.78〜-1.16 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | native_cache | p19 | 2 | 342.82 | 323.61 | -5.60 | -5.68〜-4.81 | True |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | native_cache | p19 | 8 | 615.71 | 606.42 | -1.51 | -1.73〜-1.17 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | reencode | p01 | 2 | 270.34 | 276.60 | +2.32 | +1.42〜+2.72 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | reencode | p01 | 8 | 340.61 | 328.08 | -3.68 | -4.11〜-3.25 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | reencode | p19 | 2 | 269.59 | 277.76 | +3.03 | +0.42〜+4.18 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | reencode | p19 | 8 | 340.91 | 327.47 | -3.94 | -4.02〜-2.99 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | native_cache | p01 | 2 | 226.16 | 230.84 | +2.07 | +1.25〜+3.68 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | native_cache | p01 | 8 | 297.64 | 284.54 | -4.40 | -4.78〜-3.69 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | native_cache | p19 | 2 | 226.01 | 231.72 | +2.53 | +1.32〜+3.55 | True |
| Irodori-TTS-Optimiz-rf-cuda-practical | native_cache | p19 | 8 | 298.27 | 285.69 | -4.22 | -4.56〜-3.69 | True |
| irodori-onnx-mf-cuda-fp32 | reencode | p01 | 2 | 92.84 | 93.17 | +0.36 | -1.02〜+1.11 | False |
| irodori-onnx-mf-cuda-fp32 | reencode | p01 | 8 | 135.69 | 138.85 | +2.33 | +1.53〜+3.30 | True |
| irodori-onnx-mf-cuda-fp32 | reencode | p19 | 2 | 97.98 | 97.58 | -0.41 | -0.92〜+1.34 | False |
| irodori-onnx-mf-cuda-fp32 | reencode | p19 | 8 | 139.76 | 143.93 | +2.99 | +1.38〜+4.20 | True |
| irodori-onnx-mf-cuda-fp32 | native_cache | p01 | 2 | 47.93 | 46.38 | -3.24 | -4.28〜-1.74 | True |
| irodori-onnx-mf-cuda-fp32 | native_cache | p01 | 8 | 91.30 | 93.14 | +2.02 | +1.54〜+2.37 | True |
| irodori-onnx-mf-cuda-fp32 | native_cache | p19 | 2 | 54.35 | 52.68 | -3.08 | -3.33〜-2.39 | True |
| irodori-onnx-mf-cuda-fp32 | native_cache | p19 | 8 | 97.03 | 98.93 | +1.96 | +1.73〜+2.22 | True |
| irodori-onnx-mf-cuda-practical | reencode | p01 | 2 | 71.66 | 73.76 | +2.94 | +2.21〜+3.60 | True |
| irodori-onnx-mf-cuda-practical | reencode | p01 | 8 | 91.05 | 92.18 | +1.24 | +0.83〜+2.81 | True |
| irodori-onnx-mf-cuda-practical | reencode | p19 | 2 | 72.48 | 74.67 | +3.03 | +2.61〜+5.30 | True |
| irodori-onnx-mf-cuda-practical | reencode | p19 | 8 | 90.73 | 93.21 | +2.73 | +1.35〜+4.64 | True |
| irodori-onnx-mf-cuda-practical | native_cache | p01 | 2 | 29.92 | 29.38 | -1.81 | -2.33〜-0.89 | True |
| irodori-onnx-mf-cuda-practical | native_cache | p01 | 8 | 47.91 | 47.75 | -0.34 | -1.72〜+0.49 | False |
| irodori-onnx-mf-cuda-practical | native_cache | p19 | 2 | 30.83 | 30.44 | -1.27 | -1.74〜-0.17 | True |
| irodori-onnx-mf-cuda-practical | native_cache | p19 | 8 | 48.94 | 48.82 | -0.23 | -0.77〜+0.47 | False |

## 本ベンチマーク

拡張対象: Irodori-TTS-Optimiz-rf-cuda-fp32, Irodori-TTS-Optimiz-rf-cuda-practical, Irodori-TTS-mf-cuda-fp32, Irodori-TTS-mf-cuda-practical, Irodori-TTS-rf-cuda-fp32, Irodori-TTS-rf-cuda-practical, irodori-onnx-mf-cuda-fp32, irodori-onnx-mf-cuda-practical

| 条件 | mode | cache | 入力条件数 | 旧ms中央値 | 新ms中央値 | 入力別差%中央値 |
|---|---|---|---:|---:|---:|---:|
| Irodori-TTS-Optimiz-rf-cuda-fp32 | audio_scaling | native_cache | 5 | 618.22 | 608.90 | -1.51 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | audio_scaling | reencode | 5 | 658.86 | 648.26 | -1.61 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | auto | native_cache | 20 | 1388.80 | 1382.21 | -0.50 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | auto | reencode | 20 | 1423.95 | 1421.70 | -0.25 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | matched | native_cache | 20 | 1367.43 | 1361.39 | -0.45 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | matched | reencode | 20 | 1398.87 | 1394.54 | -0.46 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | text_scaling | native_cache | 20 | 618.73 | 609.58 | -1.48 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | text_scaling | reencode | 20 | 658.64 | 649.56 | -1.40 |
| Irodori-TTS-Optimiz-rf-cuda-practical | audio_scaling | native_cache | 5 | 297.18 | 285.64 | -4.78 |
| Irodori-TTS-Optimiz-rf-cuda-practical | audio_scaling | reencode | 5 | 344.84 | 327.78 | -4.69 |
| Irodori-TTS-Optimiz-rf-cuda-practical | auto | native_cache | 20 | 586.67 | 559.17 | -11.20 |
| Irodori-TTS-Optimiz-rf-cuda-practical | auto | reencode | 20 | 625.42 | 596.87 | -10.75 |
| Irodori-TTS-Optimiz-rf-cuda-practical | matched | native_cache | 20 | 576.67 | 549.17 | -11.37 |
| Irodori-TTS-Optimiz-rf-cuda-practical | matched | reencode | 20 | 615.75 | 588.58 | -10.82 |
| Irodori-TTS-Optimiz-rf-cuda-practical | text_scaling | native_cache | 20 | 297.74 | 285.34 | -4.19 |
| Irodori-TTS-Optimiz-rf-cuda-practical | text_scaling | reencode | 20 | 342.90 | 328.32 | -4.19 |
| Irodori-TTS-mf-cuda-fp32 | audio_scaling | native_cache | 5 | 96.02 | 101.14 | +5.33 |
| Irodori-TTS-mf-cuda-fp32 | audio_scaling | reencode | 5 | 144.36 | 152.52 | +5.66 |
| Irodori-TTS-mf-cuda-fp32 | auto | native_cache | 20 | 225.49 | 241.70 | +7.81 |
| Irodori-TTS-mf-cuda-fp32 | auto | reencode | 20 | 263.98 | 283.78 | +8.28 |
| Irodori-TTS-mf-cuda-fp32 | matched | native_cache | 20 | 203.48 | 220.30 | +8.47 |
| Irodori-TTS-mf-cuda-fp32 | matched | reencode | 20 | 241.54 | 263.54 | +9.75 |
| Irodori-TTS-mf-cuda-fp32 | text_scaling | native_cache | 20 | 96.21 | 101.09 | +5.06 |
| Irodori-TTS-mf-cuda-fp32 | text_scaling | reencode | 20 | 142.82 | 151.04 | +5.74 |
| Irodori-TTS-mf-cuda-practical | audio_scaling | native_cache | 5 | 73.44 | 81.53 | +11.02 |
| Irodori-TTS-mf-cuda-practical | audio_scaling | reencode | 5 | 121.11 | 135.70 | +11.13 |
| Irodori-TTS-mf-cuda-practical | auto | native_cache | 20 | 161.03 | 178.68 | +10.95 |
| Irodori-TTS-mf-cuda-practical | auto | reencode | 20 | 208.29 | 230.09 | +11.53 |
| Irodori-TTS-mf-cuda-practical | matched | native_cache | 20 | 150.53 | 167.39 | +11.36 |
| Irodori-TTS-mf-cuda-practical | matched | reencode | 20 | 195.44 | 219.11 | +12.01 |
| Irodori-TTS-mf-cuda-practical | text_scaling | native_cache | 20 | 73.53 | 81.36 | +10.66 |
| Irodori-TTS-mf-cuda-practical | text_scaling | reencode | 20 | 122.12 | 134.56 | +10.24 |
| Irodori-TTS-rf-cuda-fp32 | audio_scaling | native_cache | 5 | 604.48 | 603.76 | -0.12 |
| Irodori-TTS-rf-cuda-fp32 | audio_scaling | reencode | 5 | 647.59 | 650.00 | +0.37 |
| Irodori-TTS-rf-cuda-fp32 | auto | native_cache | 20 | 1364.27 | 1380.03 | +1.29 |
| Irodori-TTS-rf-cuda-fp32 | auto | reencode | 20 | 1404.67 | 1425.16 | +1.65 |
| Irodori-TTS-rf-cuda-fp32 | matched | native_cache | 20 | 1342.09 | 1358.27 | +1.36 |
| Irodori-TTS-rf-cuda-fp32 | matched | reencode | 20 | 1380.17 | 1404.69 | +2.02 |
| Irodori-TTS-rf-cuda-fp32 | text_scaling | native_cache | 20 | 604.72 | 604.48 | -0.04 |
| Irodori-TTS-rf-cuda-fp32 | text_scaling | reencode | 20 | 649.01 | 651.27 | +0.38 |
| Irodori-TTS-rf-cuda-practical | audio_scaling | native_cache | 5 | 309.54 | 301.06 | -0.92 |
| Irodori-TTS-rf-cuda-practical | audio_scaling | reencode | 5 | 358.23 | 353.12 | +1.51 |
| Irodori-TTS-rf-cuda-practical | auto | native_cache | 20 | 572.00 | 579.65 | -3.61 |
| Irodori-TTS-rf-cuda-practical | auto | reencode | 20 | 615.72 | 629.11 | -2.57 |
| Irodori-TTS-rf-cuda-practical | matched | native_cache | 20 | 556.05 | 562.59 | -3.60 |
| Irodori-TTS-rf-cuda-practical | matched | reencode | 20 | 601.91 | 610.77 | -2.62 |
| Irodori-TTS-rf-cuda-practical | text_scaling | native_cache | 20 | 310.06 | 301.12 | -2.89 |
| Irodori-TTS-rf-cuda-practical | text_scaling | reencode | 20 | 357.67 | 353.84 | -1.06 |
| irodori-onnx-mf-cuda-fp32 | audio_scaling | native_cache | 5 | 94.72 | 96.32 | -0.33 |
| irodori-onnx-mf-cuda-fp32 | audio_scaling | reencode | 5 | 137.21 | 140.98 | +0.83 |
| irodori-onnx-mf-cuda-fp32 | auto | native_cache | 20 | 249.40 | 249.50 | +0.29 |
| irodori-onnx-mf-cuda-fp32 | auto | reencode | 20 | 292.59 | 294.85 | +1.06 |
| irodori-onnx-mf-cuda-fp32 | matched | native_cache | 20 | 249.10 | 248.53 | +0.25 |
| irodori-onnx-mf-cuda-fp32 | matched | reencode | 20 | 291.74 | 294.48 | +1.15 |
| irodori-onnx-mf-cuda-fp32 | text_scaling | native_cache | 20 | 94.72 | 96.28 | +1.86 |
| irodori-onnx-mf-cuda-fp32 | text_scaling | reencode | 20 | 137.09 | 141.10 | +3.07 |
| irodori-onnx-mf-cuda-practical | audio_scaling | native_cache | 5 | 48.23 | 48.36 | -0.32 |
| irodori-onnx-mf-cuda-practical | audio_scaling | reencode | 5 | 90.05 | 92.86 | +3.12 |
| irodori-onnx-mf-cuda-practical | auto | native_cache | 20 | 104.89 | 103.51 | -1.46 |
| irodori-onnx-mf-cuda-practical | auto | reencode | 20 | 147.06 | 147.90 | +0.66 |
| irodori-onnx-mf-cuda-practical | matched | native_cache | 20 | 103.81 | 102.33 | -1.63 |
| irodori-onnx-mf-cuda-practical | matched | reencode | 20 | 145.83 | 146.89 | +1.20 |
| irodori-onnx-mf-cuda-practical | text_scaling | native_cache | 20 | 48.46 | 48.50 | -0.12 |
| irodori-onnx-mf-cuda-practical | text_scaling | reencode | 20 | 90.26 | 93.02 | +3.05 |

## 段階別時間・VRAM（matched）

旧／新の順。各文章の詳細計測中央値を20文章で中央値に集計する。VRAM常駐量も中央値、peakは20文章の最大値。段階別時間はGPU同期する別実行であり、通常時間への単純な加算には使わない。

| 条件 | cache | sampling ms | DACVAE decode ms | 常駐 MiB | peak MiB |
|---|---|---:|---:|---:|---:|
| Irodori-TTS-Optimiz-rf-cuda-fp32 | native_cache | 1235.67／1235.20 | 110.98／106.78 | 8216.00／8682.00 | 10328.00／10794.00 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | reencode | 1226.60／1227.16 | 110.21／106.40 | 6575.00／6596.00 | 7018.00／6950.00 |
| Irodori-TTS-Optimiz-rf-cuda-practical | native_cache | 463.64／439.26 | 104.41／100.77 | 7174.00／7184.00 | 9286.00／9296.00 |
| Irodori-TTS-Optimiz-rf-cuda-practical | reencode | 462.03／437.49 | 103.22／99.99 | 5038.00／5130.00 | 5534.00／5974.00 |
| Irodori-TTS-mf-cuda-fp32 | native_cache | 62.42／61.87 | 118.33／139.67 | 12434.00／10164.00 | 12434.00／10164.00 |
| Irodori-TTS-mf-cuda-fp32 | reencode | 62.27／61.61 | 113.10／132.59 | 10322.00／8578.00 | 12434.00／10162.00 |
| Irodori-TTS-mf-cuda-practical | native_cache | 24.99／25.43 | 114.51／129.61 | 11912.00／9548.00 | 11912.00／9548.00 |
| Irodori-TTS-mf-cuda-practical | reencode | 25.63／25.52 | 111.39／128.59 | 9798.00／7963.00 | 11910.00／9548.00 |
| Irodori-TTS-rf-cuda-fp32 | native_cache | 1197.57／1195.76 | 123.02／141.84 | 12278.00／10138.00 | 12278.00／10138.00 |
| Irodori-TTS-rf-cuda-fp32 | reencode | 1189.59／1192.46 | 122.33／141.97 | 10164.00／8552.00 | 12276.00／10136.00 |
| Irodori-TTS-rf-cuda-practical | native_cache | 431.99／418.38 | 113.10／132.08 | 11908.00／9544.00 | 11908.00／9544.00 |
| Irodori-TTS-rf-cuda-practical | reencode | 431.37／417.52 | 111.60／131.11 | 9790.00／7954.00 | 11902.00／9538.00 |
| irodori-onnx-mf-cuda-fp32 | native_cache | 61.07／59.77 | 181.12／181.95 | 5736.00／5736.00 | 8010.00／8010.00 |
| irodori-onnx-mf-cuda-fp32 | reencode | 60.20／59.12 | 180.81／182.22 | 5720.00／5720.00 | 7994.00／7994.00 |
| irodori-onnx-mf-cuda-practical | native_cache | 33.04／31.77 | 64.94／64.48 | 3402.00／3404.00 | 5908.00／5880.00 |
| irodori-onnx-mf-cuda-practical | reencode | 32.96／31.85 | 65.00／64.49 | 3386.00／3388.00 | 5892.00／5864.00 |

## ロード・初回推論

小規模比較の各ラウンドで新しいworkerを起動し、p09を8秒生成。各版3回の中央値。OS・ドライバのキャッシュは保持。

| 条件 | 版 | 回数 | load ms | 起動からready ms | 初回推論 ms | ロード後常駐 MiB | ロードpeak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Irodori-TTS-rf-cuda-fp32 | old | 3 | 4116.28 | 4746.12 | 870.70 | 4156.0 | 4156.0 |
| Irodori-TTS-rf-cuda-fp32 | new | 3 | 6772.21 | 8170.81 | 1045.37 | 4286.0 | 4286.0 |
| Irodori-TTS-rf-cuda-practical | old | 3 | 4108.89 | 4738.70 | 1568.41 | 3798.0 | 3798.0 |
| Irodori-TTS-rf-cuda-practical | new | 3 | 6837.01 | 8249.59 | 866.34 | 3936.0 | 3936.0 |
| Irodori-TTS-mf-cuda-fp32 | old | 3 | 4130.45 | 4753.03 | 367.65 | 4184.0 | 4184.0 |
| Irodori-TTS-mf-cuda-fp32 | new | 3 | 6840.97 | 8244.27 | 558.70 | 4314.0 | 4314.0 |
| Irodori-TTS-mf-cuda-practical | old | 3 | 4129.81 | 4751.65 | 928.87 | 3818.0 | 3818.0 |
| Irodori-TTS-mf-cuda-practical | new | 3 | 6843.05 | 8255.20 | 632.70 | 3956.0 | 3956.0 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | old | 3 | 4201.98 | 4968.39 | 6205.50 | 4168.0 | 6466.0 |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | new | 3 | 6982.80 | 8407.64 | 6446.80 | 4166.0 | 6464.0 |
| Irodori-TTS-Optimiz-rf-cuda-practical | old | 3 | 4223.91 | 4993.84 | 5930.89 | 2756.0 | 5074.0 |
| Irodori-TTS-Optimiz-rf-cuda-practical | new | 3 | 6955.34 | 8378.14 | 6184.72 | 2776.0 | 5092.0 |
| irodori-onnx-mf-cuda-fp32 | old | 3 | 1460.28 | 1461.61 | 416.16 | 5558.0 | 5612.0 |
| irodori-onnx-mf-cuda-fp32 | new | 3 | 1418.15 | 1419.31 | 345.09 | 5614.0 | 5616.0 |
| irodori-onnx-mf-cuda-practical | old | 3 | 944.39 | 945.75 | 502.06 | 3230.0 | 3248.0 |
| irodori-onnx-mf-cuda-practical | new | 3 | 1129.90 | 1131.10 | 418.71 | 3280.0 | 3280.0 |

## 新旧PCMの数値差

p01・2秒・reencode、同じseedの通常推論出力を比較。相対L2差は音質スコアではない。計測有無によるPCM差は全16構成で別途検証済み。

| 条件 | 相対L2差 | 最大絶対差 | bit一致 |
|---|---:|---:|---|
| Irodori-TTS-rf-cuda-fp32 | 5.21431e-06 | 9.13441e-06 | False |
| Irodori-TTS-rf-cuda-practical | 0.560857 | 1.26376 | False |
| Irodori-TTS-mf-cuda-fp32 | 6.41724e-06 | 1.16527e-05 | False |
| Irodori-TTS-mf-cuda-practical | 0.0235944 | 0.0562229 | False |
| Irodori-TTS-Optimiz-rf-cuda-fp32 | 4.9586e-07 | 1.07288e-06 | False |
| Irodori-TTS-Optimiz-rf-cuda-practical | 0.687012 | 1.20381 | False |
| irodori-onnx-mf-cuda-fp32 | 1.54521e-05 | 2.69115e-05 | False |
| irodori-onnx-mf-cuda-practical | 0.0176353 | 0.0393066 | False |

全入力の速度・RTF・段階別時間・VRAM: [CSV](per-input.csv)。

native_cacheでの[生成長スケーリング](audio_scaling.png)／[テキスト長スケーリング](text_scaling.png)。

検証: {"preserved": true, "preflight_count": 16, "incomplete": 0, "wall_requests": 12320, "profile_requests": 12320, "measured_conditions": 1104, "affected": ["Irodori-TTS-Optimiz-rf-cuda-fp32", "Irodori-TTS-Optimiz-rf-cuda-practical", "Irodori-TTS-mf-cuda-fp32", "Irodori-TTS-mf-cuda-practical", "Irodori-TTS-rf-cuda-fp32", "Irodori-TTS-rf-cuda-practical", "irodori-onnx-mf-cuda-fp32", "irodori-onnx-mf-cuda-practical"], "expected_conditions": 1104, "missing_conditions": 0, "measurement_complete": true}

[最終独立検証](independent-validation.json)／[導入バージョン](../../../environments/runtime-ab-20261015/versions.json)。独立検証ファイルは全測定終了後に生成する。

PCMの新旧差は数値差として生データに保存。音声品質の採点は行っていない。
生データ・全依存バージョン・起動コマンド・モデルハッシュ: `artifacts/runtime-ab-20261015/`。

公式情報: [PyTorch wheels](https://download.pytorch.org/whl/cu130/torch/)、[TorchAudio互換性](https://docs.pytorch.org/audio/main/installation.html)、[ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
