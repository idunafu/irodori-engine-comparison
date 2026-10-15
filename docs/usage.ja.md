# 構築・実行手順

[概要](../README.md)／[測定方法](methodology.ja.md)／[比較結果](../benchmark/results/RESULTS.ja.md)

コマンドとパスは、すべてリポジトリルートを基準に記載しています。モデル・環境・生成物はリポジトリ内のGit管理外ディレクトリに保存します。

## 事前条件

検証済み環境は[READMEの環境表](../README.md#検証環境とビルド)を参照してください。Git、uv、Python 3、CMake、C/C++コンパイラ、Make、Rust／Cargo、CUDA Toolkit、CUDA／Vulkanを利用できるGPUドライバが必要です。モデル・依存・Vulkan SDKの取得にはネットワーク接続と十分な空き容量も必要です。

CUDA compilerは`PATH`上の`nvcc`から検出します。別のToolkitを使う場合は`CUDACXX`または`CUDAToolkit_ROOT`（`CUDA_HOME`／`CUDA_PATH`も可）で指定します。CUDAアーキテクチャの既定値は掲載結果と同じ`120`（RTX 5090）です。他のGPUでは`CUDAARCHS`または`CMAKE_CUDA_ARCHITECTURES`で変更してください。

Vulkanローダーはシステムの検索情報またはSDK内から検出します。`VULKAN_LIBRARY`で明示指定することもできます。既存SDKは`VULKAN_SDK`で指定でき、省略時はリポジトリ内のSDKを使用します。パスはソースコードを変更せずに設定できます。

| 設定 | 調整するファイル |
|---|---|
| CUDA compiler・Toolkit・architecture、Vulkanライブラリの検出と指定 | [scripts/native_toolchain.py](../scripts/native_toolchain.py)（単体／計測ビルドで共用） |
| Vulkan SDK・キャッシュの配置 | [scripts/environment.sh](../scripts/environment.sh) |
| ONNX Runtime・CUDA依存ライブラリの探索 | [scripts/onnx-environment.sh](../scripts/onnx-environment.sh)、[benchmark/run.py](../benchmark/run.py) |

speech.cppの追加A/Bビルドも`benchmark/build.py`の設定を使用します。別ハードウェアの測定は掲載結果と分けて保存してください。

```bash
# 例：対象GPUのアーキテクチャを変更（ToolkitのパスはPATHから検出）
export CUDAARCHS=90
```

システムパッケージ・ドライバ・シェル設定ファイルの変更は行いません。不足している依存は個別に用意してから該当工程を再実行します。Vulkanの開発ファイルは、公式SDK 1.4.357.0のアーカイブを検証してローカルへ展開します。

## 環境構築

```bash
# 親が記録したコミットで初期化。入れ子のSSH URLはHTTPSへ読み替える
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive

# モデル取得、ビルド、変換、短い音声生成による確認
bash scripts/setup.sh
```

公式版とOptimizはPython 3.10.20の独立環境を作り、上流の`uv.lock`を`--frozen --extra cu128`で使用します。ONNXは[environments/onnx/](../environments/onnx/)の独立したuv projectとlockを使用し、`ort 2.0.0-rc.11`に合わせてONNX Runtime 1.23.2を固定しています。C++のビルド用CUDA 13.3と、PyTorch／ONNXがvenv内で使用するCUDA 12系runtimeは別です。

セットアップは実装ごとの失敗を記録して他の工程を続け、最後に失敗の有無を報告します。結果は`artifacts/setup-results.json`、各工程のログは`logs/`に保存します。submoduleのコミットを`engines.lock.json`と照合し、自動で最新へ更新しません。

工程を絞って再開する場合は、依存する前工程を完了させてから実行します。

```bash
bash scripts/setup-vulkan.sh
bash scripts/setup-python.sh
bash scripts/setup-onnx-python.sh
python3 scripts/setup-models.py
bash scripts/setup-native.sh audio.cpp
bash scripts/setup-native.sh speech.cpp
bash scripts/setup-native.sh sttts-gui
bash scripts/setup-native.sh Irodori-TTS-burn
bash scripts/setup-native.sh irodori-onnx
bash scripts/convert-models.sh
```

`scripts/convert-models.sh`は初期動作確認用の合成参照音声を利用します。新しいcloneでは、先に`scripts/setup.sh`で参照音声まで作成するか、次のコマンドで生成してください。

```bash
bash scripts/run.sh Irodori-TTS --model rf --no-reference --output artifacts/smoke/original-rf.wav
```

## 単体の音声生成・動作確認

```bash
bash scripts/run.sh Irodori-TTS --model rf --text '今日はいい天気ですね。' --output artifacts/example.wav
bash scripts/run.sh audio.cpp --model rf --backend cuda
bash scripts/run.sh audio.cpp --model rf --backend vulkan
bash scripts/run.sh speech.cpp --model mf --backend cuda
bash scripts/run.sh speech.cpp --model rf --backend vulkan
bash scripts/run.sh sttts-gui --model mf
bash scripts/run.sh Irodori-TTS-burn --model rf
bash scripts/run.sh irodori-onnx --model mf
bash scripts/run.sh Irodori-TTS-Optimiz --model rf
```

共通オプションは`--text`、`--output`、`--reference`、`--seed`、`--no-reference`です。バックエンド省略時はCUDAを優先し、WGPU実装はVulkanを選びます。未対応のモデル・バックエンド指定はエラーになります。

単体実行の既定の参照音声は`artifacts/voices/ref1.wav`です。ベンチマーク用の`TYC054.wav`とは別で、初期セットアップ時に公式版で生成した合成音声を使います。ONNXの上流exampleは参照音声を必要とします。起動時はローカルのモデルを使用し、モデルの自動更新を無効にしています。

```bash
bash scripts/check.sh
bash scripts/check.sh audio.cpp
bash scripts/check.sh sttts-gui --model mf
bash scripts/check-onnx.sh
```

GPU処理はロックで逐次実行します。sttts-sayは単体確認後に停止し、プロセスとGPUの解放を確認します。

上流CLIを直接使う場合は`source scripts/environment.sh`でローカルのSDK・キャッシュを設定します。Python実装は対応venvのPythonを使い、`PYTHONPATH`に対象の`engines/`ディレクトリを指定します。ONNXでは`source scripts/onnx-environment.sh`でruntimeとCUDA依存ライブラリの探索先も設定できます。

## ベンチマークの準備

参照音声をリポジトリルートの`TYC054.wav`として配置するか、[benchmark/config.json](../benchmark/config.json)の`reference`を変更します。モノラル・48kHzのPCM WAVを使用し、10〜15秒程度の同じ話者による普通の速度の発話を推奨します。背景音楽・強い反響・クリッピングを避け、前後の無音は短くします。

掲載結果では10.989958秒、PCM 24bitの音声を使用しました。原音は保持し、WAVEXから通常のRIFF WAVへのコンテナ変換を一度だけ行います。PCMサンプルの一致を確認し、原音・変換後のSHA-256と前処理内容を`artifacts/benchmark/inputs/reference.json`に保存します。

```bash
bash benchmark/setup.sh
```

計測専用のuv環境、共通入力、固定生成フレーム数、FP32／F16モデル、計測用workerを準備します。計測依存は[benchmark/uv.lock](../benchmark/uv.lock)に固定しています。GGUF変換用の追加依存は`gguf==0.17.1`です。計測用の変更は派生ソースと[パッチ](../benchmark/patches/)で管理します。

初期動作確認で取得する配布GGUFと、同じ公式重みから変換するベンチマーク用GGUFは区別しています。FP32基準にF16配布物を再拡張して使いません。codecの形式変換とONNXのFP32／FP16変換は環境構築時に行います。

## 測定と集計

```bash
bash benchmark/run.sh --preflight
bash benchmark/run.sh
benchmark/.venv/bin/python benchmark/report.py
benchmark/.venv/bin/python benchmark/validate_results.py
```

事前検証ではGPU選択、生成条件、出力PCM、計測追加の影響を確認します。測定の範囲・精度・キャッシュの定義は[測定方法](methodology.ja.md)を参照してください。

実装・設定を絞って実行できます。完了した測定はJSONLのキーで識別し、再実行時にスキップします。

```bash
bash benchmark/run.sh --engines speech.cpp audio.cpp
bash benchmark/run.sh --cases speech.cpp-mf-vulkan-fp32
bash benchmark/run.sh --retry-failed
```

失敗した条件は通常の再開では再試行しません。原因の修正後に`--retry-failed`を指定します。通常速度を保持して詳細計測だけを補完する場合は`--refresh-components`を使用します。エラー・OOM・timeoutでは子プロセスを終了し、理由を記録して他の測定を続けます。

入力・参照音声・モデルlockのハッシュが変わると、既存データへの追記を停止します。異なる入力で測る場合は`artifacts/benchmark/`を別名へ退避し、新しい測定として準備し直してください。`benchmark/report.py`は公開用の`benchmark/results/`を更新するため、既存結果を保持する場合は集計先の変更または事前退避が必要です。

## speech.cppのコミット間A/B比較

主比較の測定・検証を完了させてから、比較対象をSHA指定で別にビルドします。固定submoduleと主比較の結果は更新しません。

```bash
source scripts/environment.sh
benchmark/.venv/bin/python benchmark/speech_ab_build.py --sha 6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8
benchmark/.venv/bin/python benchmark/speech_ab.py
benchmark/.venv/bin/python benchmark/speech_ab_validate.py
```

短文／長文・固定8秒／24秒で比較し、事前に定めた差の判定条件を満たす場合に全20文章へ拡張します。[A/Bの測定方法](methodology.ja.md#speechcppのコミット間ab比較)と[結果](../benchmark/results/speech-ab-6bd7b027/RESULTS.ja.md)を参照してください。途中再開に対応し、判定のみ再計算する場合は`--reconsider-expansion`、集計のみの場合は`--report-only`を指定します。

このA/Bアダプターは上記コミットを対象とします。別コミットで上流APIや入れ子のggmlが変わった場合は、アダプターと計測パッチの調整が必要です。

## PyTorch／ONNX Runtime更新のA/B比較

主比較を構築済みの環境で実行します。元のvenv・lock・エンジン・モデル・主比較の生データは更新しません。

```bash
source scripts/environment.sh
python3 benchmark/runtime_ab_setup.py
benchmark/.venv/bin/python benchmark/runtime_ab_provenance.py
benchmark/.venv/bin/python benchmark/runtime_ab.py
benchmark/.venv/bin/python benchmark/runtime_ab_validate.py
```

専用環境は`.venvs/runtime-ab-20261015/`、生データ・依存バージョン・コマンド・モデルハッシュは`artifacts/runtime-ab-20261015/`、公開結果は`benchmark/results/runtime-ab-20261015/`に保存します。2026-10-15に公式配布元の最新安定版を照会し、`environments/runtime-ab-20261015/versions.json`に固定しました。全依存の実測freezeも同じディレクトリのrequirementsへ保存しています。PyTorchのCUDA runtimeは13.0です。Python 3.10用の最新版ONNX Runtime wheelがないため、ONNX環境は3.11を使います。

`--preflight-only`でPCM・計測の事前検証のみ、`--report-only`で集計のみを実行できます。測定は完了した要求単位で再開できます。新旧の依存一覧と元環境のRECORD・設定のハッシュを保存し、最後に一致を検証します。ONNXグラフは再exportせず、同じRust実行ファイルで共有ライブラリだけを切り替えます。ONNX／ONNXScript最新版も専用環境へ導入しますが、推論速度の比較対象はONNX Runtimeです。

小規模比較は短文・長文、2秒・8秒、両キャッシュ、FP32・実用精度です。3ラウンドで順序を交互にし、差が同方向で再現して階層bootstrap 95%区間が0を跨がないエンジン・モデル・精度を本測定へ拡張します。本測定は主比較と同じ全20文章のmatched／auto／text_scalingおよび2・4・8・16・24秒のaudio_scaling、新旧両版・両キャッシュ、各条件warmup 2回・通常5回・詳細5回です。

## 保存先

| 場所 | 内容 | Git管理 |
|---|---|---|
| `engines/` | 固定した7実装と入れ子のsubmodule | 対象 |
| `engines.lock.json`／`models.lock.json` | ソースSHA・モデルrevision | 対象 |
| `environments/`／`benchmark/uv.lock` | Python依存定義・lock・A/B用の固定requirements | 対象 |
| `benchmark/config.json`／`benchmark/prompts.json` | 共通条件・入力文章 | 対象 |
| `benchmark/patches/` | 計測用パッチ | 対象 |
| `benchmark/results/` | 集計CSV・グラフ・日本語レポート | 対象 |
| `artifacts/runtime-ab-20261015/`／`artifacts/mf-runtime-cause-20261015/` | Runtime A/Bの生データ・GPUトレース・依存freeze・監視ログ | 対象外 |
| `artifacts/mf-runtime-cause-20261015/report/` | 大きい演算別CSV・公開CSVと重複する集計JSON | 対象外 |
| `.venvs/`／`benchmark/.venv/` | 分離したPython環境 | 対象外 |
| `toolchains/`／`.cache/`／`cache/` | ローカルSDK・各種キャッシュ | 対象外 |
| `models/` | 固定revisionのモデル | 対象外 |
| `artifacts/build/`／`artifacts/codec/`／`artifacts/onnx/` | 単体実行用のビルド・変換結果 | 対象外 |
| `artifacts/voices/`／`artifacts/smoke/` | 初期動作確認用の参照音声・生成WAV | 対象外 |
| `artifacts/benchmark/source/`、`artifacts/benchmark/build/`、`artifacts/benchmark/models/` | 計測用派生ソース・ビルド・重み | 対象外 |
| `artifacts/benchmark/inputs/` | 共通参照音声・ハッシュ・固定生成フレーム数 | 対象外 |
| `artifacts/benchmark/results.jsonl`、`artifacts/benchmark/preflight.jsonl` | 測定生データ・事前検証 | 対象外 |
| `artifacts/benchmark/telemetry.jsonl` | 1秒間隔の温度・クロック・他GPUプロセス情報 | 対象外 |
| `artifacts/benchmark/provenance.json` | モデルハッシュ・依存・コマンド・ビルド設定 | 対象外 |
| `artifacts/benchmark/wav/` | 品質調査などに再利用する代表出力 | 対象外 |
| `artifacts/speech-ab/` | A/B用ソース・ビルド・測定生データ・代表WAV | 対象外 |
| `logs/`／`artifacts/*results.json`／`artifacts/environment.json` | 構築・単体確認ログと環境記録 | 対象外 |

`telemetry.jsonl`の1秒間隔の補助記録と、VRAMの10ms間隔の観測は別です。生データ・環境記録には実行時のパスなどを含むため、公開用の集計ファイルと分けて扱います。

コミットには再現用コード・依存定義と、`benchmark/results/`のレポート・主要CSV・図・検証結果を含めます。仮想環境、キャッシュ、モデル、ログ、生データ、一時ファイルは`.gitignore`で除外します。旧版の集計処理が`benchmark/results/`へ出力する`operators.csv`、`summary.json`、GPUトレース、JSONLも除外します。
