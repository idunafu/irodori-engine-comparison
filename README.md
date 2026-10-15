# Irodori-TTS 実装ベンチマーク

Irodori-TTSの7実装について、同じモデル・入力テキスト・参照音声・生成条件で推論速度とVRAM使用量を比較するベンチマークです。RF（Rectified Flow）／MF（MeanFlow）、CUDA／Vulkan、重み精度、条件キャッシュの有無を分けて測定します。

- [比較結果：測定表・スケーリンググラフ](benchmark/results/RESULTS.ja.md)
- [speech.cppのコミット間A/B比較](benchmark/results/speech-ab-6bd7b027/RESULTS.ja.md)
- [構築・実行手順](docs/usage.ja.md)／[測定方法・指標の定義](docs/methodology.ja.md)

## 対象実装と固定コミット

主比較は2026年10月13日に固定した以下のコミットを使用します。完全なSHAは[engines.lock.json](engines.lock.json)に記録し、入れ子のsubmoduleも各親コミットの指定に従います。

| 実装 | コミット | モデル | バックエンド | 実用的な設定のモデル重み精度 |
|---|---|---|---|---|
| [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) | `89f9d8fbd4d5` | RF・MF | CUDA | BF16 |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `f1c299b73c06` | RF | CUDA・Vulkan | F16 |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | `c6f28804907e` | RF・MF | CUDA・Vulkan | F16 |
| [sttts-gui](https://github.com/kjranyone/sttts-gui) | `acce71245654` | RF・MF | Vulkan | FP32 |
| [Irodori-TTS-burn](https://github.com/Sanzentyo/Irodori-TTS-burn) | `ad11d15c0aed` | RF・MF | Vulkan（WGPU） | FP32 |
| [irodori-onnx](https://github.com/utyujinn/irodori-onnx) | `386eaf48a3b4` | MF | CUDA | FP16 |
| [Irodori-TTS-Optimiz](https://github.com/kuwacom/Irodori-TTS-Optimiz) | `4e5b933df4d6` | RF | CUDA | BF16 |

14通りのモデル・バックエンドの組み合わせを、FP32共通基準と実用設定の計24設定で測定しました。sttts-gui／Burnは両設定がFP32のため測定を共有します。sttts-guiはLinux用の`sttts-say`と同じTTSエンジンを対象としています。

speech.cppの追加A/B比較は、上記固定版と`6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8`を別ビルドで比較しています。主比較のsubmoduleと結果は保持しています。

## 検証環境とビルド

| 項目 | 使用環境・設定 |
|---|---|
| OS | Ubuntu 24.04 LTS、Linux 7.0.0-38-generic、x86_64 |
| CPU | AMD Ryzen 9 9950X（16コア／32スレッド）。計測時の演算スレッド数は8 |
| GPU | NVIDIA GeForce RTX 5090、VRAM 32GB |
| GPUドライバ | 610.57.04 |
| C++／CUDA | GCC 13.3.0、CMake 3.28.3、CUDA Toolkit 13.3.73 |
| Vulkan | Vulkan SDK 1.4.357.0、既存のNVIDIAドライバを利用 |
| Python | Python 3.10.20、uv 0.12.14、実装ごとに分離した環境 |
| Python GPU runtime | PyTorch 2.10.0+cu128／ONNX Runtime 1.23.2（CUDA 12系・cuDNN 9） |
| Rust | rustc／Cargo 1.99.0、上流のCargo.lockを使用 |

C++はRelease（`-O3 -DNDEBUG`）、CUDAは`CMAKE_CUDA_ARCHITECTURES=120`を指定し、CUDA／Vulkanを別々にビルドしています。Rustは`--release --locked`、公式Python版とOptimizは上流の`uv.lock`と`cu128` extraを使用します。制御可能なTF32は測定時に無効にしています。

計測用workerとパッチは[benchmark/](benchmark/)に置き、派生ソース・バイナリを生成します。submoduleの追跡ファイルは変更しません。ビルド手順と環境を調整する箇所は[構築・実行手順](docs/usage.ja.md)を参照してください。

## 使用モデルと精度

| 用途 | 公式チェックポイント | 固定revision（短縮） |
|---|---|---|
| RF | Aratako/Irodori-TTS-v4.1-Small | `2b28324dc263` |
| MF | Aratako/Irodori-TTS-v4.1-Small-MF | `ccc78f5d480b6` |
| Codec | Aratako/Semantic-DACVAE-Japanese-32dim | `47376ee24834` |

完全なrevisionと取得対象は[models.lock.json](models.lock.json)に記録しています。ベンチマーク用GGUFは同じ公式チェックポイントからFP32／F16へ変換しています。ONNXも同じ重みからFP32／FP16を作成します。

FP32基準はモデル・codecの重みをFP32に揃えます。実用設定のcodecは原則FP32で、ONNXのみdecoderがFP16、参照音声encoderはFP32です。整数への量子化は追加していません。重み精度が同じでも、内部キャスト・積算・kernelは実装ごとに異なります。[演算精度の注意点](docs/methodology.ja.md#条件)を併せて参照してください。

## 検証方法

共通条件はbatch size 1、候補数1、seed 42、RF 40ステップ／MF 4ステップです。captionは「自然な日本語で、落ち着いた口調で話す。」、RFはindependent CFG（text／caption＝3、speaker＝5、適用区間0.5〜1.0）、linear scheduleとします。透かし・ストリーミング出力・末尾トリム・追加の無音除去は無効です。設定は[config.json](benchmark/config.json)に保存しています。

- **入力と生成長**：[日本語20文章](benchmark/prompts.json)をBOS込み8〜15、16〜31、32〜63、64〜95、96〜127トークンの5群に各4件配置。主比較の生成フレーム数は公式RF・FP32の予測で固定し、全実装で共有します。自動長予測、全20文章の8秒固定、代表文章の2・4・8・16・24秒固定も別測定します。
- **参照音声**：全実装に同じ[つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/)の"VOICEACTRESS100_054.wav"を24bitに変換したもの（10.989958秒、モノラル、48kHz、PCM 24bit）を使用。共通RIFF WAVへの変換ではPCMサンプルを保持し、原音・変換後のハッシュを保存しています。
- **時間**：モデルロード・初回推論は独立workerで3回。各入力・条件を2回warmup後に5回測定し、中央値・最小値・最大値・ばらつきを保存します。要求処理開始からCPU上のPCM完成までを推論時間とし、WAV保存を除外します。全サンプリング、実際のDiT forward、各エンコード、DACVAE decodeはGPU同期を伴う別実行で測定します。RTFは推論時間÷出力秒数です。
- **キャッシュとVRAM**：条件を毎回再エンコードする場合と、各実装の既存キャッシュを使う場合を分けます。GPU処理は逐次実行し、NVMLで10ms間隔のVRAM観測ピークを記録します。ロード後・warmup後の常駐量、ロード中・推論中のピークを区別し、温度・クロック・他プロセスも記録します。OS・ドライバのディスクキャッシュは保持します。
- **妥当性確認**：GPU選択・精度・生成条件を事前確認し、計測有無によるPCM差、48kHz・有限値・非無音・指定生成長を検証します。ONNXは上流の数値比較も実行します。

固定生成長の主比較は全24設定・両キャッシュ条件で20文章すべて成功しました。自動長予測とスケーリングを含む3,120条件中、2,992条件が成功し、128条件はspeech.cppの自動生成長が30秒の上限を超えて拒否されました。[結果レポート](benchmark/results/RESULTS.ja.md)では失敗・拒否も含めて記録しています。

## 使い方

必要な開発ツール・GPUドライバを用意し、リポジトリルートから実行します。CUDA／Vulkanの導入先は自動検出または環境変数で指定します。CUDAアーキテクチャの既定値はRTX 5090向けの`120`のため、他環境では[事前条件と指定方法](docs/usage.ja.md#事前条件)を確認してください。

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
bash scripts/setup.sh

bash benchmark/setup.sh
bash benchmark/run.sh --preflight
bash benchmark/run.sh
benchmark/.venv/bin/python benchmark/report.py
benchmark/.venv/bin/python benchmark/validate_results.py
```

参照音声を差し替える場合は[benchmark/config.json](benchmark/config.json)の`reference`にリポジトリルートからの相対パスを指定します。掲載結果と同じ入力での追試には元の参照音声が必要です。

モデル・環境・ビルド・ログ・生データ・生成音声はGit管理外へ保存し、集計CSV・グラフを[benchmark/results/](benchmark/results/)に出力します。個別実装の起動、途中再開、出力ファイル、speech.cpp A/Bの実行方法は[詳細手順](docs/usage.ja.md)を参照してください。
