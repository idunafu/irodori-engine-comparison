# 測定方法と指標の定義

[概要](../README.md)／[構築・実行手順](usage.ja.md)／[主比較の結果](../benchmark/results/RESULTS.ja.md)

固定した7実装の速度・メモリ使用量を、単一GPUで逐次測定します。以下のパス表記はリポジトリルートからの相対パスです。

## 条件

[benchmark/config.json](../benchmark/config.json)と[benchmark/prompts.json](../benchmark/prompts.json)が実行条件。20文章を5種類のトークン長の群に配置し、切り捨てずに使用する。seed 42、batch 1、RF 40ステップ、MF 4ステップ、caption「自然な日本語で、落ち着いた口調で話す。」。RFはindependent CFG、text/caption 3、speaker 5、時刻0.5〜1.0、linear schedule。透かし、末尾トリム、追加の無音除去は無効。

audio.cppの文字数による自動分割の閾値は `text_chunk_size=4096` とし、今回の全入力を1つのchunkで処理する。入力は全てモデルのトークン上限内であり、1文章を1回の40-step生成として扱う。分割によって固定生成長を複数回繰り返すことを防ぐ。

FP32のモデル重みを共通基準とし、実用設定を別集計する。GGUFは公式チェックポイントから変換し、F16配布物をFP32に戻さない。audio.cppの配布GGUFから利用するのは形式・codec設定のsidecarだけで、モデルおよびcodecのテンソルは公式ソースから取得する。codecはFP32を基本とし、ONNX実用設定のdecoderはFP16。内部の積算・attentionなどの演算精度は重み精度と区別する。

| 実装 | 活性値・内部キャストと積算 | FP32基準での注意点 |
|---|---|---|
| 公式版／Optimiz | モデルの浮動小数重み・bufferを指定dtypeへ移動。RMSNorm、位置／時刻埋め込み、duration出力などにFP32演算が残る | PyTorch matmul／cuDNNのTF32を無効化。BF16実用設定も全演算がBF16になるわけではない |
| speech.cpp | graphの入力・出力は主にF32。`src/common/graph.h` の行列積はF32積算を要求。GGMLのCUDA／Vulkan kernel選択は維持 | 重み精度とbackendの演算経路を区別する。maskなど一部の入力はF16 |
| audio.cpp | graphの入力・出力は主にF32。DiT Flash AttentionはF32積算を要求。attention maskはF16 | backendによる内部変換まで全てFP32へ固定していない |
| sttts-gui | Burn/WGPUのF32テンソルを使用、GPUメモリ管理は上流のExclusivePages | kernel内部の選択・最適化は上流のまま。全kernelの命令レベルでの積算精度は未検証 |
| Burn | `WgpuFloatPrecision::Fp32`を明示。WGSLのF32経路を使用 | kernel内部の選択・最適化は上流のまま。全kernelの命令レベルでの積算精度は未検証 |
| ONNX | FP32 graphまたは上流のFP16変換graph。FP16 graphの入出力はF32を保持しCastを含む。reference codec encoderは常にFP32 | CUDA EP、`NVIDIA_TF32_OVERRIDE=0`。一部の小さな演算はORTのCPU kernelを利用し得る |

従ってFP32表は「同じFP32重み・codecを使う基準」であり、全実装の各命令を同じ精度で実行した表ではない。これらの差を速度差と併せて考慮する。

既存submoduleを編集せず、[benchmark/build.py](../benchmark/build.py) が `artifacts/benchmark/source/` に派生ソースを作成する。適用内容は [benchmark/patches/](../benchmark/patches/) に保存する。Rust依存は上流Cargo.lock、Pythonの計測依存は[benchmark/uv.lock](../benchmark/uv.lock)に固定する。変換用追加依存はgguf 0.17.1。モデル・コードのrevisionはルートのlockファイルを使う。

## 測定の読み方

| 指標 | 測定範囲・定義 |
|---|---|
| モデルロード | 重み・tokenizer・codecの読み込みからGPU配置完了まで。プロセス起動の時間と区別する |
| 初回推論 | 新しいworkerで事前warmupなしの要求からCPU上のPCM完成まで |
| warm推論 | モデル常駐・warmup後の要求処理からCPU上のPCM完成まで。WAV保存時間を除外 |
| 全サンプリング／loop | 条件エンコード後から最終latent完成まで。KV準備・ノイズ・更新を含む全体と、ループ単体を区別 |
| DiT forward | 実呼び出しごとのGPU完了まで。呼び出し数・CFG分岐・batch・形状を併記 |
| テキスト／caption | tokenizeとエンコーダを分離。共有処理は重複加算しない |
| 参照音声 | 前処理・DACVAE encoder・speaker encoderを分離 |
| DACVAE decode | 最終latentからCPU上のPCM完成まで。転送を含む |
| RTF | warm推論時間 ÷ 出力PCMの秒数 |
| VRAM | ロード後・warmup後の常駐量、ロード中・推論中の観測ピーク。allocatorの値は補助指標 |

ロード時間と初回推論は3個の新しいプロセスで測る。ディスクキャッシュ・ドライバのshader cacheは消さない。初回時間はそのプロセスでの初回であり、ドライバを初期化した直後の完全なcold状態を意味しない。

モデル常駐後、各条件で2回warmupし、通常のwall timeを5回、GPU完了を待つ段階別計測を別に5回実施する。通常推論の範囲は要求の処理開始からCPUのPCM完成までで、WAV保存とプロセス間通信を除く。通信込みの時間も生データに保存する。RTFはこの通常推論時間を出力PCMの秒数で割った値。

同一生成長の比較は公式RF・FP32による予測フレーム数を全実装で共有する。自動長予測、8秒固定での入力長変化、代表文章で2/4/8/16/24秒の出力長変化も別測定する。自動予測の上限到達・超過も記録し、上限を超えたため拒否された条件は成功した条件と区別する。末尾トリムを無効にしているためRTFの分母に同じ定義を使える。

参照音声・captionを毎回再エンコードする条件と、公開API・実装の既存キャッシュを使う条件を分ける。公式Python版は公開ref_latent API、stttsはSamplingRequest.ref_latent、speech.cpp/Burn/ONNXは登録したvoice、Optimiz/audio.cppは上流の参照・条件キャッシュを使用する。実装にないcaption/speakerキャッシュを追加しない。キャッシュがない段階は通常どおり処理する。

RFではCFG分岐をbatchにまとめる実装と、個別forwardを使う実装がある。DiTの実呼び出し数・batch・形状を保存し、1回の時間と全サンプリング時間を区別する。ステップ数で総時間を割ってforward時間を推定しない。DiT先頭stepは各要求の最初のforwardであり、新しいプロセスの初回推論とは区別する。

Optimizの共有text/caption backboneとaudio.cpp/speech.cppの融合graphは、実処理の共有時間を記録する。個別encoderの時間は詳細計測だけで行う独立probeも記録する。probeは出力に使用せず、通常の速度測定に入らない。probeの合計は融合処理時間と一致するとは限らず、加算して総時間を作らない。GPU同期のある詳細計測の総時間を通常速度の代わりに使わない。

tokenizeと参照前処理も独立scopeで保存する。codecの外側scopeに含まれる前処理は、内側scopeの実測値を差し引いてDACVAE encode列に重複計上しない。Pythonではencoder本体とin_projを個別計測する。Burnはvoice登録ごとにcodec encoderを読み込む上流の処理を維持し、encoder load列に分離する。完了済みの通常速度を保持して詳細計測だけを補完する場合は `--refresh-components` を使う。

VRAMはNVMLのcomputeとgraphicsの対象PIDを確認し、同じPIDの値を重複加算しない。取得できない場合のみGPU全体のbaseline差分を使う。10ms間隔の最大値は観測ピークであり、間隔より短い一時的な確保を見逃す可能性がある。ロード時ピーク、推論時ピーク、warmup後の待機時使用量を分ける。PyTorch allocatorのallocated/reservedピークは補助指標で、NVMLとは混ぜない。

各caseの同じworkerでreencodeを先に、native_cacheを後に測る。後者では既に長い形状を処理したallocatorの確保領域が残るため、常駐VRAMの差を条件キャッシュ単体の容量と解釈しない。ロード直後、入力別の待機時、最大観測値、allocatorのallocated／reservedを併せて確認する。

CPUの演算スレッドは8。ONNX Runtimeは環境変数だけでは専用thread poolを制限できないため、各sessionでintra-op 8／inter-op 1を明示し、待機中のintra-op threadのspinningを無効にする。

GPUクロック・電力制限は変更しない。未対応の組み合わせ、条件未達、エラー、OOMは理由とともに記録し、CPUによる代替実行をGPU測定に含めない。

## 事前検証と数値比較

各実装・精度・バックエンドで、GPU選択、caption、ステップ数、固定生成長が反映されることを確認します。計測を無効にした出力と有効にした出力を同じ初期ノイズで比較し、計測追加によるPCM差を確認します。48kHz、有限値、非無音、指定フレーム数も検証します。実装間の波形一致や聴感品質は合格条件に含めません。

ONNXは上流のテキスト・tokenizer・合成・音声登録テストをFP32 CPUとFP16 CUDAで実行しました。共通ノイズによるPython／ONNX比較の波形相関はFP32で1.000000、FP16 CUDAで約0.989です。FP16の上流テストには`IRODORI_TOL=0.05`を使用します。CPUでの数値検証とCUDAでの動作確認は分けて扱います。

## speech.cppのコミット間A/B比較

固定版`c6f28804907ec47dc4d93b526cb7d3407020b417`と比較対象`6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8`を、別の派生ビルド・保存先で測定します。主比較の固定コミット、モデル、結果は更新しません。

F16モデル／FP32 codec、RF 40・MF 4ステップ、CUDA／Vulkan、p01（短文・15トークン）／p19（長文・114トークン）、8秒／24秒、両キャッシュ条件を使用します。caption・CFG・seed・CPUスレッド数・参照音声は主比較と同じです。

3ラウンドで旧→新／新→旧／旧→新と実行順を交互にし、各条件2回warmup・通常5回・詳細5回を測定します。通常時間の階層bootstrap 95%区間が0を跨がず、全ラウンドで同方向の差がある組み合わせは、差が小さくても全20文章へ拡張します。2%以上の差はCSVの`practical_speed_difference`列で別に区別します。VRAMの拡張判定は32MiBかつ3%以上の差が同方向で再現することを条件とします。

今回の全20文章への拡張対象はRFのCUDA／Vulkanです。MFは短文／長文の比較を掲載しています。[A/B結果レポート](../benchmark/results/speech-ab-6bd7b027/RESULTS.ja.md)に判定値と測定表を記載しています。
