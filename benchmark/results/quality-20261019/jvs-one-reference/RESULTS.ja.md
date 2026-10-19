# 1クリップ参照の再現性差の調査

確認できた主な差は、参照音声のリサンプル方式と、Vulkan codec内のF16演算経路。
モデル・codecの重みをFP32にするだけでは、Vulkanの内部F16への丸めは除けない。
同じ公式PCMでのlatent誤差は、既定Vulkanで相対L2約0.09379、内部F16経路を無効にすると約0.00000310。
同じlatentから作る話者特徴量のcosineは約0.999996で、パッチ化や平均tokenの扱いは公式と対応している。
MFの生成長・ノイズ共通実験では、公式との差は通常参照-0.003280、公式PCM-0.001248、公式latent-0.000182、公式話者特徴量-0.000013。
数値はこの固定20話者での点推定。元の100話者の低下量と直接混ぜず、区間も参照する。

既存の固定診断20話者で比較。MF・RFとも通常条件は5文×5 seed。
追加の統制実験はすべてFP32重み・公式の固定生成長・共通FP32初期ノイズ。既定Vulkanの内部演算経路は維持。
MFは20話者×5文×5 seed（各500音声）、RFは同じ20話者×先頭1文×5 seed（各100音声）。
保存済み100話者centroidを共通に用い、採点器・元結果は変更しない。

| 通常条件・同じ20話者 | CAM++類似度 | 話者Top-1 |
|---|---:|---:|
| official-mf-bf16 | 0.748873 | 89.20% |
| speech-mf-f16 | 0.743599 | 86.80% |
| official-mf-fp32 | 0.749403 | 88.60% |
| speech-mf-fp32 | 0.743554 | 86.80% |
| official-rf-bf16 | 0.753109 | 87.60% |
| speech-rf-f16 | 0.747894 | 86.00% |
| official-rf-fp32 | 0.753062 | 87.60% |
| speech-rf-fp32 | 0.747853 | 86.00% |

| 中間値の比較 | 平均cosine | 平均相対L2誤差 |
|---|---:|---:|
| mf / codec_same_pcm | 0.99525145 | 0.093790 |
| mf / preprocessed_pcm | 0.99990750 | 0.012473 |
| mf / reference_latent_total | 0.98836288 | 0.151464 |
| mf / resampler | 0.99990750 | 0.012406 |
| mf / speaker_same_latent | 0.99999609 | 0.002352 |
| mf / speaker_same_pcm | 0.98122490 | 0.185824 |
| mf / speaker_total | 0.93492242 | 0.352820 |
| rf / codec_same_pcm | 0.99525145 | 0.093790 |
| rf / preprocessed_pcm | 0.99990750 | 0.012473 |
| rf / reference_latent_total | 0.98836288 | 0.151464 |
| rf / resampler | 0.99990750 | 0.012406 |
| rf / speaker_same_latent | 0.99999609 | 0.002352 |
| rf / speaker_same_pcm | 0.98122490 | 0.185824 |
| rf / speaker_total | 0.93492242 | 0.352820 |

speaker_same_latentは同一latentを話者encoderへ渡した比較。codec_same_pcmは同一前処理波形をcodecへ渡した比較。
reference_latent_totalとspeaker_totalは各実装の独自前処理を含む。

ラウドネス目標は双方-16。フレーム数と話者token数も一致し、先頭の平均tokenと4フレーム単位のpatch化は同じ。
リサンプルは公式がtorchaudioの既定Hann（幅6・rolloff 0.99）、speech.cppがKaiser（幅64・rolloff 0.9475937）。
公式の再計算latentと、元の評価で保存したlatentの最大差は全20話者で0。
同じlatentからの話者特徴量はほぼ一致し、参照波形からlatentまでの段階で差が拡大している。

| 同一公式PCMを使ったcodec比較 | latent相対L2誤差 |
|---|---:|
| cpu (20話者) | 0.00000269 |
| default (20話者) | 0.09378977 |
| no_coopmat (20話者) | 0.09378895 |
| no_f16 (20話者) | 0.00000310 |

codecの重み型は全経路でF32。CPUの100frame分割と一括encodeは同値。
GPUのno_coopmatはcooperative matrixを無効化、no_f16はさらにF16演算経路も無効化する派生診断。
元のベンチマーク設定やsubmoduleにこれらの変更は適用していない。

| 統制条件（全てFP32重み） | 類似度 | Top-1 |
|---|---:|---:|
| mf / official | 0.745907 | 89.00% |
| mf / speech_native | 0.742627 | 87.00% |
| mf / speech_official_pcm | 0.744660 | 91.00% |
| mf / speech_official_latent | 0.745726 | 88.80% |
| mf / speech_official_state | 0.745894 | 88.80% |
| rf / official | 0.751974 | 92.00% |
| rf / speech_native | 0.748492 | 93.00% |
| rf / speech_official_pcm | 0.753117 | 97.00% |
| rf / speech_official_latent | 0.752263 | 92.00% |
| rf / speech_official_state | 0.752179 | 92.00% |

speech_native: 独自前処理・codec・話者encoder。speech_official_pcm: 公式前処理波形に交換。
speech_official_latent: 公式codec latentに交換。speech_official_state: 公式話者特徴量に交換。
信頼区間は話者単位の対応ありbootstrap（2,000回）。詳細はcontrolled-comparisons.csv。
FP32重みでの通常条件の差は点推定として残るが、20話者の95%区間は0を含む。
リサンプル・codec・話者encoderの中間値と交換実験で差の発生箇所を確認したもので、全100話者での補正後スコアは未測定。
短い参照で影響が強い理由を情報量の少なさと推測できるが、参照長別の中間値交換までは実施していない。
自動話者評価であり、聴感上の自然さ・同一性を直接保証しない。

確認した固定版ソース：
- engines/Irodori-TTS/irodori_tts/codec.py: encode_waveform内の既定resample、normalize、deterministic mean
- engines/speech.cpp/src/common/resample.cpp: Kaiserの幅・rolloff
- engines/speech.cpp/src/families/irodori-tts/codec.cpp: codec encoderと窓分割
- engines/speech.cpp/src/families/irodori-tts/speaker-encoder.cpp: 4フレームpatch、1/6、RMSNorm、先頭mean token
- engines/speech.cpp/src/common/graph.h: 行列積はF32 accumulationを指定
- engines/speech.cpp/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp: FLOAT_TYPEの内部F16選択
参照・モデルのrevisionは元runのmanifest、診断コード等のhashはsource-audit.jsonに保存。
