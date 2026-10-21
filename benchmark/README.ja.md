# Irodori-TTS 性能比較

7実装のRF／MF・CUDA／Vulkanを、固定したモデルと共通入力で比較します。対象コミット、ハードウェア、重み精度の概要は[ルートREADME](../README.md)を参照してください。

詳細な説明は以下にまとめています。

- [環境構築・実行・途中再開・保存先](../docs/usage.ja.md)
- [測定条件・指標の定義・精度の注意点](../docs/methodology.ja.md)
- [主比較の結果：表とグラフ](results/RESULTS.ja.md)
- [speech.cppのコミット間A/B比較](results/speech-ab-6bd7b027/RESULTS.ja.md)
- [PyTorch／ONNX Runtime更新のA/B比較](results/runtime-ab-20261015/RESULTS.ja.md)
- [公式MFの速度低下：段階別計測と原因切り分け](results/mf-runtime-cause-20261015/RESULTS.ja.md)
- [自動品質評価の結果：読み・話者再現性・追加診断](results/quality-20261019/RESULTS.ja.md)
- [speech.cpp更新後：JVS100話者・1クリップ参照と識別率低下の原因](results/jvs-latest-20261019/RESULTS.ja.md)
- [品質評価の条件・実行手順](quality/README.ja.md)

共通設定は[config.json](config.json)、入力20文章は[prompts.json](prompts.json)、計測用の変更は[patches/](patches/)に保存しています。モデル・参照音声・ビルド成果物・測定生データは同梱しません。
