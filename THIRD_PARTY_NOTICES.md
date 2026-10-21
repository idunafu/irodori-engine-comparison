# 第三者の著作権・ライセンス表示

## speech-bench

- 上流：<https://github.com/nyosegawa/speech-bench>
- 使用コミット：`326441fd1a145d7245c7c9462f322bae9d2e9a07`（[品質評価の固定設定](benchmark/quality/config.json)）
- ライセンス：MIT
- 著作権表示：Copyright (c) 2026 Sakasegawa
- [当該コミットのLICENSE](https://github.com/nyosegawa/speech-bench/blob/326441fd1a145d7245c7c9462f322bae9d2e9a07/LICENSE)

品質評価では、`prompts/speak-ja-JP.json` の日本語20文を使用します。また、[speech_bench_worker.mjs](benchmark/quality/speech_bench_worker.mjs)から、上流のWAV読み込み（`src/core/wav.ts`）、日本語の正規化・読み戻し採点（`src/measure/heard.ts`、`src/measure/scoring.ts`）、音高・有声時間解析（`src/analysis/pitch.ts`、`src/analysis/speaker.ts`）を利用します。

[prepare.py](benchmark/quality/prepare.py)は固定コミットのソースとLICENSEをGit管理外の `artifacts/quality/assets/speech-bench/` に取得します。[build.py](benchmark/quality/build.py)および[validate.py](benchmark/quality/validate.py)では、上流の `heard.ts` に `export { heardUnits };` を追加した派生ファイル `quality-heard.ts` を同じディレクトリに生成し、上限なしCERの計算に使います。評価用文章を含む結果や派生ソースを再配布する際にも、以下の著作権表示と許諾文を添付してください。

以下は上流LICENSEの全文です。この表示はspeech-bench由来の内容に適用されます。

```text
MIT License

Copyright (c) 2026 Sakasegawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
