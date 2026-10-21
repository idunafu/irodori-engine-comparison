# 最新speech.cppのJVS100話者・1クリップ参照評価

[完了した評価と原因診断](../../results/jvs-latest-20261019/RESULTS.ja.md)に、公式・旧cppとの比較、全件の誤認先、共通条件による2,400音声の追加診断を保存しています。「最新」は2026年10月19日の固定版を指します。

speech.cpp `eacabcd891fefc2c3de3ca2479ff84e60f2679aa`、ggml `a655cf6910770404213bc338e622cd82b08ca277` を別ソース・別ビルドで固定する。
既存submodule、モデル、公式評価結果を保持する。モデルは既存のMF/RF F16 GGUFをそのまま使用する。

新規生成は100話者×5文章×5 seed×MF/RFの5,000件。参照は既存と同じnonpara30先頭1クリップ。
既存の公式BF16と旧speech.cpp F16の生成・Whisper/CAM++採点を各5,000件再利用する。
各実装の自動長予測、native seed、候補数1、caption空、末尾トリム・透かしなし。
照合用100話者centroidを再利用し、話者類似度・Top-1・Whisper CERを集計する。

起動前に出力ディレクトリを作成し、リポジトリルートで実行する。以下は測定時の起動コマンド。

```bash
mkdir -p artifacts/quality/runs/20261019-jvs-one-latest-eacabcd
tmux new-session -d -s irodori-jvs-latest 'exec env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venvs/quality/bin/python benchmark/quality/jvs_latest/run.py >> artifacts/quality/runs/20261019-jvs-one-latest-eacabcd/runner.log 2>&1'
```

入力検証→最新版ビルド→10話者試行→容量・時間見積もり→全件生成・採点→集計を自動実行する。
GPUロック `artifacts/gpu.lock` を共有し、30秒ごとにstatus.jsonを保存する。標準出力・エラーはログへ保存する。

状態確認：

```bash
cat artifacts/quality/runs/20261019-jvs-one-latest-eacabcd/status.json
tail -n 20 artifacts/quality/runs/20261019-jvs-one-latest-eacabcd/runner.log
```

停止はstatus.jsonのpidへSIGTERMを送る。子worker・ビルド処理も終了する。
同じ起動コマンドで途中再開する。条件・保存物のhashが変われば拒否する。
失敗要求の再試行は `--retry-failed` の明示時のみ。OOM・timeoutでworkerを再起動し、3回連続すると異常終了する。

レポートの再集計：

```bash
.venvs/quality/bin/python benchmark/quality/jvs_latest/run.py --report-only
```

最終レポートは出力フォルダーの `report/LATEST.ja.md`。
各seed、平均・母標準偏差、公式・旧版との差と話者単位の95% bootstrap区間をCSVへ保存する。

## 原因診断と再集計

`confusion.py`は保存済みCAM++埋め込みと元の100話者centroidから、全15,000音声の予測話者・順位・スコア差、話者別の誤認先と公式との対応関係を再計算する。入力実音声600件の採点記録があればその識別結果も集計する。新たなGPU推論は行わない。

```bash
.venvs/quality/bin/python benchmark/quality/jvs_latest/confusion.py
```

`confusion_diagnostic.py`は混同結果から介入前に固定した悪化6話者と対照2話者について、共通生成長・ノイズ、参照リサンプル、モデル重み精度とVulkan内部演算を比較する。100話者全体の改善率を測る構成ではない。元のcentroid・モデル・評価環境・更新後cppバイナリが必要。

```bash
tmux new-session -d -s irodori-jvs-confusion 'exec env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venvs/quality/bin/python benchmark/quality/jvs_latest/confusion_diagnostic.py >> artifacts/quality/runs/20261019-jvs-one-latest-eacabcd/analysis/confusion/runner.log 2>&1'
```

生成と採点は保存済みの成功結果を再利用するため、完了した元runでは再生成せずに集計できる。30秒ごとの進捗、終了状態、ログは同ディレクトリへ保存し、SIGTERMで子workerも終了する。GPU生成では既存の`artifacts/gpu.lock`を共有する。失敗要求は自動で再試行しない。元の音声やSQLiteがない環境では、この手順による元runの再集計はできない。

## 成果物の公開とパスの扱い

公開結果は`benchmark/results/jvs-latest-20261019/`へまとめる。音声・モデル・コーパス・SQLite・埋め込み・ログ・ビルド・参照cacheは`artifacts/`へ保持し、Gitには含めない。

```bash
# 元runから相対パスの公開コピーと由来・検証記録を作成
python3 benchmark/quality/jvs_latest/publish.py
# 公開済みCSV・ハッシュ・生成条件を検証。元runやGPUは不要
python3 benchmark/quality/jvs_latest/publish.py --check
```

公開コピーのパスはリポジトリルートからの相対表記に変換し、外部Vulkan loaderの場所は`${Vulkan_LIBRARY}`へ置換する。`provenance.json`に元ファイルと公開ファイルのSHA-256と変換内容を記録する。元のmanifest・再開状態・音声・数値指標は変更しない。

公開コードは`run.py`のVulkan loaderを`scripts/native_toolchain.py`で自動検出するよう整理した。必要なら`Vulkan_LIBRARY`または`VULKAN_LIBRARY`で指定できる。SDKはリポジトリ内の固定版を使用する。測定時のmanifestは当時のコードハッシュを保持しており、公開コードで元runを再開すると条件不一致で拒否する。再生成は`--out`で別の出力先を指定する。

```bash
mkdir -p artifacts/quality/runs/jvs-one-eacabcd-rerun
tmux new-session -d -s irodori-jvs-rerun 'exec env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venvs/quality/bin/python benchmark/quality/jvs_latest/run.py --out artifacts/quality/runs/jvs-one-eacabcd-rerun >> artifacts/quality/runs/jvs-one-eacabcd-rerun/runner.log 2>&1'
```

再生成には固定コミットの別checkoutとsubmodule、既存評価の公式・旧cpp結果、モデル、JVS、品質評価環境が必要。上記はそれらを準備済みの環境用であり、公開結果だけから生成を開始するコマンドではない。
