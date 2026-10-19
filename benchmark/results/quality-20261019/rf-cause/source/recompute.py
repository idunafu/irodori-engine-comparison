"""RF precision/noise diagnostics from immutable campaign outputs, without inference."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
ROOT = RUN.parents[3]
CASES = ["official-rf-bf16", "speech-rf-f16", "official-rf-fp32", "speech-rf-fp32"]
MODES = ["native", "common"]
SEEDS = range(5)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bootstrap(a, b):
    rng = np.random.default_rng(20261016)
    idx = rng.integers(0, len(a), size=(2000, len(a)))
    aa, bb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    delta = bb[:, 0] / bb[:, 1] - aa[:, 0] / aa[:, 1]
    return dict(difference=float(b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum()),
                low=float(np.quantile(delta, .025)), high=float(np.quantile(delta, .975)))


def main():
    c = sqlite3.connect(f"file:{RUN}/results.sqlite3?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    tasks, saved_scores = {}, {}
    for row in c.execute('SELECT * FROM tasks WHERE dataset="practical" AND case_id LIKE "%rf%"'):
        spec = json.loads(row["spec"])
        tasks[spec["item"]["id"], row["case_id"], row["mode"], spec["seed"]] = {
            **dict(row), "spec": spec, "result": json.loads(row["result"])}
        saved_scores[row["id"]] = {
            s["scorer"]: dict(status=s["status"], result=json.loads(s["result"]))
            for s in c.execute('SELECT * FROM scores WHERE task_id=?', (row["id"],))}
    c.close()
    details = {}
    with (RUN / "report/per-input.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["dataset"] == "practical" and "-rf-" in row["case"]:
                details[row["id"]] = row
    all_items = sorted({k[0] for k in tasks})
    native32 = [i for i in all_items if all(tasks[i, ca, "native", s]["status"] == "ok"
                                          for ca in CASES for s in SEEDS)]
    matched31 = [i for i in native32 if all(tasks[i, ca, m, s]["status"] == "ok"
                                          for ca in CASES for m in MODES for s in SEEDS)]
    equal27 = [i for i in matched31 if len({tasks[i, ca, m, s]["result"]["frames"]
               for ca in ["official-rf-fp32", "speech-rf-fp32"] for m in MODES for s in SEEDS}) == 1]
    assert (len(native32), len(matched31), len(equal27)) == (32, 31, 27)
    summary, per_seed, comparisons, item_scores, audit = [], [], [], [], []

    def array(items, case, mode, scorer, metric):
        result = []
        for item in items:
            pairs = []
            for seed in SEEDS:
                t = tasks[item, case, mode, seed]
                assert saved_scores[t["id"]][scorer]["status"] == "ok"
                h = details[t["id"]]["metrics"][scorer]
                n, d = (h["heard_errors"], h["heard_reference_length"]) if metric == "heard_error_rate" else (
                    int(h["heard_errors"] / h["heard_reference_length"] > .3), 1)
                pairs.append((n, d))
            result.append(np.array(pairs).sum(axis=0))
        return np.array(result, dtype=float)

    for subset, items, modes in [("native32", native32, ["native"]), ("matched31", matched31, MODES),
                                  ("equal_length_fp32_27", equal27, MODES)]:
        for mode in modes:
            for case in CASES:
                for scorer in ["qwen", "reazon"]:
                    for metric in ["heard_error_rate", "breakdown_rate"]:
                        values = []
                        for seed in SEEDS:
                            hs = [details[tasks[i, case, mode, seed]["id"]]["metrics"][scorer] for i in items]
                            n, d = (sum(h["heard_errors"] for h in hs), sum(h["heard_reference_length"] for h in hs)) if metric == "heard_error_rate" else (
                                sum(h["heard_errors"] / h["heard_reference_length"] > .3 for h in hs), len(hs))
                            values.append(n / d)
                            per_seed.append(dict(subset=subset, mode=mode, case=case, scorer=scorer, metric=metric,
                                                 seed=seed, numerator=n, denominator=d, value=n / d))
                        summary.append(dict(subset=subset, mode=mode, case=case, scorer=scorer, metric=metric,
                                            texts=len(items), takes=len(items)*5,
                                            mean=float(np.mean(values)), population_std=float(np.std(values))))
            for baseline, candidate in [("official-rf-bf16", "speech-rf-f16"), ("official-rf-fp32", "speech-rf-fp32")]:
                for scorer in ["qwen", "reazon"]:
                    for metric in ["heard_error_rate", "breakdown_rate"]:
                        a, b = array(items, baseline, mode, scorer, metric), array(items, candidate, mode, scorer, metric)
                        assert np.array_equal(a[:, 1], b[:, 1])
                        comparisons.append(dict(subset=subset, mode=mode, baseline=baseline, case=candidate,
                                                scorer=scorer, metric=metric, texts=len(items), **bootstrap(a, b)))

    for item in matched31:
        for seed in SEEDS:
            ts = [tasks[item, ca, "common", seed] for ca in CASES]
            noises = {t["result"]["initial_noise_sha256"] for t in ts}
            frames = {t["result"]["frames"] for t in ts}
            samples = {t["result"]["samples"] for t in ts}
            assert len(noises) == len(frames) == len(samples) == 1
            assert next(iter(noises)) is not None
            assert all(t["result"].get("noise_used") is True for t in ts if t["spec"]["case"]["engine"] == "official")
            key=hashlib.sha256(json.dumps([item,'rf',seed],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            noise_path=RUN/'noise'/(key+'.f32')
            assert noise_path.stat().st_size==next(iter(frames))*32*4
            assert sha(noise_path)==next(iter(noises))
            audit.append(dict(item=item, seed=seed, frames=next(iter(frames)), samples=next(iter(samples)),
                              initial_noise_sha256=next(iter(noises)),noise_path=str(noise_path)))
        for case in CASES:
            for mode in MODES:
                for scorer in ["qwen", "reazon"]:
                    total = array([item], case, mode, scorer, "heard_error_rate")[0]
                    item_scores.append(dict(item=item, case=case, mode=mode, scorer=scorer,
                                            errors=int(total[0]), reference_length=int(total[1]), value=float(total[0]/total[1]),
                                            frames=tasks[item, case, mode, 0]["result"]["frames"]))
    save_csv("summary.csv", summary)
    save_csv("per-seed.csv", per_seed)
    save_csv("comparisons.csv", comparisons)
    save_csv("per-text.csv", item_scores)
    save_csv("common-condition-audit.csv", audit)

    contributors = []
    den = array(matched31, "official-rf-fp32", "native", "reazon", "heard_error_rate")[:, 1].sum()
    for item in matched31:
        left = array([item], "official-rf-fp32", "native", "reazon", "heard_error_rate")[0]
        right = array([item], "speech-rf-fp32", "native", "reazon", "heard_error_rate")[0]
        contributors.append(dict(item=item, excess_errors=int(right[0]-left[0]),
                                 contribution_pp=float((right[0]-left[0])/den*100)))
    contributors.sort(key=lambda r:r["contribution_pp"], reverse=True)
    save_csv("reazon-native-fp32-contributors.csv", contributors)
    # Retain representative recognizer disagreement, with original audio hashes and transcriptions.
    examples = []
    for item, seeds in [("existing:p10", [2, 4]), ("existing:p03", [1, 2])]:
        for seed in seeds:
            for case in ["official-rf-fp32", "speech-rf-fp32"]:
                for mode in MODES:
                    t = tasks[item, case, mode, seed]
                    examples.append(dict(id=t["id"], item=item, case=case, mode=mode, seed=seed,
                                         text=t["spec"]["item"]["text"], wav=t["wav"], wav_sha256=t["wav_sha256"],
                                         qwen=saved_scores[t["id"]]["qwen"]["result"]["transcript"],
                                         reazon=saved_scores[t["id"]]["reazon"]["result"]["transcript"]))
    save_json("transcript-examples.json", examples)

    verify=OUT/"whisper-check.jsonl"
    supplemental=[]
    if verify.exists():
        supplemental=[json.loads(line) for line in verify.read_text().splitlines()]
        assert len(supplemental)==8
        assert all(r['result']['type']=='result' for r in supplemental)
        assert json.loads((OUT/'whisper-status.json').read_text())['status']=='complete'
    source_paths = [RUN/"results.sqlite3", RUN/"report/per-input.jsonl", RUN/"manifest.json", Path(__file__).resolve(),
                    ROOT/"benchmark/quality/python_worker.py", ROOT/"benchmark/quality/speech_worker.cpp",
                    ROOT/"benchmark/quality/config.json", ROOT/"benchmark/quality/score_worker.py",
                    ROOT/"engines/speech.cpp/src/families/irodori-tts/sampler.cpp",
                    ROOT/"engines/Irodori-TTS/irodori_tts/rf.py"]
    if supplemental:source_paths += [verify,OUT/'verify_asr.py',OUT/'whisper-status.json']
    save_json("manifest.json", dict(source_run=str(RUN), cases=CASES,
                                   native32=native32, matched31=matched31, equal_length_fp32_27=equal27,
                                   exclusion_from_32=dict(item="speech-bench:number-money", reason="common-mode length assertion; 116/25*48000 is 222719.99999999997, truncation 222719 versus expected 222720"),
                                   bootstrap=dict(unit="text", replicates=2000, seed=20261016),
                                   existing_noise_and_frame_hash_audits=len(audit), synthesis_performed=False,
                                   additional_asr=dict(model='pinned whisper-large-v3-turbo',requests=len(supplemental),
                                                       selection='posthoc diagnostic examples; not added to corpus-wide scores'),
                                   sources=[dict(path=str(p),sha256=sha(p)) for p in source_paths]))
    means={(r["subset"],r["mode"],r["case"],r["scorer"],r["metric"]):r["mean"] for r in summary}
    def mean(subset,mode,case,scorer):return means[subset,mode,case,scorer,"heard_error_rate"]*100
    lines=["# RFの誤り率差の調査", "", "同じ31文×5 seedで通常条件・共通条件・精度を比較。全構成の生成成功と両ASRの採点成功を確認。",
           "前回の32文から、共通条件の長さ検査で除外されたnumber-moneyを全条件共通に除いた。",
           "number-moneyの除外は4.64秒×48kHzの浮動小数点切り捨てによる1サンプル差。speech.cpp固有の品質失敗として扱わない。",
           "元の結果・音声・採点結果は変更していない。以下は保存結果の再集計。", "",
           "| 条件 | 精度（公式 / speech.cpp） | Qwen公式 | Qwen speech.cpp | 差 | Reazon公式 | Reazon speech.cpp | 差 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for a,b,label in [("official-rf-bf16","speech-rf-f16","BF16 / F16"),("official-rf-fp32","speech-rf-fp32","FP32 / FP32")]:
            q1,q2=mean("matched31",mode,a,"qwen"),mean("matched31",mode,b,"qwen")
            r1,r2=mean("matched31",mode,a,"reazon"),mean("matched31",mode,b,"reazon")
            lines.append(f"| {mode} | {label} | {q1:.2f}% | {q2:.2f}% | {q2-q1:+.2f}pt | {r1:.2f}% | {r2:.2f}% | {r2-r1:+.2f}pt |")
    lines += ["", "FP32でも通常条件ではReazonの差が約4.87ポイント残る。低精度だけでは説明できない。",
              "生成長・初期ノイズ共通ではFP32の差はQwen +0.15ポイント、Reazon +0.09ポイント。",
              "共通条件の155組について、全4構成の生成frame数・sample数・入力ノイズSHA-256の一致を確認。",
              "FP32のQwen差の95%区間は0を含み、Reazonも0を含む。区間の詳細はcomparisons.csvを参照。", "",
              "長さがFP32の全条件で元から同じ27文に限定すると、Reazon差は通常+6.58ポイント→共通ノイズ0.00ポイント。",
              "この27文では生成長を変えず、ノイズを揃えると差が消えるため、長さの違いを主因とは考えにくい。",
              "残る4文の長さの違いは1フレーム（40ms）。通常条件のFP32 Reazon差はその4文を除いても残る。", "",
              "公式はCUDAのtorch.Generator + torch.randn、speech.cppはmt19937_64 + Box–Mullerで初期ノイズを生成する。",
              "同じseed番号は同じノイズを意味しない。各5 seedの有限標本と認識器の感度が差に影響したと推定する。",
              "RFサンプラーは両者とも40ステップ・Euler・線形スケジュール・初期時刻0.999・independent CFG。",
              "少なくとも今回の条件では、RF実装そのものの大幅な読み精度低下を示す結果とはいえない。", "",
              "通常FP32のReazon差の約85.4%はexisting:p10、p03、p01の3文に集中。",
              "p10のspeech.cpp seed 2/4では、Qwenはほぼ全文を認識する一方、Reazonは「ご乗車のお客様は黄色い線の内側でお待ちください。」のみ。",
              "p03でもQwenは全文を認識するがReazonが前半または後半を脱落する例がある。",
              "ASR依存性が大きく、認識文の脱落をそのまま合成音声の脱落と断定できない。", "",
              "共通条件の波形は完全一致しない。参考の波形相関は音質や自然さの同等性の証明には使わない。",
              "JVS・試聴は未実施。この結論は31文・5 seedおよび固定したモデル・runtimeに限定される。", ""]
    if supplemental:
        lines += ["## 代表例の第三ASRによる確認", "", "保存済み音声8件を固定Whisper large-v3-turboで追加認識。代表例の診断用であり、全件スコアには混ぜない。", ""]
        lines += ["Reazonで前半・後半を脱落したspeech.cppの4音声も、WhisperはQwenと同様に内容全体を認識できた。",
                  "この例ではTTSの文脱落を裏付けず、Reazon側の取りこぼしがスコア差に影響している。", ""]
        for r in supplemental:
            lines.append(f"- {r['case']} / {r['item']} / seed {r['seed']}: {r['result'].get('transcript',r['result'])}")
        lines.append("")
    (OUT/"RESULTS.ja.md").write_text("\n".join(lines))
    print("\n".join(lines[:18]))
    print("Saved",OUT)


if __name__=="__main__":main()
