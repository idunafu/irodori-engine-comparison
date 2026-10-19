"""Recompute matched successful-text metrics using saved outputs only.

Run with .venvs/quality/bin/python PATH/TO/recompute.py. The campaign's
database, original reports, audio, and recognition results are read-only.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
CASES = ["official-mf-bf16", "onnx-mf-fp16", "speech-mf-f16",
         "official-rf-bf16", "speech-rf-f16"]
SEEDS = list(range(5))
REPLICATES = 2000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n")


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(left, right):
    """Paired bootstrap by text, retaining all seeds and edit/length weights."""
    keys = sorted(left)
    assert set(keys) == set(right)
    a = np.array([left[key] for key in keys], dtype=float)
    b = np.array([right[key] for key in keys], dtype=float)
    rng = np.random.default_rng(20261016)
    indices = rng.integers(0, len(keys), size=(REPLICATES, len(keys)))
    x, y = a[indices].sum(axis=1), b[indices].sum(axis=1)
    values = x[:, 0] / x[:, 1] - y[:, 0] / y[:, 1]
    return dict(difference=float(a[:, 0].sum() / a[:, 1].sum()
                                 - b[:, 0].sum() / b[:, 1].sum()),
                low=float(np.quantile(values, .025)),
                high=float(np.quantile(values, .975)),
                clusters=len(keys))


def main():
    db_path = RUN / "results.sqlite3"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    groups, scores, task_results = defaultdict(dict), {}, {}
    for row in db.execute('''SELECT * FROM tasks WHERE dataset='practical'
                             AND profile='practical' AND mode='native' '''):
        spec = json.loads(row["spec"])
        key = (row["case_id"], spec["seed"])
        assert key not in groups[spec["item"]["id"]]
        groups[spec["item"]["id"]][key] = (dict(row), spec)
        task_results[row["id"]] = json.loads(row["result"])
        scores[row["id"]] = {
            r["scorer"]: (r["status"], json.loads(r["result"]))
            for r in db.execute("SELECT * FROM scores WHERE task_id=?", (row["id"],))
        }
    db.close()
    expected = {(case, seed) for case in CASES for seed in SEEDS}
    assert len(groups) == 40
    assert all(set(group) == expected for group in groups.values())
    included = sorted(item for item, group in groups.items()
                      if all(row["status"] == "ok" for row, _ in group.values()))
    assert len(included) == 32
    excluded = []
    for item, group in sorted(groups.items()):
        if item in included:
            continue
        failed = [(row, spec) for row, spec in group.values() if row["status"] != "ok"]
        assert all(task_results[row["id"]].get("category") == "duration_limit"
                   for row, _ in failed)
        excluded.append(dict(item=item, text=failed[0][1]["item"]["text"],
                             failed_cases=sorted({row["case_id"] for row, _ in failed}),
                             reason="duration_limit"))

    # Cache the original derived metrics, with no new ASR or acoustic inference.
    details = {}
    with (RUN / "report/per-input.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if (row["dataset"] == "practical" and row["profile"] == "practical"
                    and row["mode"] == "native"):
                details[row["id"]] = row
    assert len(details) == 1000
    summary, per_seed, clusters = [], [], defaultdict(dict)
    for case in CASES:
        for scorer in ["qwen", "reazon"]:
            for seed in SEEDS:
                totals = defaultdict(float)
                for item in included:
                    row, spec = groups[item][case, seed]
                    assert scores[row["id"]][scorer][0] == "ok"
                    heard = details[row["id"]]["metrics"][scorer]
                    length = heard["heard_reference_length"]
                    assert length > 0
                    pairs = {
                        "heard_error_rate": (heard["heard_errors"], length),
                        "raw_cer": (heard["heard_raw_errors"], length),
                        "heard_as_said": (int(heard["heard_as_said"]), 1),
                        "breakdown_rate": (int(heard["heard_errors"] / length > .3), 1),
                    }
                    # Reference normalization must not depend on the implementation.
                    other_id = groups[item][CASES[0], seed][0]["id"]
                    assert details[other_id]["metrics"][scorer]["heard_reference_length"] == length
                    for name, pair in pairs.items():
                        metric = scorer + "_" + name
                        totals[metric + ":n"] += pair[0]
                        totals[metric + ":d"] += pair[1]
                        by_item = clusters[case, metric]
                        by_item.setdefault(item, np.zeros(2))
                        by_item[item] += pair
                for name in pairs:
                    metric = scorer + "_" + name
                    numerator, denominator = totals[metric + ":n"], totals[metric + ":d"]
                    per_seed.append(dict(case=case, metric=metric, seed=seed,
                                         numerator=numerator, denominator=denominator,
                                         value=numerator / denominator, texts=len(included)))
        for metric in sorted({r["metric"] for r in per_seed if r["case"] == case}):
            values = [r["value"] for r in per_seed if r["case"] == case and r["metric"] == metric]
            assert len(values) == 5
            summary.append(dict(case=case, metric=metric, mean=float(np.mean(values)),
                                population_std=float(np.std(values)), texts=32, takes=160))

    # Use identical eligible text sets across all five cases and all five seeds.
    voice_items = [item for item in included if all(
        scores[row["id"]]["eres2net"][1].get("embedding") is not None
        for row, _ in groups[item].values())]
    pitch_items = [item for item in included if all(
        details[row["id"]]["metrics"].get("median_pitch_hz", 0) is not None
        and details[row["id"]]["metrics"].get("median_pitch_hz", 0) > 0
        for row, _ in groups[item].values())]
    assert len(voice_items) > 1 and len(pitch_items) > 1
    for case in CASES:
        for seed in SEEDS:
            vectors = np.array([scores[groups[item][case, seed][0]["id"]]["eres2net"][1]["embedding"]
                                for item in voice_items])
            sim = vectors @ vectors.T
            cosine = float(sim[np.triu_indices(len(vectors), 1)].mean())
            pitch = [details[groups[item][case, seed][0]["id"]]["metrics"]["median_pitch_hz"]
                     for item in pitch_items]
            spread = float(np.std(12 * np.log2(pitch)))
            for metric, value, texts in [("same_voice_cosine", cosine, len(voice_items)),
                                         ("pitch_spread_semitones", spread, len(pitch_items))]:
                per_seed.append(dict(case=case, metric=metric, seed=seed,
                                     numerator=value, denominator=1, value=value, texts=texts))
        for metric, items in [("same_voice_cosine", voice_items), ("pitch_spread_semitones", pitch_items)]:
            values = [r["value"] for r in per_seed if r["case"] == case and r["metric"] == metric]
            summary.append(dict(case=case, metric=metric, mean=float(np.mean(values)),
                                population_std=float(np.std(values)), texts=len(items), takes=5 * len(items)))

    # Confirm that this aggregation reproduces every original full-40-text ASR value.
    for old in csv.DictReader((RUN / "report/summary.csv").open()):
        if (old["dataset"] != "practical" or old["case"] not in CASES
                or old["mode"] != "native" or old["kind"] != "all"
                or old["success_only"] != "False"):
            continue
        scorer, _, name = old["metric"].partition("_")
        if scorer not in {"qwen", "reazon"} or name not in {"heard_error_rate", "raw_cer", "heard_as_said", "breakdown_rate"}:
            continue
        values = []
        for seed in SEEDS:
            pairs = []
            for group in groups.values():
                heard = details[group[old["case"], seed][0]["id"]]["metrics"][scorer]
                length = heard["heard_reference_length"]
                pairs.append({"heard_error_rate": (heard["heard_errors"], length),
                              "raw_cer": (heard["heard_raw_errors"], length),
                              "heard_as_said": (int(heard["heard_as_said"]), 1),
                              "breakdown_rate": (int(heard["heard_errors"] / length > .3), 1)}[name])
            values.append(sum(x[0] for x in pairs) / sum(x[1] for x in pairs))
        assert abs(np.mean(values) - float(old["mean"])) < 1e-12, old

    comparisons = []
    for case in CASES:
        if case.startswith("official"):
            continue
        baseline = "official-" + case.split("-")[1] + "-bf16"
        for (candidate, metric), items in clusters.items():
            if candidate == case:
                comparisons.append(dict(case=case, baseline=baseline, metric=metric,
                                        **interval(items, clusters[baseline, metric])))
    save_csv("summary.csv", summary)
    save_csv("per-seed.csv", per_seed)
    save_csv("comparisons.csv", comparisons)
    save_json("selection.json", dict(included=included, excluded=excluded,
                                     voice_items=voice_items, pitch_items=pitch_items))
    sources = [db_path, RUN / "report/per-input.jsonl", RUN / "report/summary.csv",
               RUN / "manifest.json", Path(__file__).resolve()]
    save_json("manifest.json", dict(
        source_run=str(RUN), cases=CASES, seeds=SEEDS,
        selection="All five native practical-precision cases succeeded for every seed; no selection by ASR quality",
        texts=32, excluded_texts=8, takes_per_case=160,
        bootstrap=dict(unit="text", replicates=REPLICATES, seed=20261016),
        inference_performed=False,
        checks=["matched 32 texts across all cases and seeds", "ASR score status all ok",
                "identical normalized reference lengths", "reproduced original 40-text ASR means"],
        sources=[dict(path=str(path), sha256=sha256(path)) for path in sources]))

    means = {(row["case"], row["metric"]): row["mean"] for row in summary}
    lines = ["# 共通の生成成功文による比較", "",
             "実用40文から、全5構成・全5 seedで生成できた32文を共通に採用。各構成160音声。",
             "MFは公式BF16・ONNX FP16・speech.cpp Vulkan F16、RFは公式BF16・speech.cpp Vulkan F16。",
             "除外はexisting:p13〜p20の8文。ASRの良し悪しによる選別は行っていない。",
             "既存のnative結果のみを使用し、音声生成・ASRの再実行は行っていない。",
             "値は5 seed平均。誤り率は各seedで誤り数合計／参照文字数合計。破綻候補は誤り率30%超。", "",
             "| 構成 | Qwen誤り率 | Reazon誤り率 | Qwen破綻候補率 | Reazon破綻候補率 |",
             "|---|---:|---:|---:|---:|"]
    for case in CASES:
        metrics = ["qwen_heard_error_rate", "reazon_heard_error_rate", "qwen_breakdown_rate", "reazon_breakdown_rate"]
        values = " | ".join(f"{means[case, metric] * 100:.2f}%" for metric in metrics)
        lines.append(f"| {case} | {values} |")
    lines += ["", "公式との差（ポイント）と文章単位の対応ありbootstrap 95%区間：", "",
              "| 構成 | 指標 | 差 | 95%区間 |", "|---|---|---:|---:|"]
    for row in comparisons:
        if row["metric"].endswith(("heard_error_rate", "breakdown_rate")):
            lines.append(f"| {row['case']} | {row['metric']} | {row['difference'] * 100:+.2f} | [{row['low'] * 100:+.2f}, {row['high'] * 100:+.2f}] |")
    lines += ["", f"文間の声の一貫性は全構成・全seedでembeddingを取得できた共通{len(voice_items)}文に限定。",
              "JVSによる話者再現性の評価とは異なり、参照話者への類似度ではない。", "",
              "| 構成 | 文間の声の一貫性（cosine） |", "|---|---:|"]
    for case in CASES:
        lines.append(f"| {case} | {means[case, 'same_voice_cosine']:.6f} |")
    lines += ["", "個別の音声に対する自動判定であり、人が確認した誤りや聴感品質の同等性は意味しない。",
              "この32文の結果は30秒制限で拒否された8文には一般化しない。",
              "元の主比較のASR採点はすべて成功。追加のcommon-noise診断で起きた3件の採点失敗はこの集計に含まれない。", ""]
    (OUT / "RESULTS.ja.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
