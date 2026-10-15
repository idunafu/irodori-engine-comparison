"""Isolated CUDA runtime A/B: alternating screen, then the original full matrix."""
import argparse
import collections
import csv
import fcntl
import json
from pathlib import Path
import statistics as stats
import threading
import time

import numpy as np
import soundfile as sf
import run as harness
from report import components
from runtime_ab_setup import snapshot
from speech_ab import paired_interval

ROOT = harness.ROOT
OUT = ROOT / 'artifacts/runtime-ab-20261015'
DEST = ROOT / 'benchmark/results/runtime-ab-20261015'
RECORDS = OUT / 'results.jsonl'
ORIGINAL_COMMAND = harness.command
CASES = [c for c in harness.matrix() if c['engine'] in
         ('Irodori-TTS', 'Irodori-TTS-Optimiz', 'irodori-onnx')]


def case_for(base, version):
    return dict(base, base_case=base['case'], version=version,
                case=f'runtime-ab-{version}-{base["case"]}')


def command(case):
    cmd, env = ORIGINAL_COMMAND(case)
    engine = case['engine']
    env.update(PYTHONDONTWRITEBYTECODE='1')
    if engine != 'irodori-onnx':
        cmd[1] = str(OUT / 'python_worker.py')
        if case['version'] == 'new':
            cmd[0] = str(ROOT / f'.venvs/runtime-ab-20261015/{engine}/bin/python')
    elif case['version'] == 'new':
        site = ROOT / '.venvs/runtime-ab-20261015/onnx/lib/python3.11/site-packages'
        libs = sorted((site / 'nvidia').glob('*/lib'))
        # CUDA toolkit wheels also use these namespace directories in CUDA 13.
        libs += sorted((site / 'nvidia').glob('*/lib64'))
        libs += [site / 'onnxruntime/capi']
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, libs))
        env['IRODORI_ORT_DLL'] = str(next((site / 'onnxruntime/capi').glob('libonnxruntime.so.*')))
    return cmd, env


def append(record):
    harness.append(RECORDS, record)


def records():
    latest = {}
    if RECORDS.exists():
        for line in RECORDS.read_text().splitlines():
            r = json.loads(line)
            latest[r['key']] = r
    return list(latest.values())


def check(result, seconds, profile, case):
    assert result['sample_rate'] == 48000
    if seconds:
        assert result['frames'] == round(seconds * 25)
    if profile:
        stages = {e['stage'] for e in result['events']}
        assert {'sampling', 'sampling_loop', 'dit_forward'} <= stages, stages
        assert sum(e['stage'] == 'dit_forward' for e in result['events']) >= harness.CFG['steps'][case['variant']]


def preflight(monitor, done):
    for base in CASES:
        arrays = {}
        for version in ('old', 'new'):
            case = case_for(base, version)
            key = case['case'] + '/preflight'
            suffix = '.raw' if case['engine'] == 'irodori-onnx' else '.wav'
            paths = [OUT / 'raw' / f'{case["case"]}-{p}{suffix}' for p in (False, True)]
            if key not in done:
                with harness.Worker(case, monitor) as worker:
                    outputs = []
                    for profile, path in zip((False, True), paths):
                        r = worker.request(dict(id=key, text=harness.PROMPTS[0]['text'], seconds=2.,
                                                cache='reencode', profile=profile, output=str(path)))
                        check(r, 2., profile, case)
                        outputs.append(r)
                a = [np.fromfile(p, dtype='<f4') if suffix == '.raw' else sf.read(p, dtype='float32')[0] for p in paths]
                relative = float(np.linalg.norm(a[0] - a[1]) / max(np.linalg.norm(a[0]), 1e-12))
                assert relative <= (1e-4 if case['precision'] == 'fp32' else .01), relative
                append(dict(key=key, kind='preflight', status='ok', case=case, outputs=outputs,
                            load=worker.ready, profile_relative_error=relative))
            arrays[version] = np.fromfile(paths[0], dtype='<f4') if suffix == '.raw' else sf.read(paths[0], dtype='float32')[0]
        a, b = arrays['old'], arrays['new']
        append(dict(key=base['case'] + '/parity', kind='parity', status='ok', case=base,
                    relative_error=float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-12)),
                    max_abs=float(np.max(np.abs(a-b))), bit_equal=bool(np.array_equal(a, b))))
        print('PREFLIGHT', base['case'], flush=True)


def groups(phase):
    if phase == 'screen':
        work = [('screen', p, float(s)) for p in
                (harness.PROMPTS[0], max(harness.PROMPTS, key=lambda p: p['tokens'])) for s in (2, 8)]
    else:
        durations = json.loads((ROOT / 'artifacts/benchmark/inputs/durations.json').read_text())
        work = [(mode, p, None if mode == 'auto' else 8. if mode == 'text_scaling' else durations[p['id']] / 25)
                for mode in ('matched', 'auto', 'text_scaling') for p in harness.PROMPTS]
        work += [('audio_scaling', harness.PROMPTS[8], float(s)) for s in harness.CFG['audio_seconds']]
    return [(cache, mode, p, seconds) for cache in harness.CFG['cache_profiles'] for mode, p, seconds in work]


def measure(phase, bases, monitor, done):
    for index, base in enumerate(bases):
        for round_id in range(3 if phase == 'screen' else 1):
            order = ('old', 'new') if (round_id + index) % 2 == 0 else ('new', 'old')
            for version in order:
                case = case_for(base, version)
                prefix = f'{phase}/{case["case"]}/round-{round_id}'
                def group_key(cache, mode, p, seconds):
                    return f'{prefix}/{cache}/{mode}/{p["id"]}/{seconds}'
                def complete(*g):
                    key = group_key(*g)
                    return all(f'{key}/{kind}/{i}' in done for kind in ('wall', 'profile') for i in range(5))
                if all(complete(*g) for g in groups(phase)):
                    continue
                print('WORKER', prefix, flush=True)
                with harness.Worker(case, monitor) as worker:
                    cold = worker.request(dict(id=prefix+'/cold', text=harness.PROMPTS[8]['text'],
                                          seconds=8., cache='reencode', profile=False))
                    append(dict(key=prefix+'/cold', kind='cold', phase=phase, round=round_id,
                                status='ok', case=case, load=worker.ready, result=cold))
                    for cache, mode, p, seconds in groups(phase):
                        group = group_key(cache, mode, p, seconds)
                        if complete(cache, mode, p, seconds):
                            continue
                        req = dict(id=group, text=p['text'], seconds=seconds, cache=cache, profile=False)
                        common = dict(status='ok', phase=phase, round=round_id, case=case,
                                      cache=cache, mode=mode, prompt=p, seconds=seconds)
                        for _ in range(2):
                            worker.request(req)
                        time.sleep(.2)
                        append(dict(common, key=group+'/idle', kind='idle', result=monitor.snapshot()))
                        for profile in (False, True):
                            kind = 'profile' if profile else 'wall'
                            for i in range(5):
                                key = f'{group}/{kind}/{i}'
                                if key in done:
                                    continue
                                result = worker.request(dict(req, id=key, profile=profile))
                                check(result, seconds, profile, case)
                                append(dict(common, key=key, kind=kind, result=result))
                                done.add(key)
                        print('DONE', group, flush=True)


def summarize(data):
    grouped = collections.defaultdict(list)
    for r in data:
        if r['kind'] in ('wall', 'profile', 'idle'):
            grouped[(r['phase'], r['case']['base_case'], r['cache'], r['mode'], r['prompt']['id'], r['seconds'])].append(r)
    rows = []
    for (phase, base, cache, mode, prompt, seconds), xs in grouped.items():
        row = dict(phase=phase, case=base, cache=cache, mode=mode, prompt=prompt, seconds=seconds,
                   tokens=xs[0]['prompt']['tokens'])
        blocks = {}
        nrounds = 3 if phase == 'screen' else 1
        for version in ('old', 'new'):
            wall = [r for r in xs if r['case']['version'] == version and r['kind'] == 'wall']
            profiles = [r for r in xs if r['case']['version'] == version and r['kind'] == 'profile']
            idle = [r for r in xs if r['case']['version'] == version and r['kind'] == 'idle']
            blocks[version] = {i: [r['result']['ms'] for r in wall if r['round'] == i] for i in range(nrounds)}
            row[version+'_n'] = len(wall)
            row[version+'_profile_n'] = len(profiles)
            if not wall:
                continue
            row[version+'_ms'] = stats.median(stats.median(v) for v in blocks[version].values() if v)
            row[version+'_min_ms'] = min(r['result']['ms'] for r in wall)
            row[version+'_max_ms'] = max(r['result']['ms'] for r in wall)
            row[version+'_stdev_ms'] = stats.pstdev(r['result']['ms'] for r in wall)
            row[version+'_audio_seconds'] = stats.median(r['result']['samples']/48000 for r in wall)
            row[version+'_rtf'] = stats.median(r['result']['rtf'] for r in wall)
            row[version+'_peak_mib'] = max(r['result']['peak_vram'] for r in wall)/2**20
            row[version+'_resident_mib'] = stats.median(r['result']['resident_vram'] for r in idle)/2**20
            for metric in ('peak_allocated', 'peak_reserved'):
                vals = [r['result'][metric]/2**20 for r in wall if metric in r['result']]
                row[version+'_'+metric+'_mib'] = max(vals) if vals else None
            cs = [components(r['result']) for r in profiles]
            for metric in ('sampling_ms', 'sampling_loop_ms', 'decode_ms', 'forward_b1_ms',
                           'forward_b4_ms', 'forward_calls', 'forward_first_ms', 'reference_encode_ms',
                           'text_encode_ms', 'caption_encode_ms', 'text_tokenize_ms', 'caption_tokenize_ms',
                           'reference_preprocess_ms', 'reference_codec_ms', 'speaker_encode_ms',
                           'shared_encoder_ms', 'duration_ms', 'segments'):
                vals = [c[metric] for c in cs if c[metric] is not None]
                row[version+'_'+metric] = stats.median(vals) if vals else None
        row['complete'] = all(len(blocks[v][i]) == 5 for v in blocks for i in range(nrounds)) and all(row[v+'_profile_n'] == nrounds*5 for v in blocks)
        if row['complete']:
            row['delta_pct'] = (row['new_ms']/row['old_ms']-1)*100
            if phase == 'screen':
                lo, hi = paired_interval(blocks['old'], blocks['new'])
                deltas = [stats.median(blocks['new'][i])/stats.median(blocks['old'][i])-1 for i in range(3)]
                row.update(ci_low_pct=(lo-1)*100, ci_high_pct=(hi-1)*100,
                           round_deltas_pct=json.dumps([d*100 for d in deltas]),
                           speed_difference=bool((hi < 1 or lo > 1) and (all(d>0 for d in deltas) or all(d<0 for d in deltas))))
                row['practical_speed_difference'] = row['speed_difference'] and abs(row['delta_pct']) >= 2
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def report(data, decision, final=False):
    DEST.mkdir(parents=True, exist_ok=True)
    rows = summarize(data)
    write_csv(DEST / 'per-input.csv', rows)
    parity = [dict(case=r['case']['case'], relative_error=r['relative_error'],
                   max_abs=r['max_abs'], bit_equal=r['bit_equal']) for r in data if r['kind']=='parity']
    write_csv(DEST / 'pcm-parity.csv', parity)
    cold_rows = []
    for base in CASES:
        for version in ('old', 'new'):
            xs = [r for r in data if r['kind']=='cold' and r['phase']=='screen'
                  and r['case']['base_case']==base['case'] and r['case']['version']==version]
            if not xs:
                continue
            cold_rows.append(dict(case=base['case'], version=version, repeats=len(xs),
                load_ms=stats.median(r['load']['load_ms'] for r in xs),
                startup_ms=stats.median(r['load']['startup_wall_ms'] for r in xs),
                first_inference_ms=stats.median(r['result']['ms'] for r in xs),
                loaded_resident_mib=stats.median(r['load']['resident_vram']/2**20 for r in xs),
                load_peak_mib=max(r['load']['peak_vram']/2**20 for r in xs)))
    write_csv(DEST / 'cold.csv', cold_rows)
    preserved = snapshot() == json.loads((OUT / 'preservation.json').read_text())
    validation = dict(preserved=preserved, preflight_count=sum(r['kind']=='preflight' for r in data),
                      incomplete=sum(not r['complete'] for r in rows),
                      wall_requests=sum(r['kind']=='wall' for r in data),
                      profile_requests=sum(r['kind']=='profile' for r in data),
                      measured_conditions=len(rows), affected=decision['affected'])
    validation['expected_conditions'] = 64 + 130*len(decision['affected'])
    validation['missing_conditions'] = max(0, validation['expected_conditions']-len(rows))
    validation['measurement_complete'] = not validation['incomplete'] and not validation['missing_conditions']
    (DEST / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    text = ['# PyTorch / ONNX Runtime 更新 A/B（2026-10-15）', '',
            '状態: '+('全測定完了。' if validation['measurement_complete'] else '本ベンチマーク測定中。以下は完了した条件の中間集計。'), '',
            '既存venv・固定エンジン・モデル・既存結果を保持し、専用venvで比較。負の差は新版が高速。', '',
            '旧: Python 3.10.20、PyTorch 2.10.0+cu128、ONNX Runtime 1.23.2。',
            '新: PyTorch 2.14.1+cu130、TorchAudio 2.11.0、TorchCodec 0.17.0、ONNX Runtime 1.31.0、ONNX 1.23.2、ONNXScript 0.7.2。',
            'ONNX Runtimeの最新wheelはPython>=3.11のためONNX環境は3.11。PyTorch版は3.10を維持。', '',
            'ONNXはRustから共有ライブラリを直接使用するため、Pythonの実行時間は測定に含まれない。', '',
            '同じエンジンコミット・重み・ONNXグラフ・参照PCMを使用。再exportせずruntimeの更新を比較する。ONNX/ONNXScriptは導入バージョンを記録し、測定する推論自体には使用しない。', '',
            'batch 1、seed 42、RF 40 / MF 4 steps、CPU 8 threads、TF32 off、compile off。caption/CFG/codec精度/キャッシュは主比較と同じ。',
            '実用精度は公式／OptimizがBF16モデル＋FP32 codec、ONNXがFP16モデル・decoder＋FP32参照encoder。FP32条件はモデル・codecともFP32。',
            '参照音声は共通のTYC054.wav（48kHz・PCM24）。captionは「自然な日本語で、落ち着いた口調で話す。」。RFはindependent CFG（text/caption 3、speaker 5、適用区間0.5〜1.0）、linear schedule。',
            '小規模比較: p01/p19、2秒/8秒、両キャッシュ、FP32/実用精度。3ラウンド、旧新の順序を交互にし、各条件warmup 2回・通常5回・詳細5回。',
            '拡張判定: 対応ラウンドを保った階層bootstrap 95%区間が0を跨がず、全3ラウンドが同方向。2%以上は実用差として別記。条件別検定の多重性補正なし。',
            '拡張: 該当するエンジン・モデル・精度の新旧両方で主比較と同じ20文章、matched/auto/text_scaling/audio_scaling、両キャッシュ、warmup 2回・通常5回・詳細5回。拡張は1ラウンドの記述統計。', '',
            '## 小規模A/B', '', '| 条件 | cache | 入力 | 秒 | 旧ms | 新ms | 差% | 95%区間% | 有意差 |',
            '|---|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['phase']=='screen' and r['complete']:
            text.append(f'| {r["case"]} | {r["cache"]} | {r["prompt"]} | {r["seconds"]:g} | {r["old_ms"]:.2f} | {r["new_ms"]:.2f} | {r["delta_pct"]:+.2f} | {r["ci_low_pct"]:+.2f}〜{r["ci_high_pct"]:+.2f} | {r["speed_difference"]} |')
    text += ['', '## 本ベンチマーク', '', '拡張対象: '+', '.join(decision['affected']), '',
             '| 条件 | mode | cache | 入力条件数 | 旧ms中央値 | 新ms中央値 | 入力別差%中央値 |',
             '|---|---|---|---:|---:|---:|---:|']
    expanded = collections.defaultdict(list)
    for r in rows:
        if r['phase']=='expanded' and r['complete']:
            expanded[(r['case'],r['mode'],r['cache'])].append(r)
    for (case, mode, cache), xs in sorted(expanded.items()):
        text.append(f'| {case} | {mode} | {cache} | {len(xs)} | {stats.median(r["old_ms"] for r in xs):.2f} | {stats.median(r["new_ms"] for r in xs):.2f} | {stats.median(r["delta_pct"] for r in xs):+.2f} |')
    text += ['', '## 段階別時間・VRAM（matched）', '',
             '旧／新の順。各文章の詳細計測中央値を20文章で中央値に集計する。VRAM常駐量も中央値、peakは20文章の最大値。段階別時間はGPU同期する別実行であり、通常時間への単純な加算には使わない。', '',
             '| 条件 | cache | sampling ms | DACVAE decode ms | 常駐 MiB | peak MiB |',
             '|---|---|---:|---:|---:|---:|']
    for (case, mode, cache), xs in sorted(expanded.items()):
        if mode != 'matched':
            continue
        pairs = []
        for metric in ('sampling_ms', 'decode_ms', 'resident_mib', 'peak_mib'):
            vals = []
            for version in ('old', 'new'):
                numbers = [r[version+'_'+metric] for r in xs if r.get(version+'_'+metric) is not None]
                value = (max(numbers) if metric=='peak_mib' else stats.median(numbers)) if numbers else None
                vals.append(f'{value:.2f}' if value is not None else '—')
            pairs.append('／'.join(vals))
        text.append(f'| {case} | {cache} | '+' | '.join(pairs)+' |')
    text += ['', '## ロード・初回推論', '',
             '小規模比較の各ラウンドで新しいworkerを起動し、p09を8秒生成。各版3回の中央値。OS・ドライバのキャッシュは保持。', '',
             '| 条件 | 版 | 回数 | load ms | 起動からready ms | 初回推論 ms | ロード後常駐 MiB | ロードpeak MiB |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in cold_rows:
        text.append(f'| {r["case"]} | {r["version"]} | {r["repeats"]} | {r["load_ms"]:.2f} | {r["startup_ms"]:.2f} | {r["first_inference_ms"]:.2f} | {r["loaded_resident_mib"]:.1f} | {r["load_peak_mib"]:.1f} |')
    text += ['', '## 新旧PCMの数値差', '',
             'p01・2秒・reencode、同じseedの通常推論出力を比較。相対L2差は音質スコアではない。計測有無によるPCM差は全16構成で別途検証済み。', '',
             '| 条件 | 相対L2差 | 最大絶対差 | bit一致 |', '|---|---:|---:|---|']
    for p in parity:
        text.append(f'| {p["case"]} | {p["relative_error"]:.6g} | {p["max_abs"]:.6g} | {p["bit_equal"]} |')
    text += ['', '全入力の速度・RTF・段階別時間・VRAM: [CSV](per-input.csv)。', '',
             'native_cacheでの[生成長スケーリング](audio_scaling.png)／[テキスト長スケーリング](text_scaling.png)。', '',
             f'検証: {json.dumps(validation, ensure_ascii=False)}', '',
             '[最終独立検証](independent-validation.json)／[導入バージョン](../../../environments/runtime-ab-20261015/versions.json)。独立検証ファイルは全測定終了後に生成する。', '',
             'PCMの新旧差は数値差として生データに保存。音声品質の採点は行っていない。',
             '生データ・全依存バージョン・起動コマンド・モデルハッシュ: `artifacts/runtime-ab-20261015/`。', '',
             '公式情報: [PyTorch wheels](https://download.pytorch.org/whl/cu130/torch/)、[TorchAudio互換性](https://docs.pytorch.org/audio/main/installation.html)、[ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。']
    overview = ['', '## 主比較の速度差', '',
        '同じ生成フレーム数に揃えた全20文章（matched）。各文章で `(新版ms / 旧版ms - 1) × 100` を計算し、その中央値を掲載。負の値は新版が高速。', '',
        '| 実装 / モデル / 重み精度 | 再エンコード 差% | キャッシュあり 差% |',
        '|---|---:|---:|']
    for base in CASES:
        groups_for_case = [expanded.get((base['case'], 'matched', cache), [])
                           for cache in harness.CFG['cache_profiles']]
        if not all(len(xs)==20 for xs in groups_for_case):
            continue
        deltas = [stats.median(r['delta_pct'] for r in xs) for xs in groups_for_case]
        label = f'{base["engine"]} / {base["variant"].upper()} / {base["precision"].upper()}'
        overview.append(f'| {label} | {deltas[0]:+.2f} | {deltas[1]:+.2f} |')
    overview += ['', '[生成長スケーリング](audio_scaling.png)／[テキスト長スケーリング](text_scaling.png)／[全入力CSV](per-input.csv)。', '']
    text[4:4] = overview
    (DEST / 'RESULTS.ja.md').write_text('\n'.join(text)+'\n')
    if final:
        assert preserved and validation['preflight_count']==16 and not validation['incomplete'], validation
        assert sum(r['phase']=='expanded' for r in rows) == len(decision['affected'])*130
        assert len(cold_rows)==16 and all(r['repeats']==3 for r in cold_rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    for name in ('raw', 'python-cache'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    assert snapshot() == json.loads((OUT / 'preservation.json').read_text()), 'Original environment changed'
    worker = (ROOT / 'benchmark/python_worker.py').read_text()
    worker = worker.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})')
    worker = worker.replace("ROOT/'artifacts/benchmark/python-cache'", f'Path({str(OUT / "python-cache")!r})')
    (OUT / 'python_worker.py').write_text(worker)
    harness.command = command
    commands = {case_for(c,v)['case']: dict(command=command(case_for(c,v))[0], environment={k:x for k,x in command(case_for(c,v))[1].items() if k in ('LD_LIBRARY_PATH','IRODORI_ORT_DLL','OMP_NUM_THREADS','NVIDIA_TF32_OVERRIDE')}) for c in CASES for v in ('old','new')}
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    decision_path = OUT / 'decision.json'
    if args.report_only:
        report(records(), json.loads(decision_path.read_text()))
        return
    done = {r['key'] for r in records() if r['status']=='ok'}
    with (ROOT / 'artifacts/gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        monitor = harness.Monitor()
        # Reuse the complete process/temperature/clock observer with a dedicated output.
        import speech_ab
        speech_ab.OUT = OUT
        stop = threading.Event()
        observer = threading.Thread(target=speech_ab.telemetry, args=(stop,), daemon=True)
        observer.start()
        try:
            preflight(monitor, done)
            if args.preflight_only:
                return
            measure('screen', CASES, monitor, done)
            rows = summarize(records())
            decision = dict(affected=sorted({r['case'] for r in rows if r['phase']=='screen' and r.get('speed_difference')}),
                            triggers=[r for r in rows if r['phase']=='screen' and r.get('speed_difference')])
            decision_path.write_text(json.dumps(decision, indent=2)+'\n')
            report(records(), decision)
            print('EXPANSION', decision['affected'], flush=True)
            measure('expanded', [c for c in CASES if c['case'] in decision['affected']], monitor, done)
        finally:
            stop.set()
            observer.join()
            monitor.close()
    report(records(), decision, final=True)
    print('RUNTIME A/B COMPLETE', DEST, flush=True)


if __name__ == '__main__':
    main()
