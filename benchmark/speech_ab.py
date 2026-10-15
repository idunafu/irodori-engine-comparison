"""Serial old/new speech.cpp A/B, with reproducibility gates and adaptive expansion."""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import threading
import time

import numpy as np
import soundfile as sf
import run as harness
from report import components
from speech_ab_build import digest

ROOT=harness.ROOT
BASELINE='c6f28804907ec47dc4d93b526cb7d3407020b417'
LATEST='6bd7b027674ca6d8aa8329436b56ff4ca3fab4f8'
OUT=ROOT/'artifacts/speech-ab'/LATEST/'measurement'
DEST=ROOT/'benchmark/results'/f'speech-ab-{LATEST[:8]}'
SOURCE=ROOT/'artifacts/speech-ab'/LATEST
RECORDS=OUT/'results.jsonl'
ORIGINAL_COMMAND=harness.command

def command(case):
    cmd,env=ORIGINAL_COMMAND(case)
    if case['version']=='new':cmd[0]=str(SOURCE/f'artifacts/benchmark/build/speech.cpp/{case["backend"]}/comparison_bench')
    return cmd,env

def case_for(version,variant,backend):
    return dict(engine='speech.cpp',variant=variant,backend=backend,precision='f16',precision_profile='practical',
        version=version,commit=BASELINE if version=='old' else LATEST,case=f'speech-ab-{version}-{variant}-{backend}')

def preservation():
    files=[ROOT/'engines.lock.json',ROOT/'models.lock.json',ROOT/'TYC054.wav',
           ROOT/'artifacts/benchmark/results.jsonl',ROOT/'artifacts/benchmark/preflight.jsonl']
    files+=list((ROOT/'benchmark/results').glob('*.*'))
    files+=[ROOT/f'artifacts/benchmark/build/speech.cpp/{b}/comparison_bench' for b in ('cuda','vulkan')]
    return {str(p.relative_to(ROOT)):digest(p) for p in files if p.is_file()}

def telemetry(stop):
    nv=harness.nv;dev=nv.nvmlDeviceGetHandleByIndex(0)
    with (OUT/'telemetry.jsonl').open('a',buffering=1) as f:
        while not stop.is_set():
            processes={}
            for kind,query in (('compute',nv.nvmlDeviceGetComputeRunningProcesses),('graphics',nv.nvmlDeviceGetGraphicsRunningProcesses)):
                for p in query(dev):
                    item=processes.setdefault(p.pid,dict(pid=p.pid,kinds=[],vram=0))
                    item['kinds'].append(kind)
                    if isinstance(p.usedGpuMemory,int) and p.usedGpuMemory<1<<60:item['vram']=max(item['vram'],p.usedGpuMemory)
            f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                temperature=nv.nvmlDeviceGetTemperature(dev,nv.NVML_TEMPERATURE_GPU),
                sm_clock=nv.nvmlDeviceGetClockInfo(dev,nv.NVML_CLOCK_SM),gpu_used=nv.nvmlDeviceGetMemoryInfo(dev).used,
                utilization=nv.nvmlDeviceGetUtilizationRates(dev).gpu,processes=list(processes.values())))+'\n')
            stop.wait(1)

def append(record):harness.append(RECORDS,record)

def read_records():
    latest={}
    if RECORDS.exists():
        for line in RECORDS.read_text().splitlines():
            r=json.loads(line);latest[r['key']]=r
    return list(latest.values())

def validate_result(result,seconds,profile,variant):
    if result['sample_rate']!=48000 or result['frames']!=round(seconds*25):raise RuntimeError('Rate/frame mismatch')
    if profile:
        stages={e['stage'] for e in result.get('events',[])}
        if not {'sampling','sampling_loop','text_tokenize','dit_forward'}<=stages:raise RuntimeError('Incomplete profiling')
        count=sum(e['stage']=='dit_forward' for e in result['events'])
        if count!=harness.CFG['steps'][variant]:raise RuntimeError('Wrong step/forward count')

def preflight(monitor,done):
    raw=OUT/'raw';raw.mkdir(exist_ok=True)
    p=harness.PROMPTS[0]
    for variant in ('rf','mf'):
        for backend in ('cuda','vulkan'):
            for version in ('old','new'):
                case=case_for(version,variant,backend);key=case['case']+'/preflight'
                if key in done:continue
                with harness.Worker(case,monitor) as worker:
                    outputs=[];arrays=[]
                    for profile in (False,True):
                        path=raw/f'{case["case"]}-{profile}.raw'
                        result=worker.request(dict(id=key,text=p['text'],seconds=2.,cache='reencode',profile=profile,output=str(path)))
                        validate_result(result,2.,profile,variant)
                        outputs.append(result);arrays.append(np.fromfile(path,dtype='<f4'))
                    relative=float(np.linalg.norm(arrays[0]-arrays[1])/max(np.linalg.norm(arrays[0]),1e-12))
                    if relative>1e-4:raise RuntimeError('Profiling changes PCM: '+str(relative))
                    append(dict(key=key,kind='preflight',status='ok',case=case,load=worker.ready,outputs=outputs,profile_relative_error=relative))
            a=np.fromfile(raw/f'speech-ab-old-{variant}-{backend}-False.raw',dtype='<f4')
            b=np.fromfile(raw/f'speech-ab-new-{variant}-{backend}-False.raw',dtype='<f4')
            append(dict(key=f'cross-version/{variant}/{backend}',kind='parity',status='ok',
                relative_error=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),
                max_abs=float(np.max(np.abs(a-b))),bit_equal=bool(np.array_equal(a,b))))

def measure(phase,pairs,prompts,monitor,done):
    for variant,backend in pairs:
        for round_id in range(3):
            order=('old','new') if round_id%2==0 else ('new','old')
            for version in order:
                case=case_for(version,variant,backend)
                groups=[(cache,p,float(seconds)) for cache in harness.CFG['cache_profiles'] for p in prompts for seconds in (8,24)]
                prefix=f'{phase}/{case["case"]}/round-{round_id}'
                def complete(cache,p,seconds):
                    group=f'{prefix}/{cache}/{p["id"]}/{seconds}'
                    return all(f'{group}/{kind}/{i}' in done for kind in ('wall','profile') for i in range(5))
                if all(complete(*g) for g in groups):continue
                print('WORKER',phase,case['case'],'round',round_id,'order',order,flush=True)
                with harness.Worker(case,monitor) as worker:
                    cold=worker.request(dict(id=prefix+'/cold',text=harness.PROMPTS[0]['text'],seconds=8.,cache='reencode',profile=False))
                    append(dict(key=prefix+'/cold',kind='cold',status='ok',phase=phase,round=round_id,case=case,load=worker.ready,result=cold))
                    for cache,p,seconds in groups:
                        if complete(cache,p,seconds):continue
                        group=f'{prefix}/{cache}/{p["id"]}/{seconds}'
                        common=dict(status='ok',phase=phase,round=round_id,case=case,cache=cache,prompt=p,seconds=seconds)
                        request=dict(id=group,text=p['text'],seconds=seconds,cache=cache,profile=False)
                        for _ in range(2):worker.request(request)
                        monitor.reset();time.sleep(.2)
                        append(dict(common,key=group+'/idle',kind='idle',result=monitor.snapshot()))
                        for profile in (False,True):
                            kind='profile' if profile else 'wall'
                            for rep in range(5):
                                key=f'{group}/{kind}/{rep}'
                                if key in done:continue
                                path=OUT/'wav'/f'{phase}-{version}-{variant}-{backend}-{p["id"]}-{int(seconds)}.wav'
                                raw=path.with_suffix('.raw')
                                save=round_id==0 and cache=='native_cache' and not profile and rep==0
                                result=worker.request(dict(request,id=key,profile=profile,output=str(raw) if save else '-'))
                                validate_result(result,seconds,profile,variant)
                                if save:
                                    audio=np.fromfile(raw,dtype='<f4');sf.write(path,audio,48000,subtype='FLOAT');raw.unlink()
                                append(dict(common,key=key,kind=kind,result=result));done.add(key)
                        print('DONE',group,flush=True)

def paired_interval(old,new):
    """Hierarchical bootstrap: preserve round pairing; resample the five requests inside each worker."""
    rng=np.random.default_rng(42)
    rounds=rng.integers(0,3,(3000,3))
    a=np.array([old[r] for r in range(3)])[rounds[:,:,None],rng.integers(0,5,(3000,3,5))]
    b=np.array([new[r] for r in range(3)])[rounds[:,:,None],rng.integers(0,5,(3000,3,5))]
    ratios=np.median(np.median(b,axis=2),axis=1)/np.median(np.median(a,axis=2),axis=1)
    return np.percentile(ratios,[2.5,97.5]).tolist()

def summarize(records):
    groups=collections.defaultdict(list)
    for r in records:
        if r['kind'] in ('wall','profile','idle') and r['status']=='ok':
            c=r['case'];groups[(r['phase'],c['variant'],c['backend'],r['cache'],r['prompt']['id'],r['seconds'])].append(r)
    rows=[]
    for group,xs in sorted(groups.items()):
        row=dict(zip(('phase','variant','backend','cache','prompt','seconds'),group));results={}
        row.update(tokens=xs[0]['prompt'].get('tokens'),characters=xs[0]['prompt'].get('characters'),
                   precision='f16',codec_precision='fp32')
        for version in ('old','new'):
            wall=[r for r in xs if r['case']['version']==version and r['kind']=='wall']
            profiles=[components(r['result']) for r in xs if r['case']['version']==version and r['kind']=='profile']
            idle=[r['result'] for r in xs if r['case']['version']==version and r['kind']=='idle']
            cold=[r for r in records if r['kind']=='cold' and r['phase']==row['phase'] and
                r['case']['variant']==row['variant'] and r['case']['backend']==row['backend'] and r['case']['version']==version]
            if cold:
                row[version+'_load_ms']=stats.median(r['load']['load_ms'] for r in cold)
                row[version+'_startup_ms']=stats.median(r['load']['startup_wall_ms'] for r in cold)
                row[version+'_cold_ms']=stats.median(r['result']['ms'] for r in cold)
                row[version+'_loaded_resident_mib']=stats.median(r['load']['resident_vram']/2**20 for r in cold)
                row[version+'_load_peak_mib']=max(r['load']['peak_vram']/2**20 for r in cold)
            block={r:[x['result']['ms'] for x in wall if x['round']==r] for r in range(3)}
            results[version]=block
            values=[x['result']['ms'] for x in wall]
            row[version+'_n']=len(values);row[version+'_profile_n']=len(profiles)
            if not values:continue
            row[version+'_ms']=stats.median([stats.median(v) for v in block.values() if v])
            row[version+'_min_ms']=min(values);row[version+'_max_ms']=max(values);row[version+'_stdev_ms']=stats.pstdev(values)
            row[version+'_rtf']=row[version+'_ms']/1000/row['seconds']
            row[version+'_peak_mib']=max(x['result']['peak_vram'] for x in wall)/2**20
            row[version+'_resident_mib']=stats.median(x['resident_vram']/2**20 for x in idle)
            for metric in ('sampling_ms','sampling_loop_ms','decode_ms','forward_b1_ms','forward_b4_ms','forward_calls',
                'text_tokenize_ms','caption_tokenize_ms','text_encode_ms','caption_encode_ms','shared_encoder_ms',
                'reference_encode_ms','reference_preprocess_ms','reference_codec_ms','speaker_encode_ms'):
                vals=[p[metric] for p in profiles if p[metric] is not None]
                row[version+'_'+metric]=stats.median(vals) if vals else None
        if all(len(results[v][r])==5 for v in results for r in range(3)):
            row['delta_pct']=(row['new_ms']/row['old_ms']-1)*100
            lo,hi=paired_interval(results['old'],results['new'])
            row['ci_low_pct']=(lo-1)*100;row['ci_high_pct']=(hi-1)*100
            deltas=[stats.median(results['new'][r])/stats.median(results['old'][r])-1 for r in range(3)]
            row['round_deltas_pct']=json.dumps([d*100 for d in deltas])
            speed=(hi<1 or lo>1) and (all(d>0 for d in deltas) or all(d<0 for d in deltas))
            peaks={v:[max(x['result']['peak_vram'] for x in xs if x['kind']=='wall' and x['case']['version']==v and x['round']==r)/2**20 for r in range(3)] for v in ('old','new')}
            mem=[b-a for a,b in zip(peaks['old'],peaks['new'])]
            memory=abs(stats.median(mem))>=max(32,stats.median(peaks['old'])*.03) and (all(m>0 for m in mem) or all(m<0 for m in mem))
            row['difference']=bool(speed or memory);row['speed_difference']=bool(speed);row['memory_difference']=bool(memory)
            row['practical_speed_difference']=bool(speed and abs(row['delta_pct'])>=2)
            row['complete']=row['old_profile_n']==15 and row['new_profile_n']==15
        else:row['complete']=False;row['difference']=False
        rows.append(row)
    return rows

def write_report(records,rows,decision,preserved):
    DEST.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/'per-input.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    failures=[r for r in records if r['status']!='ok']
    validation=dict(errors=failures,incomplete=[r for r in rows if not r['complete']],preserved=preserved,
        preflight_count=sum(r['kind']=='preflight' for r in records),measured_conditions=len(rows),
        primary_requests=sum(r['kind']=='wall' for r in records),profile_requests=sum(r['kind']=='profile' for r in records))
    (OUT/'aggregate-validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    rf_groups=collections.defaultdict(list)
    for row in rows:
        if row['phase']=='expanded' and row['variant']=='rf' and row['complete']:
            rf_groups[(row['backend'],row['cache'],row['seconds'])].append(row)
    result_text='総合速度と段階別計測の結果を以下に示す。'
    if rf_groups and all(stats.median(r['new_ms'] for r in xs)>stats.median(r['old_ms'] for r in xs) for xs in rf_groups.values()):
        result_text='全20文章に拡張したRF各条件では、新版の通常推論中央値が旧版より増加した。'
    if not any(r.get('difference') for r in rows if r['phase']=='screen' and r['variant']=='mf'):
        result_text+=' MFは短い比較で一貫した差が出なかった。'
    text=['# speech.cpp 固定版／最新版 A/B比較','',f'旧版 `{BASELINE}`、新版 `{LATEST}`。既存の固定submodule・結果を保持。',
        '',result_text,
        '', '共通参照TYC054.wav（48kHz・PCM24）、同じF16モデル・FP32 codec、seed 42、RF 40／MF 4ステップ、同じcaption・CFG。固定8秒／24秒。CPU 8スレッド。',
        '', 'caption「自然な日本語で、落ち着いた口調で話す。」、RFはindependent CFG（text／caption 3、speaker 5、0.5〜1.0、linear）。短文p01はBOS込み15 tokens、長文p19は114 tokens。',
        '', '3ラウンド、旧→新／新→旧／旧→新の順で単独workerを実行。各入力2回warmup、通常速度5回、同期する段階別計測5回。表の時間はラウンド中央値の中央値。負の差は新版が高速。',
        '', '拡張条件: 通常推論の対応するラウンドを保つ階層bootstrap 95%区間が0を跨がず、3ラウンドとも同じ方向。小さな差も拡張対象とする（2%以上は別途practical_speed_difference列）。VRAMは32MiBかつ3%以上の差が同方向で再現。段階別時間は総合時間とは別実行。',
        '', '## 短文／長文の確認','', '| model | backend | cache | 入力 | 秒 | 旧 ms | 新 ms | 差 % | 95%区間 % | 旧／新peak MiB | 拡張対象 |',
        '|---|---|---|---|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        if r['phase']!='screen' or not r['complete']:continue
        text.append(f'| {r["variant"]} | {r["backend"]} | {r["cache"]} | {r["prompt"]} | {r["seconds"]:.0f} | {r["old_ms"]:.1f} | {r["new_ms"]:.1f} | {r["delta_pct"]:+.2f} | {r["ci_low_pct"]:+.2f}〜{r["ci_high_pct"]:+.2f} | {r["old_peak_mib"]:.0f}／{r["new_peak_mib"]:.0f} | {"あり" if r["difference"] else "なし"} |')
    labels='、'.join(v.upper()+' / '+b for v,b in decision['affected']) or 'なし'
    text+=['','## 全20文章への拡張','',f'拡張対象: **{labels}**。条件別の判定値は生データの `decision.json` に保存。','',
        '| model | backend | cache | 秒 | 入力数 | 旧 ms | 新 ms | 中央値の差 % | 差が再現した入力数 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    expanded=collections.defaultdict(list)
    for r in rows:
        if r['phase']=='expanded' and r['complete']:expanded[(r['variant'],r['backend'],r['cache'],r['seconds'])].append(r)
    for (v,b,c,s),xs in sorted(expanded.items()):
        a=stats.median(r['old_ms'] for r in xs);n=stats.median(r['new_ms'] for r in xs)
        text.append(f'| {v} | {b} | {c} | {s:.0f} | {len(xs)}/20 | {a:.1f} | {n:.1f} | {(n/a-1)*100:+.2f} | {sum(r["difference"] for r in xs)} |')
    text+=['','## 段階別計測（全20文章）','',
        '各セルは旧／新の順。文章ごとの詳細計測中央値を、20文章で中央値に集計する。総合速度とは別の推論でGPU完了を待って計測。',
        '', '| model / backend | cache | 秒 | sampling ms | loop ms | DiT B1 ms | DiT B4 ms | DACVAE ms | 常駐 MiB |',
        '|---|---|---:|---|---|---|---|---|---|']
    for (v,b,c,s),xs in sorted(expanded.items()):
        pairs=[]
        for metric in ('sampling_ms','sampling_loop_ms','forward_b1_ms','forward_b4_ms','decode_ms','resident_mib'):
            values=[]
            for version in ('old','new'):
                vals=[r[version+'_'+metric] for r in xs if r.get(version+'_'+metric) is not None]
                values.append(f'{stats.median(vals):.2f}' if vals else '—')
            pairs.append('／'.join(values))
        text.append(f'| {v} / {b} | {c} | {s:.0f} | '+' | '.join(pairs)+' |')
    text+=['','## ロード・初回推論','',
        '短文／長文比較の各ラウンドで、新しいworkerのロードとp01の8秒初回推論を測定。各版3回の中央値。',
        '', '| model / backend | 旧／新 load ms | 旧／新 起動からready ms | 旧／新 初回 ms |',
        '|---|---|---|---|']
    for v in ('rf','mf'):
        for b in ('cuda','vulkan'):
            xs=[r for r in records if r['kind']=='cold' and r['phase']=='screen' and r['case']['variant']==v and r['case']['backend']==b]
            pairs=[]
            for part,key in (('load','load_ms'),('load','startup_wall_ms'),('result','ms')):
                values=[f'{stats.median(r[part][key] for r in xs if r["case"]["version"]==version):.1f}' for version in ('old','new')]
                pairs.append('／'.join(values))
            text.append(f'| {v} / {b} | '+' | '.join(pairs)+' |')
    text+=['','全入力の最小・最大・標準偏差、段階別計測、RTF、常駐／ピークVRAMは[CSV](per-input.csv)に保存。',
        '', 'ロード・プロセス起動からready・初回推論もCSVに保存。初回は各ラウンドの新しいworkerでp01を8秒生成する要求（事前warmupなし）。OS・ドライバのキャッシュは保持する。',
        '', f'検証: 事前検証{validation["preflight_count"]}/8、未完了{len(validation["incomplete"])}、エラー{len(failures)}、既存ファイルのハッシュ保持={preserved}。',
        '', '品質評価は実施していない。新旧のPCM差と計測追加の有無によるPCM差は生データに記録。',
        '', f'生データ・モデルハッシュ・コマンド・ビルド設定・代表WAV: `artifacts/speech-ab/{LATEST}/measurement/`。']
    parity_path=OUT/'pcm-parity.json'
    if parity_path.exists():
        parity=json.loads(parity_path.read_text())
        text+=['',f'保存した新旧WAV {len(parity)*2}個を検証。PCMのbit単位一致: {sum(p["bit_equal"] for p in parity)}/{len(parity)}組。これは新旧の数値的一致の検証で、品質採点ではない。']
    (DEST/'RESULTS.ja.md').write_text('\n'.join(text)+'\n')
    if failures or validation['incomplete'] or not preserved or validation['preflight_count']!=8:raise RuntimeError('A/B validation failed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true');parser.add_argument('--reconsider-expansion',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'wav').mkdir(exist_ok=True)
    decision_path=OUT/'decision.json'
    if args.report_only:
        records=read_records();rows=summarize(records)
        write_report(records,rows,json.loads(decision_path.read_text()),preservation()==json.loads((OUT/'preservation.json').read_text()))
        return
    baseline_path=OUT/'preservation.json'
    if baseline_path.exists():
        if preservation()!=json.loads(baseline_path.read_text()):raise RuntimeError('Existing baseline artifacts changed')
    else:baseline_path.write_text(json.dumps(preservation(),indent=2)+'\n')
    run_settings=dict(baseline=BASELINE,latest=LATEST,conditions=harness.CFG,precision='f16',codec_precision='fp32',
        rounds=3,seconds=[8,24],screen_prompts=[harness.PROMPTS[0],max(harness.PROMPTS,key=lambda p:p['tokens'])],
        expansion='Affected model/backend pairs, all 20 prompts, both durations and cache conditions',
        thresholds=dict(confidence=95,same_direction_rounds=3,practical_delta_percent=2,vram_mib=32,vram_percent=3),
        inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'TYC054.wav',ROOT/'artifacts/benchmark/inputs/reference.wav',
            ROOT/'artifacts/benchmark/models/speech-rf-f16.gguf',ROOT/'artifacts/benchmark/models/speech-mf-f16.gguf',ROOT/'benchmark/prompts.json']})
    (OUT/'config.json').write_text(json.dumps(run_settings,ensure_ascii=False,indent=2)+'\n')
    commands={case_for(v,m,b)['case']:dict(command=command(case_for(v,m,b))[0],
        environment={k:x for k,x in command(case_for(v,m,b))[1].items() if k in ('OMP_NUM_THREADS','NVIDIA_TF32_OVERRIDE','LD_LIBRARY_PATH')})
        for v in ('old','new') for m in ('rf','mf') for b in ('cuda','vulkan')}
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    harness.command=command
    done={r['key'] for r in read_records() if r['status']=='ok'}
    with (ROOT/'artifacts/gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);monitor=harness.Monitor();stop=threading.Event()
        observer=threading.Thread(target=telemetry,args=(stop,),daemon=True);observer.start()
        try:
            preflight(monitor,done)
            pairs=[(m,b) for m in ('rf','mf') for b in ('cuda','vulkan')]
            measure('screen',pairs,run_settings['screen_prompts'],monitor,done)
            rows=summarize(read_records())
            if decision_path.exists() and not args.reconsider_expansion:decision=json.loads(decision_path.read_text())
            else:
                affected=sorted({(r['variant'],r['backend']) for r in rows if r['phase']=='screen' and r['complete'] and r['difference']})
                decision=dict(affected=affected,triggers=[{k:r[k] for k in ('variant','backend','cache','prompt','seconds','delta_pct','ci_low_pct','ci_high_pct','speed_difference','memory_difference')} for r in rows if r['phase']=='screen' and r.get('difference')])
                decision_path.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
            print('EXPANSION DECISION',json.dumps(decision,ensure_ascii=False),flush=True)
            measure('expanded',[tuple(p) for p in decision['affected']],harness.PROMPTS,monitor,done)
        finally:stop.set();observer.join();monitor.close()
    records=read_records();rows=summarize(records)
    write_report(records,rows,decision,preservation()==json.loads(baseline_path.read_text()))
    print('A/B COMPLETE',DEST,flush=True)

if __name__=='__main__':main()
