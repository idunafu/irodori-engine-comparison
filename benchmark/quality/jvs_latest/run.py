"""Run the current pinned speech.cpp against the existing one-clip JVS baseline."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run as quality_run
import report as quality_report
from common import ROOT,ASSETS,CONFIG,atomic_json,digest,environment,sha256,utc
from store import Store
from worker import command as original_command
sys.path.insert(0,str(ROOT/'scripts'))
from native_toolchain import vulkan_library

COMMIT='eacabcd891fefc2c3de3ca2479ff84e60f2679aa'
GGML_COMMIT='a655cf6910770404213bc338e622cd82b08ca277'
SOURCE=ROOT/'artifacts/quality/source'/('speech-'+COMMIT)
PARENT=ROOT/'artifacts/quality/runs/20261018-jvs'
DEFAULT_OUT=ROOT/'artifacts/quality/runs/20261019-jvs-one-latest-eacabcd'
BASE_CASES=['official-mf-bf16','official-rf-bf16','speech-mf-f16','speech-rf-f16']
NEW_WHERE="case_id IN ('speech-mf-f16','speech-rf-f16')"
vulkan_loader=vulkan_library()
BUILD_FLAGS=['-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=OFF','-DGGML_VULKAN=ON',
             '-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_NATIVE=ON','-DGGML_OPENMP=ON',
             *(['-DVulkan_LIBRARY='+vulkan_loader] if vulkan_loader else []),
             '-DCMAKE_PREFIX_PATH='+str(ROOT/'toolchains/1.4.357.0/x86_64'),
             '-DVulkan_INCLUDE_DIR='+str(ROOT/'toolchains/1.4.357.0/x86_64/include'),
             '-DVulkan_GLSLC_EXECUTABLE='+str(ROOT/'toolchains/1.4.357.0/x86_64/bin/glslc'),
             '-DVulkan_GLSLANG_VALIDATOR_EXECUTABLE='+str(ROOT/'toolchains/1.4.357.0/x86_64/bin/glslangValidator')]


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def parent_rows():
    conn=sqlite3.connect(f'file:{PARENT}/results.sqlite3?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    rows=list(conn.execute("SELECT * FROM tasks WHERE case_id IN (?,?,?,?) AND mode='native' AND profile='practical' AND json_extract(spec,'$.item.kind')='one' ORDER BY rowid",BASE_CASES))
    if len(rows)!=10000:raise ValueError('Expected 10,000 cached official/previous speech.cpp tasks')
    return conn,rows


def plan():
    if git(SOURCE,'rev-parse','HEAD')!=COMMIT or git(SOURCE/'ggml','rev-parse','HEAD')!=GGML_COMMIT:
        raise ValueError('Latest-source or ggml pin mismatch')
    if git(SOURCE,'status','--porcelain','--untracked-files=no') or git(SOURCE/'ggml','status','--porcelain','--untracked-files=no'):
        raise ValueError('Evaluation source checkout is modified')
    old=json.loads((PARENT/'manifest.json').read_text())
    code=[HERE/'run.py',HERE/'CMakeLists.txt',HERE.parent/'speech_worker.cpp',HERE.parent/'run.py',
          HERE.parent/'common.py',HERE.parent/'worker.py',HERE.parent/'report.py',HERE.parent/'store.py',
          HERE.parent/'score_worker.py',HERE.parent/'config.json',HERE.parent/'dataset.py',
          ROOT/'models.lock.json',ROOT/'engines.lock.json',ROOT/'scripts/native_toolchain.py',
          ASSETS/'jvs-index.json',ASSETS/'models.lock.json',
          ASSETS/'requirements-resolved.txt',ASSETS/'runtime-packages.json',
          PARENT/'manifest.json',PARENT/'results.sqlite3',PARENT/'speaker-centroids.json',PARENT/'splits.json']
    # Reuse the original scorer and aggregation rules byte for byte.
    for name in ['config.json','score_worker.py','report.py','dataset.py']:
        rel='benchmark/quality/'+name
        if sha256(ROOT/rel)!=old['files'][rel]:raise ValueError('Baseline evaluation implementation changed: '+name)
    for variant in ['mf','rf']:
        cmd,_=original_command(dict(engine='speech',variant=variant,precision='f16'))
        model=Path(cmd[1]);rel=str(model.relative_to(ROOT))
        if sha256(model)!=old['files'][rel]:raise ValueError('Existing F16 model changed')
        code.append(model)
    code.append(ASSETS/'models/campplus'/CONFIG['models']['campplus']['file'])
    code+=list((ASSETS/'models/whisper').rglob('*'))
    code=[p for p in code if p.is_file()]
    index=json.loads((ASSETS/'jvs-index.json').read_text())
    if len(index)!=100:raise ValueError('Expected all 100 JVS speakers')
    clips={}
    for speaker in index:
        selected=[speaker['references'][0],*speaker['parallel'][:15]]
        if set(c['path'] for c in speaker['parallel'][:5])&set(c['path'] for c in speaker['parallel'][5:15]):
            raise ValueError('JVS target and held-out clips overlap')
        for clip in selected:
            if sha256(ROOT/clip['path'])!=clip['sha256']:raise ValueError('JVS input hash changed: '+clip['path'])
            clips[clip['path']]=clip['sha256']
    conn,rows=parent_rows()
    selection=[]
    for row in rows:
        spec=json.loads(row['spec']);item=spec['item']
        if row['status']!='ok' or len(item['references'])!=1 or item['caption'] or item['reference_missing']:
            raise ValueError('Incomplete or incompatible cached baseline task: '+row['id'])
        scores=list(conn.execute('SELECT * FROM scores WHERE task_id=?',(row['id'],)))
        if {r['scorer'] for r in scores if r['status']=='ok'}!={'whisper','campplus'}:
            raise ValueError('Cached scores unavailable: '+row['id'])
        if any(r['wav_sha256']!=row['wav_sha256'] for r in scores):raise ValueError('Cached scores refer to another waveform')
        selection.append(dict(id=row['id'],case=row['case_id'],spec_sha256=digest(spec),wav_sha256=row['wav_sha256']))
    conn.close()
    return dict(format=1,datasets=['jvs'],source_url='https://github.com/nyosegawa/speech.cpp',
                source_commit=COMMIT,ggml_commit=GGML_COMMIT,model_precision='F16',runtime='Vulkan default',
                parent=str(PARENT),references='first nonpara30 clip only; identical to saved baseline',
                speakers=sorted(s['id'] for s in index),texts_per_speaker=5,seeds=list(range(5)),variants=['mf','rf'],
                new_generations=5000,reused_official=5000,reused_previous=5000,
                steps=CONFIG['steps'],caption='',batch=1,candidates=1,watermark=False,trim_tail=False,
                duration='native automatic prediction',initial_noise='native seeded RNG; same procedure as original JVS run',
                build_flags=BUILD_FLAGS,pilot_speakers=sorted(s['id'] for s in index)[:10],
                packages={name:importlib.metadata.version(name) for name in ['numpy','soundfile','soxr','torch','torchaudio','transformers','sherpa-onnx']},
                files={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(code))},
                jvs_inputs=clips,baseline_selection_digest=digest(selection))


def ensure_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Resume conditions or artifact hashes changed: '+str(path))
    else:atomic_json(path,value)


def terminate_process(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
        except (ProcessLookupError,subprocess.TimeoutExpired):
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


class Campaign(quality_run.Campaign):
    build_proc=None

    def build(self):
        binary=self.directory/'build/speech-latest-quality'
        stamp=self.directory/'build-manifest.json'
        if stamp.exists():
            saved=json.loads(stamp.read_text())
            if not binary.exists() or sha256(binary)!=saved['binary_sha256']:
                raise ValueError('Built latest binary changed; refusing resume')
            return binary
        self.stage('build-latest-vulkan')
        commands=[['cmake','-S',str(HERE),'-B',str(binary.parent),'-DSPEECH_SOURCE='+str(SOURCE),*BUILD_FLAGS],
                  ['cmake','--build',str(binary.parent),'--target','speech-latest-quality','--parallel','8']]
        with (self.directory/'build.log').open('a',buffering=1) as log:
            for cmd in commands:
                print(utc(),' '.join(cmd),flush=True)
                try:
                    self.build_proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=self.build_proc.wait()
                    if code:raise RuntimeError('Build failed, see build.log; exit '+str(code))
                finally:
                    terminate_process(self.build_proc);self.build_proc=None
        cache=binary.parent/'CMakeCache.txt'
        atomic_json(stamp,dict(source_commit=COMMIT,ggml_commit=GGML_COMMIT,flags=BUILD_FLAGS,
                               binary=str(binary),binary_sha256=sha256(binary),cmake_cache_sha256=sha256(cache),
                               built_at=utc(),commands=commands))
        return binary

    def seed_tasks(self,pilot):
        self.stage('reuse-baselines-and-enumerate',15000)
        conn,rows=parent_rows();new=[];imported=0
        for row in rows:
            spec=json.loads(row['spec']);case=spec['case'];is_previous=case['engine']=='speech'
            if is_previous:
                latest=json.loads(row['spec']);latest['case']['commit']=COMMIT
                key=digest([row['id'],COMMIT])[:24]
                new.append((key,latest,latest['item']['speaker'] in pilot))
                case['engine']='previous';case['id']='previous-'+case['variant']+'-f16'
            item=spec['item']
            values=(row['id'],json.dumps(spec,ensure_ascii=False),case['id'],row['dataset'],row['profile'],row['mode'],
                    int(item['speaker'] in pilot),row['status'],row['result'],row['wav'],row['wav_sha256'])
            existing=self.store.db.execute('SELECT spec,status,wav_sha256 FROM tasks WHERE id=?',(row['id'],)).fetchone()
            if existing and (json.loads(existing['spec'])!=spec or existing['status']!=row['status'] or existing['wav_sha256']!=row['wav_sha256']):
                raise ValueError('Reused task differs from its saved source')
            self.store.db.execute('INSERT OR IGNORE INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',values)
            for score in conn.execute('SELECT * FROM scores WHERE task_id=?',(row['id'],)):
                self.store.db.execute('INSERT OR IGNORE INTO scores VALUES(?,?,?,?,?)',tuple(score))
            imported+=1;self.progress()
            if imported%1000==0:self.store.db.commit()
        for key,spec,p in new:
            old=self.store.db.execute('SELECT spec FROM tasks WHERE id=?',(key,)).fetchone()
            if old and json.loads(old[0])!=spec:raise ValueError('New task conditions changed')
        self.store.populate(new);conn.close()
        for _ in new:self.progress()
        for name in ['speaker-centroids.json','splits.json']:
            dest=self.directory/name;source=PARENT/name
            if dest.exists():
                if sha256(dest)!=sha256(source):raise ValueError('Saved centroid or split changed')
            else:shutil.copyfile(source,dest)
        self.store.verify();self.save()

    def estimate(self):
        rows=list(self.store.db.execute(f"SELECT wav,result FROM tasks WHERE {NEW_WHERE} AND status='ok'"))
        count=len(rows);remaining=self.store.db.execute(f"SELECT COUNT(*) FROM tasks WHERE {NEW_WHERE} AND status='pending'").fetchone()[0]
        size=sum(Path(r['wav']).stat().st_size for r in rows)
        mean=size/max(1,count);free=shutil.disk_usage(self.directory).free
        generated_seconds=sum(json.loads(r['result'])['ms']/1000 for r in rows)
        scoring=list(self.store.db.execute(f"SELECT s.result FROM scores s JOIN tasks t ON s.task_id=t.id WHERE {NEW_WHERE} AND s.status='ok'"))
        scoring_seconds=sum(json.loads(r[0]).get('ms',0)/1000 for r in scoring)
        estimated_seconds=(generated_seconds/max(1,count)+2*scoring_seconds/max(1,len(scoring)))*remaining
        value=dict(pilot_generations=count,remaining_requests=remaining,mean_wav_bytes=mean,
                   estimated_additional_bytes=mean*remaining,reserve_factor=1.3,free_bytes=free,
                   estimated_remaining_seconds=estimated_seconds,status='ok' if mean*remaining*1.3<free else 'insufficient_space')
        atomic_json(self.directory/'estimate.json',value)
        if value['status']!='ok':raise RuntimeError('Insufficient disk space')
        print(utc(),'pilot-estimate',json.dumps(value),flush=True)


def additional_report(directory):
    dest=directory/'report'
    rows=[json.loads(line) for line in (dest/'per-input.jsonl').read_text().splitlines()]
    grouped=defaultdict(list)
    for row in rows:
        if row['kind']!='one':raise ValueError('Unexpected reference condition in report')
        grouped[row['case']].append(row)
    comparisons=[]
    for variant in ['mf','rf']:
        latest=grouped['speech-'+variant+'-f16']
        for baseline in ['official-'+variant+'-bf16','previous-'+variant+'-f16']:
            cached=grouped[baseline]
            for metric in ['camplus_cosine','speaker_top1','whisper_cer']:
                cluster=[]
                for selected in [latest,cached]:
                    values=defaultdict(lambda:[0.,0.])
                    for row in selected:
                        if metric in row['metrics']:
                            speaker=row['item'].split(':')[1]
                            values[speaker][0]+=row['metrics'][metric];values[speaker][1]+=1
                    cluster.append(values)
                interval=quality_report.bootstrap_difference(*cluster,CONFIG['bootstrap_replicates'])
                if interval:comparisons.append(dict(variant=variant,baseline=baseline,metric=metric,**interval))
    quality_report.write_csv(dest/'latest-paired-differences.csv',comparisons)
    import csv
    summary=list(csv.DictReader((dest/'summary.csv').open()))
    def metric(case,name):
        return float(next(r['mean'] for r in summary if r['case']==case and r['metric']==name and r['kind']=='one' and r['success_only']=='False'))
    coverage=json.loads((dest/'coverage.json').read_text())
    validation=dict(expected_new_generations=5000,new_generations=directory_counts(directory),
                    expected_reused_official=5000,expected_reused_previous=5000,
                    source_commit=COMMIT,ggml_commit=GGML_COMMIT,coverage=coverage)
    atomic_json(directory/'validation.json',validation)
    lines=['# 最新speech.cpp：JVS100話者・1クリップ参照','',
           f'最新コミット `{COMMIT}`、ggml `{GGML_COMMIT}`。',
           '100話者×5文章×5 seed×MF/RF＝新規5,000音声。speech.cppはF16重み・通常Vulkan。',
           '公式CUDA BF16・旧speech.cpp F16の各5,000音声および採点結果を既存評価から再利用した。',
           '各実装の自動長予測・native seed処理を使用。captionは空、末尾トリムと透かしは無効。',
           '参照は既存と同じnonpara30先頭1クリップ。照合用parallel100の10音声から作った100話者centroidも再利用。','',
           'CAM++ cosine×100を類似度％として表示する。Top-1は100話者識別の正答率。','',
           '| 方式 | 実装 | 類似度 | 話者Top-1 | Whisper CER |','|---|---|---:|---:|---:|']
    for variant in ['mf','rf']:
        for case,label in [('official-'+variant+'-bf16','公式BF16（再利用）'),('previous-'+variant+'-f16','旧speech.cpp F16（再利用）'),('speech-'+variant+'-f16','最新speech.cpp F16')]:
            lines.append(f"| {variant.upper()} | {label} | {metric(case,'camplus_cosine')*100:.2f}% | {metric(case,'speaker_top1')*100:.2f}% | {metric(case,'whisper_cer')*100:.2f}% |")
    lines+=['','最新版−比較基準の差。単位は％ポイント、95%区間は話者単位の対応bootstrap（2,000回）。','',
            '| 方式 | 基準 | 指標 | 差 | 95%区間 |','|---|---|---|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['variant'].upper()} | {row['baseline']} | {row['metric']} | {row['difference']*100:+.3f} | [{row['low']*100:+.3f}, {row['high']*100:+.3f}] |")
    lines+=['','生成失敗・採点失敗はcoverage.jsonとfailures.csvを参照。各seedの値・母標準偏差はper-seed.csvとsummary.csv。',
            '修正前からは49コミットとggmlの更新を含むため、差を単一修正だけの効果とは断定しない。',
            '自動評価であり、聴感上の自然さや品質同等性は判定しない。','']
    (dest/'LATEST.ja.md').write_text('\n'.join(lines))


def directory_counts(directory):
    with sqlite3.connect(f'file:{directory}/results.sqlite3?mode=ro',uri=True) as conn:
        return dict(conn.execute(f'SELECT status,COUNT(*) FROM tasks WHERE {NEW_WHERE} GROUP BY status').fetchall())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    directory=args.out.resolve();directory.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        quality_report.report(directory);additional_report(directory);return
    owner=(directory/'run.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=Campaign(directory,args.retry_failed)
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        campaign.stage('validate-inputs-and-pins')
        conditions=plan();ensure_json(directory/'plan.json',conditions)
        binary=campaign.build()
        ensure_json(directory/'manifest.json',dict(conditions,worker_binary_sha256=sha256(binary)))
        campaign.seed_tasks(set(conditions['pilot_speakers']))
        def latest_command(case):
            if case['engine']!='speech':raise ValueError('Only latest speech.cpp may generate in this run')
            old,env=original_command(case)
            for flag in ['GGML_VK_DISABLE_F16','GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2']:
                env.pop(flag,None)
            return [str(binary),old[1],old[2]],env
        quality_run.command=latest_command
        campaign.stage('waiting-for-gpu-lock')
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            campaign.generate(NEW_WHERE+' AND pilot=1','pilot-generation')
            campaign.score(NEW_WHERE+' AND pilot=1','pilot-scoring')
            failures=campaign.store.db.execute(f"SELECT COUNT(*) FROM tasks WHERE {NEW_WHERE} AND pilot=1 AND status='failed'").fetchone()[0]
            score_failures=campaign.store.db.execute(f"SELECT COUNT(*) FROM scores s JOIN tasks t ON s.task_id=t.id WHERE {NEW_WHERE} AND t.pilot=1 AND s.status='failed'").fetchone()[0]
            if failures or score_failures:raise RuntimeError('Pilot has failed requests or scores; inspect saved reasons')
            campaign.estimate()
            campaign.generate(NEW_WHERE,'full-generation')
            campaign.score(NEW_WHERE,'full-scoring')
        campaign.stage('aggregate-and-report')
        quality_report.report(directory);additional_report(directory)
        pending=campaign.store.db.execute(f"SELECT COUNT(*) FROM tasks WHERE {NEW_WHERE} AND status='pending'").fetchone()[0]
        if pending:raise RuntimeError('Planned requests remain pending')
        campaign.state.update(status='complete',stage='complete',exit_code=0,finished_at=utc(),new_generations=directory_counts(directory));campaign.save()
    except BaseException as error:
        campaign.state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                              error=f'{type(error).__name__}: {error}',exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,finished_at=utc());campaign.save()
        traceback.print_exc();raise
    finally:
        campaign.close_worker();terminate_process(campaign.build_proc)
        campaign.finished.set();campaign.thread.join(timeout=2);campaign.store.db.close()


if __name__=='__main__':main()
