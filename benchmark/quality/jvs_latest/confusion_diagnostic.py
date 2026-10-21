"""Matched reference-path interventions on six affected speakers and two controls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import traceback

import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'sample_rate'))
import run as base
from common import ROOT,ASSETS,PYTHON,atomic_json,digest,sha256,utc
from worker import command
sys.path.insert(0,str(HERE))
import confusion
from report import bootstrap_difference,write_csv

OUT=confusion.OUT
RUN=confusion.RUN
MODES=['official_bf16_common','cpp_f16_kaiser_common','cpp_f16_hann_common',
       'cpp_fp32_hann_common','official_fp32_common','cpp_fp32_hann_strict']
BINARY=RUN/'build/speech-latest-quality'


def targets(speakers):
    conn=sqlite3.connect(f'file:{RUN}/results.sqlite3?mode=ro',uri=True)
    rows=[]
    for key,spec,result in conn.execute("SELECT id,spec,result FROM tasks WHERE case_id IN ('official-mf-bf16','official-rf-bf16') ORDER BY rowid"):
        spec=json.loads(spec)
        if spec['item']['speaker'] in speakers:
            rows.append(dict(source_id=key,spec=spec,frames=json.loads(result)['frames']))
    conn.close()
    assert len(rows)==len(speakers)*50
    return rows


def manifest(selection):
    files=[Path(__file__),HERE/'confusion.py',HERE/'diagnostic_official.py',
           HERE.parent/'sample_rate/run.py',HERE.parent/'python_worker.py',HERE.parent/'score_worker.py',
           HERE.parent/'worker.py',HERE.parent/'common.py',HERE.parent/'report.py',HERE.parent/'config.json',
           RUN/'manifest.json',RUN/'results.sqlite3',RUN/'speaker-centroids.json',
           OUT/'selection.json',ASSETS/'jvs-index.json',BINARY]
    for variant in ['mf','rf']:
        for precision in ['f16','fp32']:
            cmd,_=command(dict(engine='speech',variant=variant,precision=precision));files.append(Path(cmd[1]))
        files.append(ROOT/f'models/{variant}/model.safetensors')
    files.append(ROOT/'models/codec/weights.pth')
    return dict(format=1,selection=selection,modes=MODES,seeds=list(range(5)),texts_per_speaker=5,variants=['mf','rf'],
                expected_generations=len(selection['speakers'])*50*len(MODES),
                controls='Official BF16 native duration frames and one shared FP32 noise per text/speaker/seed/variant',
                hann='torchaudio defaults, mono 24kHz to 48kHz, saved float32 before loudness normalization',
                centroid='Original 100 speaker held-out centroid; unchanged',
                files={str(p):sha256(p) for p in sorted(set(files))})


class Diagnostic(base.Experiment):
    def natural(self):
        index=json.loads((ASSETS/'jvs-index.json').read_text())
        self.stage('classify-natural-recordings',600)
        with self.process([str(PYTHON),str(HERE.parent/'score_worker.py'),'campplus'],'natural-campplus'):
            for speaker in index:
                clips=[('reference',speaker['references'][0])]+[('natural_target',c) for c in speaker['parallel'][:5]]
                for role,clip in clips:
                    wav=ROOT/clip['path'];checksum=sha256(wav);assert checksum==clip['sha256']
                    self.request(f'natural:{role}:{speaker["id"]}:{clip["id"]}','natural',dict(wav=str(wav)),
                                 meta=dict(role=role,speaker=speaker['id'],clip=clip['id'],wav_sha256=checksum))
        atomic_json(OUT/'natural-scores.json',self.ledger.rows('natural'));self.save()

    def prepare_hann(self,speakers):
        import torch
        import torchaudio
        torch.set_num_threads(8)
        index={r['id']:r for r in json.loads((ASSETS/'jvs-index.json').read_text())}
        rows=[]
        for speaker in speakers:
            reference=ROOT/index[speaker]['references'][0]['path']
            audio,rate=sf.read(reference,dtype='float32',always_2d=True);audio=torch.from_numpy(audio.mean(1))
            if rate!=48000:audio=torchaudio.functional.resample(audio,rate,48000)
            dest=OUT/'hann-references'/(speaker+'.wav');dest.parent.mkdir(exist_ok=True)
            sf.write(dest,audio.numpy(),48000,subtype='FLOAT')
            rows.append(dict(speaker=speaker,source=str(reference),source_sha256=sha256(reference),
                             wav=str(dest),wav_sha256=sha256(dest),samples=len(audio)))
        atomic_json(OUT/'hann-references.json',rows)

    def generate(self,selected,pilot=False):
        rows=targets(selected)
        if pilot:
            first=selected[0]
            rows=[r for r in rows if r['spec']['item']['speaker']==first and r['spec']['item']['id'].endswith(':VOICEACTRESS100_001') and r['spec']['seed']==0]
            assert len(rows)==2
        self.stage('controlled-pilot' if pilot else 'controlled-generation',len(rows)*len(MODES))
        for variant in ['mf','rf']:
            subset=[r for r in rows if r['spec']['case']['variant']==variant]
            for mode in MODES:
                official=mode.startswith('official')
                precision='bf16' if mode.startswith('official_bf16') else ('f16' if mode.startswith('cpp_f16') else 'fp32')
                if official:
                    cmd=[str(ROOT/'.venvs/Irodori-TTS/bin/python'),str(HERE/'diagnostic_official.py'),variant,precision]
                    flags=dict(QUALITY_REFERENCE_CACHE=str(OUT/'official-reference-cache'))
                else:
                    old,_=command(dict(engine='speech',variant=variant,precision=precision));cmd=[str(BINARY),old[1],old[2]]
                    flags=base.STRICT if mode.endswith('_strict') else {}
                with self.process(cmd,('pilot-' if pilot else '')+mode+'-'+variant,flags):
                    for row in subset:
                        spec=row['spec'];item=spec['item'];speaker=item['speaker'];seed=spec['seed'];frames=row['frames']
                        group=digest([variant,item['id'],seed])[:24]
                        noise=OUT/'noise'/(group+'.f32');noise.parent.mkdir(exist_ok=True)
                        if not noise.exists():np.random.default_rng(int(group[:16],16)).standard_normal((frames,32),dtype=np.float32).tofile(noise)
                        wav=OUT/'audio'/variant/mode/(group+'.wav');wav.parent.mkdir(parents=True,exist_ok=True)
                        refs=item['references']
                        if not official and '_hann_' in mode:refs=[str(OUT/'hann-references'/(speaker+'.wav'))]
                        request=dict(text=item['text'],references=refs,caption='',seed=seed,
                                     seconds=float(np.nextafter(frames/25,np.inf)),noise=str(noise),output=str(wav))
                        result=self.request(f'controlled:{variant}:{mode}:{group}','generation',request,[wav],
                                            dict(variant=variant,mode=mode,speaker=speaker,item=item['id'],seed=seed,
                                                 frames_expected=frames,noise_sha256=sha256(noise),wav=str(wav)))
                        if result.get('failed'):raise RuntimeError('Controlled generation failed: '+result['id'])
                        if official and not result['noise_used']:raise ValueError('Official common-noise override unused')
        self.save()

    def score(self):
        rows=self.ledger.rows('generation');self.stage('controlled-campplus',len(rows))
        with self.process([str(PYTHON),str(HERE.parent/'score_worker.py'),'campplus'],'controlled-campplus'):
            for row in rows:
                wav=Path(row['wav']);assert sha256(wav)==row['files'][str(wav)]
                meta={k:row[k] for k in ['variant','mode','speaker','item','seed','frames_expected','noise_sha256','wav']}
                meta['wav_sha256']=sha256(wav)
                self.request('score:'+row['id'],'score',dict(wav=str(wav)),meta=meta)
        self.save()


def report(run,selection):
    centroids=json.loads((RUN/'speaker-centroids.json').read_text())['centroids'];ids=sorted(centroids);mat=np.array([centroids[s] for s in ids])
    scores=run.ledger.rows('score');rows=[]
    for row in scores:
        decision,_=confusion.classify(row['embedding'],row['speaker'],ids,mat)
        rows.append({**{k:row[k] for k in ['variant','mode','speaker','item','seed','frames_expected','noise_sha256','wav']},**decision})
    write_csv(OUT/'controlled-decisions.csv',rows)
    summary=[];per_speaker=[];differences=[]
    strata=dict(affected=set(selection['affected']),controls=set(selection['controls']),all=set(selection['speakers']))
    for variant in ['mf','rf']:
        for mode in MODES:
            for stratum,speakers in strata.items():
                group=[r for r in rows if r['variant']==variant and r['mode']==mode and r['speaker'] in speakers]
                if group:summary.append(dict(variant=variant,mode=mode,stratum=stratum,takes=len(group),
                                             top1=float(np.mean([r['correct'] for r in group])),
                                             similarity=float(np.mean([r['true_similarity'] for r in group])),
                                             margin=float(np.mean([r['margin'] for r in group]))))
            for speaker in selection['speakers']:
                group=[r for r in rows if r['variant']==variant and r['mode']==mode and r['speaker']==speaker]
                if group:
                    wrong=__import__('collections').Counter(r['predicted'] for r in group if not r['correct'])
                    per_speaker.append(dict(variant=variant,mode=mode,speaker=speaker,correct=sum(r['correct'] for r in group),takes=len(group),
                                            wrong=';'.join(f'{k}:{v}' for k,v in wrong.most_common())))
        for aa,bb in [('cpp_f16_kaiser_common','official_bf16_common'),('cpp_f16_hann_common','official_bf16_common'),
                      ('cpp_fp32_hann_common','official_fp32_common'),('cpp_fp32_hann_strict','official_fp32_common')]:
            for stratum,speakers in strata.items():
                for metric in ['correct','true_similarity']:
                    groups=[]
                    for mode in [aa,bb]:
                        values=defaultdict(lambda:[0.,0.])
                        for row in rows:
                            if row['variant']==variant and row['mode']==mode and row['speaker'] in speakers:
                                values[row['speaker']][0]+=row[metric];values[row['speaker']][1]+=1
                        groups.append(values)
                    interval=bootstrap_difference(*groups,2000)
                    if interval:differences.append(dict(variant=variant,left=aa,right=bb,stratum=stratum,metric=metric,**interval))
    write_csv(OUT/'controlled-summary.csv',summary);write_csv(OUT/'controlled-per-speaker.csv',per_speaker);write_csv(OUT/'controlled-differences.csv',differences)
    generated=run.ledger.rows('generation');groups=defaultdict(list)
    for row in generated:groups[row['variant'],row['item'],row['seed']].append(row)
    assert len(generated)==len(scores)==2400 and len(groups)==400
    assert all(len(g)==len(MODES) and len({r['noise_sha256'] for r in g})==len({r['frames_expected'] for r in g})==1 for g in groups.values())
    atomic_json(OUT/'validation.json',dict(generations=len(generated),scores=len(scores),groups=len(groups),
                                         expected_modes=6,common_noise_and_frames=True,natural_scores=len(run.ledger.rows('natural'))))
    text=['# 話者混同：生成条件を揃えた追加診断','',
          '元評価のMF/RF合計で誤り純増が大きい6話者＋両実装で全正解の男女各1話者を、介入出力を見る前に選択。',
          '悪化話者：'+', '.join(selection['affected'])+'。対照：'+', '.join(selection['controls'])+'。',
          '各セルは各話者5文章×5 seed。各文章の公式BF16自動生成長を全条件で固定し、同じFP32初期ノイズを注入する。',
          'hann条件では公式torchaudio既定方式で参照を48kHzへ変換してcppへ渡す。ラウドネス・codec・TTSは各実装の処理を維持。',
          'strictはVulkanの内部F16・cooperative matrixを無効にする追加診断。元評価の設定は変更していない。','',
          '| 方式 | 条件 | 悪化6話者Top-1 | 対照2話者Top-1 |','|---|---|---:|---:|']
    for variant in ['mf','rf']:
        for mode in MODES:
            def value(stratum):return next(r['top1'] for r in summary if r['variant']==variant and r['mode']==mode and r['stratum']==stratum)
            text.append(f"| {variant.upper()} | {mode} | {value('affected')*100:.2f}% | {value('controls')*100:.2f}% |")
    text+=['','| 話者 | 方式 | 条件 | 正解数/25 | 誤認先 |','|---|---|---|---:|---|']
    for row in per_speaker:
        if row['speaker'] in selection['affected']:
            text.append(f"| {row['speaker']} | {row['variant'].upper()} | {row['mode']} | {row['correct']} | {row['wrong']} |")
    text+=['','悪化例に基づく診断集合なので、この集合の集計値を100話者全体の改善率とは扱わない。',
           '全体の混同行列と公式比較はRESULTS.ja.md、介入の個別予測はcontrolled-decisions.csvに保存。','']
    (OUT/'CONTROLLED.ja.md').write_text('\n'.join(text))


def main():
    OUT.mkdir(parents=True,exist_ok=True);owner=(OUT/'run.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Diagnostic(OUT,False)
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def heartbeat():
        while not run.stopped.wait(30):atomic_json(OUT/'status.json',dict(run.state,heartbeat_at=utc()))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        selection=json.loads((OUT/'selection.json').read_text());conditions=manifest(selection);path=OUT/'manifest.json'
        if path.exists():
            if json.loads(path.read_text())!=conditions:raise ValueError('Diagnostic resume manifest mismatch')
        else:atomic_json(path,conditions)
        run.natural();run.prepare_hann(selection['speakers'])
        subprocess.run([str(PYTHON),str(HERE/'confusion.py')],cwd=ROOT,check=True)
        run.stage('waiting-for-gpu-lock',0)
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run.generate(selection['speakers'],pilot=True);run.score();run.generate(selection['speakers'])
        run.score();run.stage('analysis',1);report(run,selection);run.progress()
        run.state.update(status='complete',stage='complete',exit_code=0,finished_at=utc());run.save()
    except BaseException as error:
        run.state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                         error=f'{type(error).__name__}: {error}',exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,finished_at=utc());run.save();traceback.print_exc();raise
    finally:
        if run.worker:run.worker.close()
        run.stopped.set();thread.join(timeout=2);run.ledger.conn.close()


if __name__=='__main__':main()
