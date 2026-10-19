"""F16-weight speech.cpp extension of the immutable sample-rate experiment."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'sample_rate'))
import run as base
from analyze import interval
from common import atomic_json, digest, sha256, utc
from worker import command

PARENT = base.DEFAULT_OUT
DEFAULT_OUT = base.ROOT/'artifacts/quality/runs/20261019-tsukuyomi-samplerate-f16'
MODE = 'speech_f16'


def read_rows(folder, kind):
    with sqlite3.connect(f'file:{folder}/results.sqlite3?mode=ro', uri=True) as conn:
        return [json.loads(r[0]) for r in conn.execute('SELECT result FROM tasks WHERE kind=? AND status="ok" ORDER BY id',(kind,))]


def validate_parent():
    state=json.loads((PARENT/'status.json').read_text())
    validation=json.loads((PARENT/'validation.json').read_text())
    if state['status']!='complete' or validation['failed'] or validation['generations']!=3600 or validation['scores']!=3600:
        raise ValueError('Parent experiment must be complete without missing results')
    manifest=json.loads((PARENT/'manifest.json').read_text())
    for name,checksum in manifest['sources'].items():
        if sha256(name)!=checksum:raise ValueError('Parent pinned source changed: '+name)
    if sha256(base.ROOT/'wav_tsukuyomi.zip')!=manifest['archive_sha256']:
        raise ValueError('Source archive changed')
    inputs=json.loads((PARENT/'inputs.json').read_text())
    for row in inputs:
        if row['id'] in base.REFS+base.HELDOUT and sha256(row['path'])!=row['sha256']:
            raise ValueError('Parent input changed: '+row['path'])


def configuration():
    paths=[Path(__file__),HERE.parent/'sample_rate/run.py',HERE.parent/'sample_rate/analyze.py',
           base.PREVIOUS/'speech-diagnostic',base.PREVIOUS/'codec-audit',
           HERE.parent/'score_worker.py',HERE.parent/'worker.py',HERE.parent/'common.py',HERE.parent/'config.json',
           PARENT/'manifest.json',PARENT/'results.sqlite3',PARENT/'speaker-centroid.json',PARENT/'inputs.json']
    for variant in ['mf','rf']:
        native,_=command(dict(engine='speech',variant=variant,precision='f16'))
        paths.append(Path(native[1]))
    return dict(schema=1,parent=str(PARENT),rates=base.RATES,references=base.REFS,heldout=base.HELDOUT,
                target_ids=base.TARGET_IDS,seeds=list(range(5)),variants=['mf','rf'],mode=MODE,
                model_precision='F16 weights',codec_precision='audit stored codec input-convolution weight dtype separately',
                internal_vulkan='default; F16 operand paths enabled',generation_count=900,reference_count=60,
                baseline='Saved official FP32 and speech.cpp FP32 results; no regeneration or condition changes',
                generation_length_and_noise='Reuse exact frame counts and noise files of every parent matched take',
                sources={str(p):sha256(p) for p in sorted(set(paths))})


class Experiment(base.Experiment):
    def input(self, rate, ref):
        return PARENT/'inputs'/str(rate)/(ref+'.wav')

    def speech(self,variant):
        native,_=command(dict(engine='speech',variant=variant,precision='f16'))
        return [str(base.PREVIOUS/'speech-diagnostic'),native[1],native[2]]

    def feature_paths(self,variant,rate,ref):
        return self.paths(variant,rate,ref)

    def encode_f16(self):
        self.stage('codec-precision-audit',2)
        for variant in ['mf','rf']:
            native=self.speech(variant)
            with self.process([str(base.PREVIOUS/'codec-audit'),native[1],'gpu'],'codec-audit-'+variant):
                stem=self.out/('codec-audit-'+variant)
                files=[str(stem.with_suffix('.windowed.f32')),str(stem.with_suffix('.whole.f32'))]
                pcm=PARENT/'features/speech/48000'/f'{base.REFS[0]}.pcm.f32'
                result=self.request('audit:'+variant,'codec_audit',dict(pcm=str(pcm),windowed=files[0],whole=files[1]),files,
                                    dict(variant=variant,pcm_sha256=sha256(pcm)))
                if result.get('failed') or result['codec_input_dtype']!=0:
                    raise ValueError('Expected F32 stored codec input-convolution weights')
        self.stage('f16-reference-encoding',60)
        for variant in ['mf','rf']:
            with self.process(self.speech(variant),'encode-'+variant):
                for rate in base.RATES:
                    for ref in base.REFS:
                        paths=self.feature_paths(variant,rate,ref)
                        request=dict(op='dump',reference=str(self.input(rate,ref)),**paths)
                        files=[paths[k] for k in ['resampled_out','pcm_out','latent_out','state_out']]
                        result=self.request(f'encode:{variant}:{rate}:{ref}','encode',request,files,
                                            dict(variant=variant,rate=rate,ref=ref))
                        if result.get('failed'):raise RuntimeError('F16 reference encoding failed: '+result['id'])

    def generate_f16(self):
        self.stage('f16-controlled-generation',900)
        parent_rows=read_rows(PARENT,'generation')
        selected=[r for r in parent_rows if r['mode']=='speech']
        if len(selected)!=900:raise ValueError('Missing matched parent speech.cpp FP32 take')
        by_key={(r['variant'],r['rate'],r['ref'],r['text_id'],r['seed']):r for r in selected}
        for variant in ['mf','rf']:
            with self.process(self.speech(variant),'generate-'+variant):
                for rate in base.RATES:
                    for ref in base.REFS:
                        for text_id,text in self.targets().items():
                            for seed in range(5):
                                row=by_key[variant,rate,ref,text_id,seed]
                                ident=digest([variant,ref,text_id,seed])[:24]
                                noise=PARENT/'noise'/(ident+'.f32')
                                if sha256(noise)!=row['noise_sha256']:raise ValueError('Parent common-noise hash mismatch')
                                frames=row['frames_expected']
                                wav=self.out/'audio'/variant/str(rate)/(ident+'.wav');wav.parent.mkdir(parents=True,exist_ok=True)
                                latent=self.feature_paths(variant,rate,ref)['latent_out']
                                request=dict(op='synthesize',text=text,seed=seed,seconds=float(np.nextafter(frames/25,np.inf)),
                                             noise=str(noise),output=str(wav),latent_in=latent)
                                self.request(f'generate:{variant}:{rate}:{ident}','generation',request,[wav],
                                             dict(variant=variant,mode=MODE,rate=rate,ref=ref,text_id=text_id,seed=seed,wav=str(wav),
                                                  frames_expected=frames,noise_sha256=row['noise_sha256'],latent_sha256=sha256(latent)))
        self.save()


def write_csv(out,name,rows):
    if not rows:return
    with (out/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(out):
    centroid=np.array(json.loads((PARENT/'speaker-centroid.json').read_text())['embedding'])
    actual=np.array(json.loads((out/'speaker-centroid.json').read_text())['embedding'])
    if not np.array_equal(centroid,actual):raise ValueError('Held-out centroid changed')
    parent_generation=read_rows(PARENT,'generation')
    original={ (r['variant'],r['mode'],r['rate'],r['ref'],r['text_id'],r['seed']):r for r in parent_generation }
    def key(r):return r['variant'],r['rate'],r['ref'],r['text_id'],r['seed']
    parent_score=read_rows(PARENT,'score')
    reference={ (r['variant'],r['mode'],r['rate'],r['ref'],r['text_id'],r['seed']):float(centroid@np.array(r['embedding'])) for r in parent_score }
    generated=read_rows(out,'generation');scores=read_rows(out,'score')
    mismatches=[]
    for row in generated:
        parent=original[row['variant'],'speech',row['rate'],row['ref'],row['text_id'],row['seed']]
        if row['noise_sha256']!=parent['noise_sha256'] or row['frames_expected']!=parent['frames_expected']:
            mismatches.append(row['id'])
    rows=[]
    for row in scores:
        own=float(centroid@np.array(row['embedding']))
        head=(row['variant'],row['rate'],row['ref'],row['text_id'],row['seed'])
        off=reference[(head[0],'official',*head[1:])]
        speech=reference[(head[0],'speech',*head[1:])]
        rows.append(dict(variant=row['variant'],rate=row['rate'],ref=row['ref'],text_id=row['text_id'],seed=row['seed'],
                         official_fp32=off,speech_fp32=speech,speech_f16=own,
                         delta_f16_vs_speech_fp32=own-speech,delta_f16_vs_official_fp32=own-off,wav=row['wav']))
    write_csv(out,'per-input.csv',rows)
    summary=[];differences=[]
    for variant in ['mf','rf']:
        for rate in base.RATES:
            group=[r for r in rows if r['variant']==variant and r['rate']==rate]
            if not group:continue
            saved=dict(variant=variant,rate=rate,takes=len(group))
            for metric in ['official_fp32','speech_fp32','speech_f16','delta_f16_vs_speech_fp32','delta_f16_vs_official_fp32']:
                saved[metric]=float(np.mean([r[metric] for r in group]))
            seed_means=[np.mean([r['speech_f16'] for r in group if r['seed']==seed]) for seed in range(5)]
            saved['seed_population_std']=float(np.std(seed_means))
            summary.append(saved)
            for metric in ['delta_f16_vs_speech_fp32','delta_f16_vs_official_fp32']:
                differences.append(dict(variant=variant,rate=rate,comparison=metric,**interval([(r['ref'],r[metric]) for r in group])))
    write_csv(out,'summary.csv',summary);write_csv(out,'paired-differences.csv',differences)
    tensors=[]
    for variant in ['mf','rf']:
        for rate in base.RATES:
            for ref in base.REFS:
                a=np.fromfile(PARENT/'features/speech'/str(rate)/(ref+'.latent.f32'),dtype='<f4')
                b=np.fromfile(out/'features'/variant/str(rate)/(ref+'.latent.f32'),dtype='<f4')
                tensors.append(dict(variant=variant,rate=rate,ref=ref,**base.metrics(a,b)))
    write_csv(out,'codec-latent-comparison.csv',tensors)
    with sqlite3.connect(f'file:{out}/results.sqlite3?mode=ro',uri=True) as conn:
        failures=[json.loads(r[0]) for r in conn.execute('SELECT result FROM tasks WHERE status="failed"')]
    atomic_json(out/'failures.json',failures)
    validation=dict(generations=len(generated),scores=len(scores),expected=900,encodings=len(read_rows(out,'encode')),
                    expected_encodings=60,failed=len(failures),length_or_noise_mismatches=mismatches,
                    identical_centroid=True,codec_latent_max_relative_l2=max(r['relative_l2'] for r in tensors))
    atomic_json(out/'validation.json',validation)
    if mismatches or (not failures and (len(generated)!=900 or len(scores)!=900)):
        raise ValueError('Planned F16 results incomplete or controls differ')
    lines=['# speech.cpp F16重みによるサンプルレート比較','',
           f'生成{len(generated)}/900件、採点{len(scores)}/900件、失敗{len(failures)}件。','',
           '既存FP32評価と同じ96・48・44.1・32・24・16kHz入力、参照5クリップ、3文章、seed 0〜4、MF/RFを使用。',
           '生成長と初期ノイズは元評価の各音声と同一。照合用10音声のCAM++ centroidも完全一致を確認した。',
           'speech.cppのTTS本体をF16重みへ変更し、通常Vulkanで測定。codec入力畳み込み重みはF32型を確認した。',
           '公式の比較基準は保存済みFP32結果。公式BF16やONNX FP16の追加評価は含まない。','',
           'CAM++ cosine×100の％表示（各セル75音声）。','',
           '| 入力kHz | MF公式FP32 | MF speech FP32 | MF speech F16 | RF公式FP32 | RF speech FP32 | RF speech F16 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    lookup={(r['variant'],r['rate']):r for r in summary}
    for rate in base.RATES:
        values=[]
        for variant in ['mf','rf']:
            row=lookup.get((variant,rate))
            values += [f"{row[k]*100:.2f}%" if row else '欠測' for k in ['official_fp32','speech_fp32','speech_f16']]
        lines.append('| '+f'{rate/1000:g}'+' | '+' | '.join(values)+' |')
    lines+=['','F16重み−FP32重み（同じspeech.cpp内）の差。単位は％ポイント。','',
            '| 入力kHz | MF | RF |','|---|---:|---:|']
    for rate in base.RATES:
        values=[]
        for variant in ['mf','rf']:
            row=lookup.get((variant,rate));values.append(f"{row['delta_f16_vs_speech_fp32']*100:+.4f}" if row else '欠測')
        lines.append('| '+f'{rate/1000:g}'+' | '+' | '.join(values)+' |')
    lines+=['','1話者の自動話者類似度評価。参照5クリップ単位の95%対応bootstrap区間はpaired-differences.csvに保存。',
            '聴感・自然さ・読みの正確さは今回の追加診断では評価していない。',
            f"codec latentのFP32重み実行時からの最大相対L2差：{validation['codec_latent_max_relative_l2']:.9g}。",'',
            '再集計：`.venvs/quality/bin/python benchmark/quality/sample_rate_f16/run.py --report-only`','']
    (out/'RESULTS.ja.md').write_text('\n'.join(lines))
    print(json.dumps(validation,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.report_only:report(out);return
    ownership=(out/'run.lock').open('a');fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Experiment(out,args.retry_failed)
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def heartbeat():
        while not run.stopped.wait(30):atomic_json(out/'status.json',dict(run.state,heartbeat_at=utc()))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        run.stage('validate-parent',1);validate_parent();run.progress()
        manifest=configuration();path=out/'manifest.json'
        if path.exists():
            if json.loads(path.read_text())!=manifest:raise ValueError('Resume manifest mismatch')
        else:atomic_json(path,manifest)
        run.stage('waiting-for-gpu-lock',0)
        with (base.ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run.encode_f16();run.generate_f16();run.score()
        run.stage('analysis',1);report(out);run.progress()
        run.state.update(status='complete',stage='complete',exit_code=0,finished_at=utc());run.save()
    except BaseException as error:
        run.state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                         error=f'{type(error).__name__}: {error}',exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,finished_at=utc())
        run.save();traceback.print_exc();raise
    finally:
        if run.worker:run.worker.close()
        run.stopped.set();thread.join(timeout=2);run.ledger.conn.close()


if __name__=='__main__':main()
