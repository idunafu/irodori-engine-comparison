"""Real GPU preflight independent of JVS availability; short outputs stay outside the campaign."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
from common import ASSETS, BASE, HERE, PYTHON, ROOT, atomic_json, cases, environment
from worker import Process, command


def main(lock_gpu=True):
    dest=BASE/'preflight';dest.mkdir(parents=True,exist_ok=True)
    heard=ASSETS/'speech-bench/src/measure/heard.ts'
    heard.with_name('quality-heard.ts').write_text(heard.read_text()+'\nexport { heardUnits };\n')
    noise=dest/'noise.f32'
    np.random.default_rng(123).standard_normal((25,32),dtype=np.float32).astype('<f4').tofile(noise)
    checks=[]
    frames={}
    def record(name,**result):
        checks.append(dict(name=name,**result));atomic_json(dest/'validation.json',checks)
        print(name,result,flush=True)
    def synth(worker,case,name,**overrides):
        output=dest/(case['id']+'-'+name+'.wav')
        req=dict(id=name,text='今日はよく晴れています。',seed=42,caption='',references=[],output=str(output))
        req.update(overrides)
        result=worker.request(req)
        if result['type']!='result':raise RuntimeError(str(result))
        if result.get('raw_float'):
            audio=np.fromfile(output,dtype='<f4');sf.write(output,audio,48000,subtype='FLOAT',format='WAV')
        else:audio,sr=sf.read(output,dtype='float32');assert sr==48000
        assert len(audio)>0 and np.isfinite(audio).all() and np.sqrt(np.mean(audio**2))>1e-8
        if overrides.get('seconds'):assert len(audio)==round(overrides['seconds']*48000)
        record(case['id']+':'+name,ok=True,frames=result['frames'],samples=len(audio))
        return result,audio,output
    with (ROOT/'artifacts/gpu.lock').open('a') as lock:
        if lock_gpu:fcntl.flock(lock,fcntl.LOCK_EX)
        for profile in ['fp32','practical']:
            for case in cases(profile):
                cmd,env=command(case);worker=Process(cmd,env,dest/(case['id']+'.log'))
                try:
                    a,_,_=synth(worker,case,'no-ref-auto');frames[case['id']]=a['frames']
                    _,audio,_=synth(worker,case,'no-ref-fixed',seconds=1.,noise=str(noise))
                    if case['engine']=='official' and profile=='fp32':
                        np.save(dest/(case['variant']+'-official-no-ref.npy'),audio)
                    if case['engine']=='onnx' and profile=='fp32':
                        ref=np.load(dest/'mf-official-no-ref.npy')
                        corr=float(np.corrcoef(ref,audio)[0,1])
                        if corr<.98:raise ValueError(f'ONNX no-ref FP32 common-noise correlation too low: {corr}')
                        record('onnx-no-ref-fp32-waveform',ok=True,correlation=corr)
                    _,audio,path=synth(worker,case,'reference-fixed',seconds=1.,noise=str(noise),
                        references=[str(ROOT/'artifacts/benchmark/inputs/reference.wav')])
                    # Exercise a second seed and repeated reference registration independently of request order.
                    synth(worker,case,'reference-repeat',seconds=1.,
                        seed=43,references=[str(ROOT/'artifacts/benchmark/inputs/reference.wav')])
                    if case['engine']=='onnx':
                        _,new,_=synth(worker,case,'reference-auto',caption='自然な日本語で、落ち着いた口調で話す。',
                            references=[str(ROOT/'artifacts/benchmark/inputs/reference.wav')])
                        baseline=Process([str(ROOT/'artifacts/build/irodori-onnx/release/examples/comparison_bench'),
                            str(ROOT),case['precision']],env,dest/(case['id']+'-baseline.log'),prefix='BENCH ')
                        try:
                            old_path=dest/(case['id']+'-baseline.f32')
                            old=baseline.request(dict(id='baseline',text='今日はよく晴れています。',
                                cache='native_cache',output=str(old_path)))
                            if old['type']!='result':raise RuntimeError(str(old))
                            old_audio=np.fromfile(old_path,dtype='<f4')
                            if new.shape!=old_audio.shape or not np.allclose(new,old_audio,atol=1e-6,rtol=1e-5):
                                raise ValueError('ONNX reference-conditioned output changed')
                            record(case['id']+':reference-unchanged',ok=True,
                                max_abs=float(np.max(np.abs(new-old_audio))),bit_identical=bool(np.array_equal(new,old_audio)))
                        finally:baseline.close()
                finally:worker.close()
        if abs(frames['official-mf-fp32']-frames['onnx-mf-fp32'])>1:
            raise ValueError('ONNX no-reference duration differs from official FP32 by more than one frame')
        record('onnx-no-reference-duration',ok=True,official_frames=frames['official-mf-fp32'],onnx_frames=frames['onnx-mf-fp32'])
        wav=dest/'official-mf-fp32-reference-fixed.wav'
        for name in ['kana','whisper','campplus','eres2net','qwen','reazon']:
            worker=Process([str(PYTHON),str(HERE/'score_worker.py'),name],environment(),dest/f'score-{name}.log')
            try:
                out=worker.request(dict(id='scorer-smoke',wav=str(wav)))
                if out['type']!='result':raise RuntimeError(str(out))
                record('scorer:'+name,ok=True,transcript=out.get('transcript'),dimensions=len(out.get('embedding',[])))
            finally:worker.close()
    record('complete',ok=True)


if __name__=='__main__':main()
