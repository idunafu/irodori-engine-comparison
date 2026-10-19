"""Bounded, detached reference-path ablation; original campaign outputs stay immutable."""
from collections import defaultdict
import fcntl
import json
import math
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import soundfile as sf

OUT=Path(__file__).resolve().parent;RUN=OUT.parents[1];ROOT=OUT.parents[5]
sys.path.insert(0,str(ROOT/'benchmark/quality'))
from common import HERE,PYTHON,atomic_json,digest,environment,sha256,utc
from worker import Process,command
MODES=['official','speech_native','speech_official_pcm','speech_official_latent','speech_official_state']


def main():
    state=dict(status='running',pid=__import__('os').getpid(),stage='waiting-for-gpu-lock',done=0,total=0,started_at=utc())
    worker=None;stop=threading.Event()
    def save():atomic_json(OUT/'status.json',dict(state,heartbeat_at=utc()))
    def heartbeat():
        while not stop.wait(30):save()
    def terminate(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();save()
    def stage(name,total):state.update(stage=name,done=0,total=total);save();print(utc(),name,total,flush=True)
    def request(req):
        result=worker.request(req)
        if result.get('type')!='result':raise RuntimeError(result)
        state['done']+=1;state['last_progress_at']=utc()
        return result
    speakers=json.loads((RUN/'splits.json').read_text())['diagnostic_jvs']
    index={s['id']:s for s in json.loads((ROOT/'artifacts/quality/assets/jvs-index.json').read_text())}
    c=sqlite3.connect(f'file:{RUN}/results.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
    targets={}
    for row in c.execute('SELECT * FROM tasks WHERE case_id IN ("official-mf-fp32","official-rf-fp32") AND json_extract(spec,"$.item.kind")="one"'):
        spec=json.loads(row['spec'])
        if spec['case']['variant']=='rf' and not spec['item']['id'].endswith(':'+index[spec['item']['speaker']]['parallel'][0]['id']):continue
        assert row['status']=='ok'
        targets[spec['case']['variant'],spec['item']['id'],spec['seed']]=dict(spec=spec,frames=json.loads(row['result'])['frames'])
    c.close()
    assert sum(k[0]=='mf' for k in targets)==500 and sum(k[0]=='rf' for k in targets)==100
    def paths(variant,sp,mode):
        d=OUT/'features'/variant/sp;d.mkdir(parents=True,exist_ok=True)
        return dict(id=f'{variant}:{sp}:{mode}',op='dump',reference=str(ROOT/index[sp]['references'][0]['path']),
                    pcm_out=str(d/(mode+'-pcm.f32')),resampled_out=str(d/(mode+'-resampled.f32')),
                    latent_out=str(d/(mode+'-latent.f32')),latent_pt=str(d/(mode+'-latent.pt')),state_out=str(d/(mode+'-state.f32')))
    records=[]
    def append(path,row):
        with path.open('a',buffering=1) as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();__import__('os').fsync(f.fileno())
    try:
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            atomic_json(OUT/'manifest.json',dict(speakers=speakers,seeds=list(range(5)),precision='fp32',
                        mf_targets_per_speaker=5,rf_targets_per_speaker=1,modes=MODES,
                        source_manifest_sha256=sha256(RUN/'manifest.json'),binary_sha256=sha256(OUT/'speech-diagnostic'),
                        original_outputs_modified=False,selection='Existing fixed 20-speaker diagnostic; RF uses first fixed text only'))
            stage('reference-dumps',160)
            for variant in ['mf','rf']:
                worker=Process([str(ROOT/'.venvs/Irodori-TTS/bin/python'),str(OUT/'official.py'),variant],environment(),OUT/f'official-{variant}.log')
                for sp in speakers:
                    req=paths(variant,sp,'official');out=request(req);records.append(dict(**req,result=out))
                    atomic_json(OUT/'dump-results.json',records)
                worker.close();worker=None
                native,_=command(dict(engine='speech',variant=variant,precision='fp32'))
                worker=Process([str(OUT/'speech-diagnostic'),native[1],native[2]],environment(),OUT/f'speech-{variant}.log')
                for sp in speakers:
                    official=paths(variant,sp,'official')
                    for mode in ['native','official_pcm','official_latent']:
                        req=paths(variant,sp,'speech_'+mode)
                        if mode=='official_pcm':req['pcm_in']=official['pcm_out']
                        if mode=='official_latent':req['latent_in']=official['latent_out']
                        out=request(req);records.append(dict(**req,result=out));atomic_json(OUT/'dump-results.json',records)
                worker.close();worker=None
            stage('common-noise-generation',3000)
            for variant in ['mf','rf']:
                selected=[(k,t) for k,t in sorted(targets.items()) if k[0]==variant]
                for mode in MODES:
                    if mode=='official':cmd=[str(ROOT/'.venvs/Irodori-TTS/bin/python'),str(OUT/'official.py'),variant]
                    else:
                        native,_=command(dict(engine='speech',variant=variant,precision='fp32'));cmd=[str(OUT/'speech-diagnostic'),native[1],native[2]]
                    worker=Process(cmd,environment(),OUT/f'{mode}-{variant}-generation.log')
                    for key,t in selected:
                        s=t['spec'];sp=s['item']['speaker'];frames=t['frames']
                        ident=digest([variant,s['item']['id'],s['seed']])[:24]
                        noise=OUT/'noise'/(ident+'.f32');noise.parent.mkdir(exist_ok=True)
                        if not noise.exists():np.random.default_rng(int(ident[:16],16)).standard_normal((frames,32),dtype=np.float32).tofile(noise)
                        wav=OUT/'audio'/variant/mode/(ident+'.wav');wav.parent.mkdir(parents=True,exist_ok=True)
                        own=paths(variant,sp,'speech_native');official=paths(variant,sp,'official')
                        req=dict(id=f'{variant}:{mode}:{ident}',op='synthesize',text=s['item']['text'],seed=s['seed'],
                                 seconds=float(np.nextafter(frames/25,np.inf)),noise=str(noise),output=str(wav),latent_pt=official['latent_pt'])
                        if mode=='speech_native':req['latent_in']=own['latent_out']
                        elif mode=='speech_official_pcm':req['latent_in']=paths(variant,sp,'speech_official_pcm')['latent_out']
                        elif mode=='speech_official_latent':req['latent_in']=official['latent_out']
                        elif mode=='speech_official_state':req['state_in']=official['state_out']
                        out=request(req)
                        if out.get('raw_float'):
                            a=np.fromfile(wav,dtype='<f4');sf.write(wav,a,48000,subtype='FLOAT',format='WAV')
                        info=sf.info(wav);assert info.frames==frames*1920 and info.samplerate==48000
                        append(OUT/'generation.jsonl',dict(id=req['id'],variant=variant,mode=mode,speaker=sp,item=s['item']['id'],seed=s['seed'],
                                                         wav=str(wav),wav_sha256=sha256(wav),frames=frames,noise_sha256=sha256(noise)))
                    worker.close();worker=None
            rows=[json.loads(line) for line in (OUT/'generation.jsonl').read_text().splitlines()]
            assert len(rows)==3000 and len({r['id'] for r in rows})==3000
            stage('campplus-scoring',len(rows))
            worker=Process([str(PYTHON),str(HERE/'score_worker.py'),'campplus'],environment(),OUT/'campplus.log')
            centroids=json.loads((RUN/'speaker-centroids.json').read_text())['centroids'];ids=sorted(centroids);mat=np.array([centroids[s] for s in ids])
            for row in rows:
                assert sha256(Path(row['wav']))==row['wav_sha256']
                out=request(dict(id=row['id'],wav=row['wav']));sim=mat@np.array(out['embedding'])
                append(OUT/'scores.jsonl',dict(**row,cosine=float(sim[ids.index(row['speaker'])]),top1=int(ids[int(np.argmax(sim))]==row['speaker'])))
            worker.close();worker=None
        stage('analysis',0)
        subprocess.run([str(PYTHON),str(OUT/'analyze.py')],cwd=ROOT,check=True)
        state.update(status='complete',stage='complete',exit_code=0,finished_at=utc());save()
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}',exit_code=1,finished_at=utc());save();traceback.print_exc();raise
    finally:
        if worker:worker.close()
        stop.set();thread.join(timeout=2)


if __name__=='__main__':main()
