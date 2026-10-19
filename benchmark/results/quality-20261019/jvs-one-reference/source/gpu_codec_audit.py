"""Serial Vulkan codec kernel-route checks, after the shared GPU lock is free."""
import fcntl
import json
from pathlib import Path
import sys
import signal
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[5]
sys.path.insert(0,str(ROOT/'benchmark/quality'))
from common import atomic_json,environment,utc
from worker import Process,command

def main():
    state=dict(status='running',stage='waiting-for-gpu-lock',started_at=utc(),done=0,total=60)
    def save():atomic_json(OUT/'gpu-codec-status.json',dict(state,updated_at=utc()))
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save();worker=None;rows=[]
    try:
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state['stage']='codec-kernel-check';save()
            cmd,_=command(dict(engine='speech',variant='mf',precision='fp32'))
            for mode,flags in [('default',{}),('no_coopmat',{'GGML_VK_DISABLE_COOPMAT':'1','GGML_VK_DISABLE_COOPMAT2':'1'}),
                               ('no_f16',{'GGML_VK_DISABLE_COOPMAT':'1','GGML_VK_DISABLE_COOPMAT2':'1','GGML_VK_DISABLE_F16':'1'})]:
                env=environment();env.update(flags)
                worker=Process([str(OUT/'codec-audit'),cmd[1],'gpu'],env,OUT/('codec-'+mode+'.log'))
                for sp in json.loads((OUT/'manifest.json').read_text())['speakers']:
                    d=OUT/'features/mf'/sp
                    req=dict(id=mode+':'+sp,pcm=str(d/'official-pcm.f32'),windowed=str(d/(mode+'-gpu-windowed.f32')),whole=str(d/(mode+'-gpu-whole.f32')))
                    result=worker.request(req)
                    if result.get('type')!='result':raise RuntimeError(result)
                    a=np.fromfile(d/'official-latent.f32',dtype='<f4').astype(float)
                    b=np.fromfile(req['windowed'],dtype='<f4').astype(float)
                    c=np.fromfile(req['whole'],dtype='<f4').astype(float)
                    rows.append(dict(**req,mode=mode,speaker=sp,flags=flags,result=result,
                                     relative_l2=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                                     whole_relative_l2=float(np.linalg.norm(c-a)/np.linalg.norm(a)),
                                     window_vs_whole=float(np.linalg.norm(b-c)/np.linalg.norm(c))))
                    atomic_json(OUT/'gpu-codec-audit.json',rows);state['done']+=1;save()
                worker.close();worker=None
        state.update(status='complete',exit_code=0,finished_at=utc());save()
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}',exit_code=1,finished_at=utc());save();raise
    finally:
        if worker:worker.close()

if __name__=='__main__':main()
