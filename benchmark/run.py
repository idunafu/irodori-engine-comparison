"""Serial, resumable GPU benchmark with resident workers and NVML monitoring."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import threading
import time
from contextlib import ExitStack

import numpy as np
import pynvml as nv
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmark'
CFG=json.loads((ROOT/'benchmark/config.json').read_text())
PROMPTS=json.loads((ROOT/'benchmark/prompts.json').read_text())

class Monitor:
    def __init__(self):
        nv.nvmlInit();self.dev=nv.nvmlDeviceGetHandleByIndex(0);self.pid=None
        self.lock=threading.Lock();self.running=True;self.points=[];self.last={};self.baseline=nv.nvmlDeviceGetMemoryInfo(self.dev).used
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        while self.running:
            t=time.perf_counter()
            try:
                total=nv.nvmlDeviceGetMemoryInfo(self.dev).used;memory=[]
                for query in (nv.nvmlDeviceGetComputeRunningProcesses,nv.nvmlDeviceGetGraphicsRunningProcesses):
                    try:
                        for p in query(self.dev):
                            if p.pid==self.pid and isinstance(p.usedGpuMemory,int) and p.usedGpuMemory < 1<<60:memory.append(p.usedGpuMemory)
                    except nv.NVMLError:pass
                point=dict(at=t,vram=max(memory) if memory else max(0,total-self.baseline),
                           method='nvml_process' if memory else 'nvml_device_delta',gpu_used=total,
                           temperature=nv.nvmlDeviceGetTemperature(self.dev,nv.NVML_TEMPERATURE_GPU),
                           sm_clock=nv.nvmlDeviceGetClockInfo(self.dev,nv.NVML_CLOCK_SM))
                with self.lock:self.points.append(point);self.last=point
            except nv.NVMLError:pass
            time.sleep(max(0,CFG['vram_interval_ms']/1000-(time.perf_counter()-t)))
    def reset(self):
        with self.lock:self.points=[]
    def snapshot(self):
        with self.lock:
            pts=list(self.points);last=dict(self.last)
        return dict(resident_vram=last.get('vram'),peak_vram=max((p['vram'] for p in pts),default=last.get('vram',0)),
                    vram_method=last.get('method'),nvml_samples=len(pts),telemetry=last)
    def close(self):self.running=False;self.thread.join();nv.nvmlShutdown()

def matrix():
    rows=[]
    for engine,variants,backends,precision in (
        ('Irodori-TTS',['rf','mf'],['cuda'],'bf16'),('Irodori-TTS-Optimiz',['rf'],['cuda'],'bf16'),
        ('audio.cpp',['rf'],['cuda','vulkan'],'f16'),('speech.cpp',['rf','mf'],['cuda','vulkan'],'f16'),
        ('sttts-gui',['rf','mf'],['vulkan'],'fp32'),('Irodori-TTS-burn',['rf','mf'],['vulkan'],'fp32'),
        ('irodori-onnx',['mf'],['cuda'],'fp16')):
        for variant in variants:
            for backend in backends:
                for profile in ('fp32','practical'):
                    if precision=='fp32' and profile=='practical':continue
                    p='fp32' if profile=='fp32' else precision
                    rows.append(dict(engine=engine,variant=variant,backend=backend,precision=p,precision_profile=profile,
                        shared_profiles=['fp32','practical'] if precision=='fp32' else [profile],
                        case=f'{engine}-{variant}-{backend}-{profile}'))
    return rows

def command(case):
    e,v,b,p=case['engine'],case['variant'],case['backend'],case['precision']
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',RAYON_NUM_THREADS='8',
             HF_HUB_OFFLINE='1',WGPU_BACKEND='vulkan',WGPU_BACKENDS='vulkan',NVIDIA_TF32_OVERRIDE='0',
             TOKENIZERS_PARALLELISM='false')
    if e in ('Irodori-TTS','Irodori-TTS-Optimiz'):
        cmd=[ROOT/f'.venvs/{e}/bin/python',ROOT/'benchmark/python_worker.py',e,v,p]
    elif e=='speech.cpp':
        model=next((OUT/f'models/speech-{v}').glob('*.gguf')) if p=='fp32' else OUT/f'models/speech-{v}-f16.gguf'
        cmd=[OUT/f'build/{e}/{b}/comparison_bench',model,OUT/'inputs/reference.wav',str(CFG['steps'][v])]
    elif e=='audio.cpp':cmd=[OUT/f'build/{e}/{b}/bin/comparison_bench',OUT/f'models/audio-rf-{"orig" if p=="fp32" else "f16"}.gguf',OUT/'inputs/reference.wav',b]
    else:
        cmd=[ROOT/f'artifacts/build/{e}/release/examples/comparison_bench',ROOT,v] if e!='irodori-onnx' else [ROOT/f'artifacts/build/{e}/release/examples/comparison_bench',ROOT,p]
        if e=='irodori-onnx':
            site=ROOT/'.venvs/onnx/lib/python3.10/site-packages'
            libs=[str(x) for x in (site/'nvidia').glob('*/lib')]+[str(site/'onnxruntime/capi'),env.get('LD_LIBRARY_PATH','')]
            env['LD_LIBRARY_PATH']=':'.join(libs);env['IRODORI_ORT_DLL']=str(site/'onnxruntime/capi/libonnxruntime.so.1.23.2')
    return list(map(str,cmd)),env

class Worker:
    def __init__(self,case,monitor):
        self.case=case;self.monitor=monitor;self.queue=queue.Queue();self.timeout=CFG['timeout_seconds']
        cmd,env=command(case);self.log=open(ROOT/f'logs/benchmark-{case["case"]}.log','a')
        monitor.reset();t=time.perf_counter()
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,env=env,start_new_session=True,cwd=ROOT)
        monitor.pid=self.proc.pid
        threading.Thread(target=self.read,daemon=True).start()
        try:self.ready=self.receive()
        except BaseException:self.close();raise
        self.ready.update(startup_wall_ms=(time.perf_counter()-t)*1000,**monitor.snapshot())
        if self.ready.get('type')!='ready':self.close();raise RuntimeError(str(self.ready))
    def read(self):
        for line in self.proc.stdout:
            if line.startswith('BENCH '):
                try:self.queue.put(json.loads(line[6:]))
                except json.JSONDecodeError:self.queue.put(dict(type='error',error=line[:300]))
        self.queue.put(dict(type='error',error=f'Worker exited ({self.proc.poll()})'))
    def receive(self):
        try:return self.queue.get(timeout=self.timeout)
        except queue.Empty:raise TimeoutError('Worker request timed out')
    def request(self,req):
        e=self.case['engine']; payload=dict(req)
        if e=='speech.cpp':
            payload='\t'.join([req['id'],'1' if req.get('profile') else '0',req['cache'],str(req.get('seconds') or 0),req['text'],req.get('output','-'),'end'])
        else:
            if e=='audio.cpp':
                payload.update(caption=CFG['caption'],ref_wav=str(OUT/'inputs/reference.wav'),num_steps=40,seed=42,
                    cfg_scale_text=3,cfg_scale_speaker=5,cfg_scale_caption=3,cfg_guidance_mode='independent',cfg_min_t=.5,cfg_max_t=1.,trim_tail=False)
            payload=json.dumps(payload,ensure_ascii=False)
        self.monitor.reset();t=time.perf_counter();self.proc.stdin.write(payload+'\n');self.proc.stdin.flush()
        out=self.receive();out.update(request_wall_ms=(time.perf_counter()-t)*1000,**self.monitor.snapshot())
        if out.get('type')!='result':raise RuntimeError(str(out))
        if not out.get('finite') or out.get('rms',0)<=1e-8:raise RuntimeError('Invalid or silent PCM')
        if req.get('seconds'):
            expected=int(round(req['seconds']*48000))
            if abs(out['samples']-expected)>1:raise RuntimeError(f'Output length mismatch: {out["samples"]} != {expected}')
        out['rtf']=out['ms']/1000/(out['samples']/out['sample_rate'])
        return out
    def close(self):
        if self.proc.poll() is None:
            try:self.proc.stdin.write(('stop' if self.case['engine']=='speech.cpp' else '{"type":"stop"}')+'\n');self.proc.stdin.flush();self.proc.wait(timeout=30)
            except Exception:
                os.killpg(self.proc.pid,signal.SIGTERM)
                try:self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
        self.monitor.pid=None;self.log.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def append(path,record):
    with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--engines',nargs='+');parser.add_argument('--cases',nargs='+');parser.add_argument('--preflight',action='store_true');parser.add_argument('--force-preflight',action='store_true');parser.add_argument('--calibrate',action='store_true');parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--refresh-components',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'wav').mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
    signature={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ('benchmark/config.json','benchmark/prompts.json','models.lock.json','engines.lock.json')}
    signature['reference']=json.loads((OUT/'inputs/reference.json').read_text())['prepared_sha256']
    signature_path=OUT/'inputs/run-signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        raise RuntimeError('Benchmark inputs changed: archive the existing run before starting a different dataset')
    signature_path.write_text(json.dumps(signature,indent=2)+'\n')
    result_path=OUT/('preflight.jsonl' if args.preflight else 'results.jsonl')
    done={x['key'] for x in map(json.loads,result_path.read_text().splitlines()) if x.get('status')=='ok' and
          (x.get('kind')!='profile' or max(x.get('instrumentation_version',0),x.get('result',{}).get('instrumentation_version',0)) >=
           (4 if args.refresh_components else (3 if x['case']['engine'] in ('Irodori-TTS','Irodori-TTS-Optimiz') else 2)))} if result_path.exists() else set()
    if result_path.exists() and not args.retry_failed:
        done.update(x['key'] for x in map(json.loads,result_path.read_text().splitlines()) if x.get('status')=='failed' and x.get('mode'))
    rows=[x for x in matrix() if (not args.engines or x['engine'] in args.engines) and (not args.cases or x['case'] in args.cases)]
    with (ROOT/'artifacts/gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);monitor=Monitor()
        try:
            calibration=OUT/'inputs/durations.json'
            if not calibration.exists() or args.calibrate:
                case=matrix()[0];durations={}
                with Worker(case,monitor) as worker:
                    for prompt in PROMPTS:
                        out=worker.request(dict(id=prompt['id'],text=prompt['text'],cache='native_cache',profile=False))
                        durations[prompt['id']]=out['frames'];print('duration',prompt['id'],out['frames'],flush=True)
                calibration.write_text(json.dumps(durations,indent=2)+'\n')
            durations=json.loads(calibration.read_text())
            for case in rows:
                print('CASE',case['case'],flush=True)
                try:
                    if case['engine']=='audio.cpp':
                        stamps=[OUT/f'source/audio.cpp/.bench-built-{b}.json' for b in ('cuda','vulkan')]
                        if not all(s.exists() and json.loads(s.read_text()).get('text_chunk_size')==4096 for s in stamps):
                            for backend in ('cuda','vulkan'):
                                subprocess.run([str(ROOT/'benchmark/.venv/bin/python'),str(ROOT/'benchmark/build.py'),'audio.cpp','--backend',backend],check=True)
                    if case['engine']=='irodori-onnx':
                        stamp=OUT/'source/irodori-onnx/.bench-built-rust.json'
                        if not stamp.exists() or json.loads(stamp.read_text()).get('ort_intra_threads')!=8:
                            # Build between cases, while holding the GPU lock and before any timed worker.
                            subprocess.run([str(ROOT/'benchmark/.venv/bin/python'),str(ROOT/'benchmark/build.py'),'irodori-onnx'],check=True)
                            for checked in [r for r in matrix() if r['engine']=='irodori-onnx']:
                                with Worker(checked,monitor) as check_worker:
                                    outputs=[];arrays=[]
                                    for profiled in (False,True):
                                        path=OUT/'raw'/f'{checked["case"]}-{profiled}.raw'
                                        outputs.append(check_worker.request(dict(id='preflight',text=PROMPTS[0]['text'],seconds=2.,cache='reencode',profile=profiled,output=str(path))))
                                        arrays.append(np.fromfile(path,dtype='<f4'))
                                    relative=float(np.linalg.norm(arrays[0]-arrays[1])/max(np.linalg.norm(arrays[0]),1e-12))
                                    if relative>(1e-4 if checked['precision']=='fp32' else .01):raise RuntimeError('ONNX thread configuration changed profiling parity')
                                    append(OUT/'preflight.jsonl',dict(key=checked['case']+'/preflight',status='ok',case=checked,load=check_worker.ready,outputs=outputs,profile_relative_error=relative))
                    key=case['case']+'/preflight'
                    if args.preflight and key in done and not args.force_preflight:continue
                    if args.preflight:
                        with Worker(case,monitor) as worker:
                            outputs=[]
                            for profile in (False,True):
                                suffix='.wav' if case['engine'].startswith('Irodori-TTS') and case['backend']=='cuda' else '.raw'
                                path=OUT/'raw'/f'{case["case"]}-{profile}{suffix}'
                                req=dict(id='preflight',text=PROMPTS[0]['text'],seconds=2.,cache='reencode',profile=profile,output=str(path))
                                out=worker.request(req);outputs.append(out)
                            arrays=[]
                            for profile in (False,True):
                                path=OUT/'raw'/f'{case["case"]}-{profile}{suffix}'
                                arrays.append(sf.read(path,dtype='float32')[0] if suffix=='.wav' else np.fromfile(path,dtype='<f4'))
                            rel=float(np.linalg.norm(arrays[0]-arrays[1])/max(np.linalg.norm(arrays[0]),1e-12))
                            if rel>(1e-4 if case['precision']=='fp32' else .01):raise RuntimeError(f'Profiling changed PCM: relative error {rel}')
                            forwards=[e for e in outputs[1].get('events',[]) if e['stage']=='dit_forward']
                            # Independent CFG may use several B1 forwards or one batched forward per sampler step.
                            expected=CFG['steps'][case['variant']]
                            if len(forwards)<expected:raise RuntimeError(f'DiT calls {len(forwards)} < sampler steps {expected}')
                            sf.write(OUT/'wav'/f'{case["case"]}-preflight.wav',arrays[0],48000,subtype='FLOAT')
                            append(result_path,dict(key=key,status='ok',case=case,load=worker.ready,outputs=outputs,profile_relative_error=rel))
                        continue
                    cold_prompt=PROMPTS[8]
                    for rep in range(CFG['cold_repeats']):
                        key=f'{case["case"]}/cold/{rep}'
                        if key in done:continue
                        with Worker(case,monitor) as worker:
                            out=worker.request(dict(id=key,text=cold_prompt['text'],seconds=durations[cold_prompt['id']]/25,cache='reencode',profile=False))
                            append(result_path,dict(key=key,status='ok',case=case,kind='cold',load=worker.ready,result=out))
                    with ExitStack() as resources:
                        worker=resources.enter_context(Worker(case,monitor))
                        for cache in CFG['cache_profiles']:
                            work=[(mode,p, None if mode=='auto' else (8. if mode=='text_scaling' else durations[p['id']]/25))
                                  for mode in ('matched','auto','text_scaling') for p in PROMPTS]
                            work += [('audio_scaling',PROMPTS[8],float(sec)) for sec in CFG['audio_seconds']]
                            for mode,p,seconds in work:
                                group=f'{case["case"]}/{cache}/{mode}/{p["id"]}/{seconds}'
                                if group+'/failure' in done:continue
                                if all(f'{group}/wall/{i}' in done for i in range(CFG['repeats'])) and all(f'{group}/profile/{i}' in done for i in range(CFG['repeats'])):continue
                                req=dict(text=p['text'],seconds=seconds,cache=cache,profile=False,id=group)
                                # First visit of each shape is retained, separate from steady-state measurements.
                                try:
                                    for i in range(CFG['warmup']):
                                        out=worker.request(req)
                                        if i==0:append(result_path,dict(key=group+'/shape_first',status='ok',case=case,kind='shape_first',mode=mode,cache=cache,prompt=p,result=out))
                                    monitor.reset();time.sleep(.2)
                                    append(result_path,dict(key=group+'/idle',status='ok',case=case,kind='idle',mode=mode,cache=cache,prompt=p,result=monitor.snapshot()))
                                    for profile in (False,True):
                                        for rep in range(CFG['repeats']):
                                            kind='profile' if profile else 'wall';key=f'{group}/{kind}/{rep}'
                                            if key in done:continue
                                            req.update(id=key,profile=profile)
                                            out=worker.request(req)
                                            append(result_path,dict(key=key,status='ok',instrumentation_version=4,case=case,kind=kind,mode=mode,cache=cache,prompt=p,seconds=seconds,result=out))
                                    print('DONE',group,flush=True)
                                except Exception as e:
                                    append(result_path,dict(key=group+'/failure',status='failed',case=case,mode=mode,cache=cache,prompt=p,error=str(e)))
                                    print('GROUP FAILED',group,str(e),flush=True)
                                    worker.close()
                                    worker=resources.enter_context(Worker(case,monitor))
                except Exception as e:
                    append(result_path,dict(key=case['case']+'/failure',status='failed',case=case,error=str(e)))
                    print('FAILED',case['case'],str(e),flush=True)
        finally:monitor.close()

if __name__=='__main__':main()
