"""Detached, serial quality campaign. All progress is durable and externally inspectable."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import soundfile as sf
from common import ASSETS, BASE, CONFIG, DATASETS, HERE, PYTHON, ROOT, atomic_json, digest, environment, failure_category, sha256, utc
from dataset import inputs, tasks
from store import Store
from worker import Process, command


def signature(datasets=None):
    selected=sorted(DATASETS if datasets is None else datasets)
    paths = [ROOT/'engines.lock.json',ROOT/'models.lock.json',ROOT/'benchmark/prompts.json',
             ASSETS/'models.lock.json',ASSETS/'requirements-resolved.txt',
             ASSETS/'runtime-packages.json',ROOT/'artifacts/benchmark/provenance.json',
             BASE/'build/manifest.json',ROOT/'artifacts/benchmark/inputs/reference.wav']
    if 'jvs' in selected:
        paths += [ASSETS/'jvs-index.json',ASSETS/'download-manifest.json']
    paths += [p for p in HERE.iterdir() if p.suffix in ('.py','.cpp','.rs','.mjs','.json','.sh')]
    paths += [ROOT/f'models/{v}/model.safetensors' for v in ['rf','mf']]
    paths += [p for v in ['rf','mf'] for p in (ROOT/f'models/{v}/tokenizer').rglob('*') if p.is_file()]
    paths += [ROOT/'models/codec/weights.pth']
    paths += [BASE/'build/speech-quality',BASE/'build/onnx-quality',
              ROOT/'artifacts/benchmark/build/speech.cpp/vulkan/speech']
    paths += list((ASSETS/'jkyb-evaluator/src').rglob('*.py'))
    paths += list((ASSETS/'speech-bench/src').rglob('*.ts'))
    paths += [ASSETS/'jkyb/data/common_kanji_source.jsonl',ASSETS/'speech-bench/prompts/speak-ja-JP.json']
    paths += list((ROOT/'artifacts/onnx/export/ref').glob('*.onnx*'))
    paths += list((ROOT/'artifacts/benchmark/models').glob('speech-*/*.gguf'))
    paths += list((ROOT/'artifacts/benchmark/models').glob('speech-*-f16.gguf'))
    files={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}
    speakers=json.loads((ASSETS/'jvs-index.json').read_text()) if 'jvs' in selected else []
    for speaker in speakers:
        for clip in speaker['references']+speaker['parallel'][:15]:
            actual=sha256(ROOT/clip['path'])
            if actual!=clip['sha256']:
                raise ValueError('JVS source recording changed: '+clip['path'])
            files[clip['path']]=actual
    for engine, expected in json.loads((ROOT/'engines.lock.json').read_text()).items():
        if engine in ('Irodori-TTS','irodori-onnx','speech.cpp'):
            actual=subprocess.check_output(['git','-C',str(ROOT/'engines'/engine),'rev-parse','HEAD'],text=True).strip()
            if actual != expected['commit']:
                raise ValueError(f'Engine revision mismatch: {engine}')
    return dict(format=1,datasets=selected,files=files,fingerprint=digest(dict(datasets=selected,files=files)))


def audio_metrics(audio):
    if len(audio)==0:
        raise ValueError('Empty audio')
    if not np.isfinite(audio).all():
        raise ValueError('Non-finite audio')
    rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    if rms<=1e-8:
        raise ValueError('Silent audio')
    levels=np.sqrt(np.mean(audio[:len(audio)//960*960].reshape(-1,960).astype(np.float64)**2,axis=1))
    active=np.flatnonzero(levels>max(1e-5,float(levels.max())*.01))
    return dict(seconds=len(audio)/48000,rms=rms,peak=float(np.max(np.abs(audio))),
                voiced_seconds=float(np.count_nonzero(levels>=float(levels.max())*.1))*.02,
                clipping_fraction=float(np.mean(np.abs(audio)>=1)),
                leading_silence_seconds=(int(active[0])*.02 if len(active) else len(audio)/48000),
                trailing_silence_seconds=((len(levels)-1-int(active[-1]))*.02 if len(active) else len(audio)/48000))


def infrastructure(error):
    text=str(error).lower()
    return any(x in text for x in ['out of memory','timeout','worker exited','broken pipe',
                                   'cuda error','device lost','startup failed','response id mismatch'])


def ensure_manifest(directory,current):
    path=Path(directory)/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=current:
            raise ValueError('Inputs, environment, code or model hashes changed; refusing resume')
    else:
        atomic_json(path,current)


class Campaign:
    def __init__(self,directory,retry_failed=False):
        self.directory=directory
        self.store=Store(directory)
        self.worker=None
        self.retry_failed=retry_failed
        self.state=dict(status='running',pid=os.getpid(),started=utc(),stage='initializing',
                        phase_total=0,phase_done=0,phase_failed=0,last_progress_at=utc())
        self.finished=threading.Event()
        self.write_lock=threading.Lock()
        self.phase_start=time.monotonic()
        self.thread=threading.Thread(target=self.heartbeat,daemon=True)
        self.thread.start()

    def save(self):
        with self.write_lock:
            state=dict(self.state,heartbeat_at=utc())
            done=state['phase_done']
            state['estimated_remaining_seconds']=((time.monotonic()-self.phase_start)/done*
                max(0,state['phase_total']-done) if done else None)
            atomic_json(self.directory/'status.json',state)

    def heartbeat(self):
        while not self.finished.wait(CONFIG['heartbeat_seconds']):
            self.save()

    def stage(self,name,total=0):
        self.phase_start=time.monotonic()
        self.state.update(stage=name,phase_total=total,phase_done=0,phase_failed=0)
        self.save()
        print(utc(),name,'pending',total,flush=True)

    def progress(self,failed=False):
        self.state['phase_done']+=1
        self.state['phase_failed']+=int(failed)
        self.state['last_progress_at']=utc()
        if self.state['phase_done']%100==0:
            print(self.state['stage'],self.state['phase_done'],'/',self.state['phase_total'],flush=True)

    def close_worker(self):
        if self.worker:
            self.worker.close()
            self.worker=None

    def generate(self,where,label):
        db=self.store.db
        statuses="('pending','failed')" if self.retry_failed else "('pending')"
        selection=f"({where}) AND status IN {statuses}"
        self.stage(label,db.execute(f'SELECT COUNT(*) FROM tasks WHERE {selection}').fetchone()[0])
        case_ids=[x[0] for x in db.execute(f'SELECT DISTINCT case_id FROM tasks WHERE {selection} ORDER BY case_id')]
        for case_id in case_ids:
            errors=0
            # Do not retain a cursor across commits or load the whole campaign into memory.
            cursor=db.execute(f'SELECT * FROM tasks WHERE {selection} AND case_id=? ORDER BY rowid',(case_id,))
            for row in cursor:
                spec=json.loads(row['spec']); item=spec['item']; key=row['id']
                wav=self.directory/'wav'/case_id/(key+'.wav')
                wav.parent.mkdir(parents=True,exist_ok=True)
                temp=wav.with_suffix('.tmp.wav')
                try:
                    if item.get('reference_missing'):
                        raise ValueError('Insufficient reference duration')
                    req=dict(id=key,text=item['text'],references=item['references'],
                             caption=item['caption'],seed=spec['seed'],output=str(temp))
                    if spec['mode']=='common':
                        source=db.execute('SELECT status,result FROM tasks WHERE id=?',(spec['length_source'],)).fetchone()
                        if not source or source['status']!='ok':
                            raise ValueError('Official FP32 duration oracle failed')
                        frames=json.loads(source['result'])['frames']
                        req['seconds']=frames/25
                        noise=self.directory/'noise'/(digest([item['id'],spec['case']['variant'],spec['seed']])+'.f32')
                        if not noise.exists():
                            noise.parent.mkdir(exist_ok=True)
                            seed=int(digest([item['id'],spec['seed']])[:16],16)
                            np.random.default_rng(seed).standard_normal((frames,32),dtype=np.float32).astype('<f4').tofile(noise)
                        req['noise']=str(noise)
                    if self.worker is None:
                        cmd,env=command(spec['case'])
                        self.worker=Process(cmd,env,self.directory/f'{case_id}.log')
                    out=self.worker.request(req)
                    if out.get('type')!='result':
                        raise RuntimeError(out.get('error',str(out)))
                    if out.get('raw_float'):
                        audio=np.fromfile(temp,dtype='<f4')
                        sf.write(temp,audio,48000,subtype='FLOAT',format='WAV')
                    else:
                        audio,sr=sf.read(temp,dtype='float32')
                        if sr!=48000 or audio.ndim!=1:
                            raise ValueError('Output must be mono 48kHz')
                    metrics=audio_metrics(audio)
                    if req.get('seconds') is not None and len(audio)!=round(req['seconds']*48000):
                        raise ValueError('Fixed-frame output length mismatch')
                    temp.replace(wav)
                    out.update(metrics,at_duration_limit=out['frames']>=750,
                               seconds_per_character=metrics['seconds']/max(1,len(item['text'])),
                               initial_noise_sha256=sha256(req['noise']) if req.get('noise') else None)
                    self.store.outcome(key,'ok',out,wav)
                    errors=0
                    self.progress()
                except Exception as error:
                    temp.unlink(missing_ok=True)
                    is_infrastructure=infrastructure(error)
                    self.store.outcome(key,'failed',dict(error=str(error),category=failure_category(error),infrastructure=is_infrastructure))
                    self.progress(True)
                    print('Generation failed',key,str(error),flush=True)
                    if is_infrastructure:
                        errors+=1
                        self.close_worker()
                        if errors>=CONFIG['consecutive_infrastructure_failures']:
                            raise RuntimeError(f'Three consecutive infrastructure failures: {case_id}') from error
                    else:
                        errors=0
            self.close_worker()

    def score(self,where,label):
        db=self.store.db
        for scorer,datasets in [('kana', ['jkyb']),('whisper',['jkyb','jvs']),
                                ('qwen',['practical']),('reazon',['practical']),
                                ('campplus',['jvs']),('eres2net',['practical'])]:
            if self.retry_failed:
                db.execute('DELETE FROM scores WHERE scorer=? AND status=?',(scorer,'failed'))
                db.commit()
            quoted=','.join('?' for _ in datasets)
            query=f'''SELECT t.* FROM tasks t LEFT JOIN scores s ON s.task_id=t.id AND s.scorer=?
                WHERE ({where}) AND t.status!='pending' AND t.dataset IN ({quoted}) AND s.task_id IS NULL ORDER BY t.rowid'''
            params=[scorer]+datasets
            total=db.execute('SELECT COUNT(*) FROM ('+query+')',params).fetchone()[0]
            self.stage(label+':'+scorer,total)
            errors=0
            for row in db.execute(query,params):
                try:
                    if row['status']!='ok':
                        out=dict(type='result',id=row['id'],transcript='',generation_failed=True)
                        if scorer in ('campplus','eres2net'):
                            out=dict(type='result',id=row['id'],unavailable='generation_failed')
                    else:
                        if scorer=='eres2net' and json.loads(row['result'])['voiced_seconds']<1.5:
                            self.store.score(row['id'],scorer,'ok',dict(unavailable='less_than_1.5_voiced_seconds'),row['wav_sha256'])
                            self.progress()
                            continue
                        if self.worker is None:
                            self.worker=Process([str(PYTHON),str(HERE/'score_worker.py'),scorer],
                                environment(),self.directory/f'score-{scorer}.log')
                        out=self.worker.request(dict(id=row['id'],wav=row['wav']))
                        if out['type']!='result':
                            raise RuntimeError(out.get('error',str(out)))
                    self.store.score(row['id'],scorer,'ok',out,row['wav_sha256'])
                    errors=0
                    self.progress()
                except Exception as error:
                    self.store.score(row['id'],scorer,'failed',dict(error=str(error)),row['wav_sha256'])
                    self.progress(True)
                    print('Scoring failed',scorer,row['id'],str(error),flush=True)
                    errors+=1
                    self.close_worker()
                    if errors>=CONFIG['consecutive_infrastructure_failures']:
                        raise RuntimeError(f'Three consecutive scoring failures: {scorer}') from error
            self.close_worker()

    def capacity(self):
        rows=self.store.db.execute("SELECT wav FROM tasks WHERE status='ok'").fetchall()
        actual=sum(Path(r['wav']).stat().st_size for r in rows)
        remaining=self.store.db.execute("SELECT COUNT(*) FROM tasks WHERE status='pending'").fetchone()[0]
        estimate=(actual/max(1,len(rows)))*remaining
        free=shutil.disk_usage(self.directory).free
        out=dict(completed=len(rows),bytes_written=actual,remaining_requests=remaining,
                 estimated_additional_bytes=estimate,free_bytes=free,reserve_factor=1.2,
                 status='ok' if estimate*1.2<free else 'insufficient_space')
        atomic_json(self.directory/'capacity.json',out)
        if out['status']!='ok':
            raise RuntimeError('Insufficient disk space for estimated WAV storage')

    def run(self):
        practical="profile='practical' AND mode='native'"
        self.generate(practical+' AND pilot=1','pilot-generation')
        self.score(practical+' AND pilot=1','pilot-scoring')
        self.capacity()
        self.generate(practical,'full-generation')
        self.score(practical,'full-scoring')
        self.generate("profile='fp32' AND mode='native'",'fp32-generation')
        self.score("profile='fp32' AND mode='native'",'fp32-scoring')
        self.generate("mode='common'",'common-noise-generation')
        self.score("mode='common'",'common-noise-scoring')
        self.stage('report')
        from report import report
        report(self.directory)


def execute(args):
    directory=BASE/'runs'/args.run_id
    directory.mkdir(parents=True,exist_ok=True)
    # A per-run lock rejects two orchestrators; the shared GPU lock serializes all campaigns.
    with (directory/'run.lock').open('w') as run_lock:
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=signature(args.datasets)
        path=directory/'manifest.json'
        new=not path.exists()
        ensure_manifest(directory,current)
        if new:
            atomic_json(directory/'splits.json',inputs(args.datasets)[3])
        campaign=Campaign(directory,args.retry_failed)
        def stop(sig,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        exit_code=1
        try:
            campaign.state['datasets']=args.datasets
            campaign.store.populate(tasks(args.datasets))
            if args.resume:
                campaign.stage('verifying-completed-artifacts')
                campaign.store.verify()
            campaign.stage('waiting-for-gpu-lock')
            with (ROOT/'artifacts/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                preflight=directory/'preflight.json'
                if not preflight.exists() or json.loads(preflight.read_text()).get('fingerprint')!=current['fingerprint']:
                    campaign.stage('preflight')
                    from validate import main as validate
                    validate(lock_gpu=False)
                    atomic_json(preflight,dict(fingerprint=current['fingerprint'],passed_at=utc()))
                campaign.run()
            pending=campaign.store.counts().get('pending',0)
            scoring_failures=campaign.store.db.execute("SELECT COUNT(*) FROM scores WHERE status='failed'").fetchone()[0]
            if pending or scoring_failures:
                raise RuntimeError(f'Incomplete evaluation: pending={pending}, scorer failures={scoring_failures}')
            campaign.state.update(status='complete',stage='complete',totals=campaign.store.counts())
            exit_code=0
        except KeyboardInterrupt:
            campaign.state.update(status='interrupted',error='Stopped by signal')
            exit_code=130
        except BaseException as error:
            traceback.print_exc()
            campaign.state.update(status='failed',error=f'{type(error).__name__}: {error}')
        finally:
            campaign.close_worker()
            campaign.finished.set()
            campaign.thread.join(timeout=2)
            campaign.state.update(exit_code=exit_code,finished_at=utc())
            campaign.save()
            campaign.store.db.close()
        return exit_code


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,help='default: all; resume inherits the saved selection')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--bootstrap',action='store_true',help='prepare inputs and retry public JVS quota failures before starting')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    latest=BASE/'latest.json'
    if not args.run_id:
        if args.resume or args.status or args.stop:
            args.run_id=json.loads(latest.read_text())['run_id']
        else:
            args.run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Unsafe run ID')
    directory=BASE/'runs'/args.run_id
    if args.datasets is None:
        manifest=directory/'manifest.json'
        args.datasets=(json.loads(manifest.read_text()).get('datasets',list(DATASETS))
                       if args.resume and manifest.exists() else list(DATASETS))
    args.datasets=sorted(set(args.datasets))
    if args.status or args.stop:
        state=json.loads((directory/'status.json').read_text())
        if state.get('status')=='running':
            try:
                os.kill(state['pid'],0)
            except ProcessLookupError:
                state['status']='interrupted'
        if args.stop and state.get('status')=='running':
            # Verify process identity before sending a signal to a saved PID.
            cmdline=Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            if not any(str(HERE/name).encode() in cmdline for name in ['run.py','bootstrap.py']) or args.run_id.encode() not in cmdline:
                raise RuntimeError('Saved PID does not identify this campaign')
            os.kill(state['pid'],signal.SIGTERM)
            print('Stop requested; current result remains resumable')
        else:
            print(json.dumps(state,ensure_ascii=False,indent=2))
        return 0
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(latest,dict(run_id=args.run_id))
    if args.bootstrap and not args.detach:
        cmd=[str(PYTHON),str(HERE/'bootstrap.py'),'--run-id',args.run_id]
        cmd+=['--datasets',*args.datasets]
        if args.resume:cmd+=['--resume']
        if args.retry_failed:cmd+=['--retry-failed']
        os.execv(PYTHON,cmd)
    if args.detach:
        cmd=[str(PYTHON),str(HERE/('bootstrap.py' if args.bootstrap else 'run.py')),'--run-id',args.run_id]
        cmd+=['--datasets',*args.datasets]
        if args.resume:cmd+=['--resume']
        if args.retry_failed:cmd+=['--retry-failed']
        import shlex
        shell='exec '+shlex.join(cmd)+' >> '+shlex.quote(str(directory/'run.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s','irodori-quality-'+args.run_id,
                        '-c',str(ROOT),shell],check=True)
        print('Started tmux session: irodori-quality-'+args.run_id)
        print('Results:',directory)
        return 0
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
