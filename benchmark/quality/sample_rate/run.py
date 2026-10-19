"""Detached, resumable 96 kHz corpus sample-rate experiment; no listening step."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import zipfile

import numpy as np
import soundfile as sf
import soxr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from common import atomic_json, digest, environment, sha256, utc, PYTHON
from worker import Process, command

PREVIOUS = ROOT / 'artifacts/quality/runs/20261018-jvs/analysis/one-reference'
DEFAULT_OUT = ROOT / 'artifacts/quality/runs/20261019-tsukuyomi-samplerate'
RATES = [96000, 48000, 44100, 32000, 24000, 16000]
EXCLUDED = ['VOICEACTRESS100_050']  # Header is 48 kHz; do not fabricate a native 96 kHz original.
WAVE_COUNT = 99 * len(RATES)
REFS = [f'VOICEACTRESS100_{i:03d}' for i in range(1, 6)]
HELDOUT = [f'VOICEACTRESS100_{i:03d}' for i in range(11, 21)]
TARGET_IDS = [f'VOICEACTRESS100_{i:03d}' for i in range(21, 24)]
MODES = ['official', 'speech', 'speech_no_f16', 'official_speech_pcm']
STRICT = dict(GGML_VK_DISABLE_COOPMAT='1', GGML_VK_DISABLE_COOPMAT2='1', GGML_VK_DISABLE_F16='1')


def metrics(a, b):
    """Pointwise differences, preserving length mismatches as explicit fields."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    n = min(len(a), len(b))
    if not n or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Empty or non-finite comparison')
    x, y = a[:n].astype(np.float64), b[:n].astype(np.float64)
    xn, yn = np.linalg.norm(x), np.linalg.norm(y)
    if not xn or not yn:
        raise ValueError('Silent comparison')
    d = y - x
    return dict(relative_l2=float(np.linalg.norm(d) / xn), cosine=float(x @ y / (xn * yn)),
                rmse=float(np.sqrt(np.mean(d*d))), max_abs=float(np.max(np.abs(d))),
                left_samples=len(a), right_samples=len(b), compared_samples=n, length_delta=len(b)-len(a))


class Ledger:
    def __init__(self, out, retry_failed=False):
        self.conn = sqlite3.connect(out / 'results.sqlite3')
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, kind TEXT, spec TEXT, status TEXT, result TEXT)')
        self.conn.commit()
        self.retry_failed = retry_failed

    def get(self, ident, spec):
        row = self.conn.execute('SELECT spec,status,result FROM tasks WHERE id=?', (ident,)).fetchone()
        if row is None:
            return None
        if row[0] != digest(spec):
            raise RuntimeError('Resume specification mismatch: ' + ident)
        if row[1] == 'failed' and self.retry_failed:
            return None
        result = json.loads(row[2])
        if row[1] == 'ok':
            for p, h in result.get('files', {}).items():
                if not Path(p).exists() or sha256(p) != h:
                    raise RuntimeError('Resume artifact hash mismatch: ' + p)
        return result

    def put(self, ident, kind, spec, result):
        status = 'failed' if result.get('failed') else 'ok'
        self.conn.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?)',
                          (ident, kind, digest(spec), status, json.dumps(result, ensure_ascii=False, allow_nan=False)))
        self.conn.commit()

    def rows(self, kind):
        return [json.loads(r[0]) for r in self.conn.execute('SELECT result FROM tasks WHERE kind=? AND status="ok" ORDER BY id', (kind,))]


def build(out):
    binary = out / 'wave-worker'
    manifest = out / 'wave-worker-source.sha256'
    sources = [HERE / 'wave_worker.cpp', ROOT / 'engines/speech.cpp/src/common/resample.cpp',
               ROOT / 'engines/speech.cpp/src/common/wav.cpp']
    source_hash = digest({str(p): sha256(p) for p in sources})
    if binary.exists() and manifest.exists() and manifest.read_text() == source_hash:
        return binary
    folder = ROOT / 'artifacts/benchmark/build/speech.cpp/vulkan'
    cmd = shlex.split((folder / 'CMakeFiles/comparison_bench.dir/link.txt').read_text())
    obj = next(v for v in cmd if v.endswith('.cpp.o'))
    source = ROOT / 'engines/speech.cpp'
    cmd[cmd.index(obj):cmd.index(obj)+1] = [str(HERE / 'wave_worker.cpp'), str(source / 'tools/common/json-reader.cpp'), '-std=c++17',
        '-I'+str(source / 'src/common'), '-I'+str(source / 'tools/common'), '-I'+str(source / 'ggml/include')]
    cmd[cmd.index('-o')+1] = str(binary)
    subprocess.run(cmd, cwd=folder, check=True)
    manifest.write_text(source_hash)
    return binary


def configuration(binary):
    source = [*HERE.glob('*.py'), HERE / 'wave_worker.cpp', PREVIOUS / 'official.py', PREVIOUS / 'speech-diagnostic',
              ROOT / 'benchmark/quality/worker.py', ROOT / 'benchmark/quality/score_worker.py',
              ROOT / 'benchmark/quality/common.py', ROOT / 'benchmark/quality/config.json', binary,
              ROOT / 'engines/Irodori-TTS/irodori_tts/codec.py', ROOT / 'engines/speech.cpp/src/common/resample.cpp',
              ROOT / 'artifacts/quality/runs/20261018-jvs/manifest.json',
              ROOT / 'artifacts/quality/assets/jvs/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt']
    for variant in ['mf', 'rf']:
        cpp, _ = command(dict(engine='speech', variant=variant, precision='fp32'))
        source += [ROOT/f'models/{variant}/model.safetensors', Path(cpp[1])]
    source += [ROOT / 'models/codec/weights.pth']
    if set(REFS)&set(HELDOUT) or set(REFS)&set(TARGET_IDS) or set(HELDOUT)&set(TARGET_IDS):
        raise ValueError('Reference/held-out/target data overlap')
    return dict(schema=1, archive_sha256=sha256(ROOT/'wav_tsukuyomi.zip'), rates=RATES, references=REFS,
                heldout=HELDOUT, target_ids=TARGET_IDS, seeds=list(range(5)), variants=['mf', 'rf'], modes=MODES,
                precision='fp32 weights; default Vulkan may use F16 operands', strict_flags=STRICT,
                input_conversion='libsoxr VHQ, directly from original 96kHz, float32 WAV at every rate',
                controlled_length='official 48kHz reference prediction per variant/reference/text, shared across rates and modes',
                controlled_noise='same raw float32 array per variant/reference/text/seed across rates and modes',
                scoring='CAM++ cosine to held-out VOICEACTRESS100_011..020 centroid; one speaker, no Top-1',
                waveform_count=WAVE_COUNT, generation_count=3600,
                excluded_sources={EXCLUDED[0]:'source header is 48000 Hz rather than 96000 Hz'},
                packages={p: importlib.metadata.version(p) for p in ['numpy','soundfile','soxr','torch','torchaudio','sherpa-onnx']},
                sources={str(p): sha256(p) for p in sorted(set(source))})


class Experiment:
    def __init__(self, out, retry_failed):
        self.out = out
        self.ledger = Ledger(out, retry_failed)
        self.worker = None
        self.stopped = threading.Event()
        self.state = dict(status='running', stage='initializing', pid=os.getpid(), started_at=utc(),
                          done=0, total=0, failed=0, last_progress_at=None, eta_seconds=None)
        self.stage_start = time.monotonic()

    def save(self):
        self.state['failed'] = self.ledger.conn.execute('SELECT count(*) FROM tasks WHERE status="failed"').fetchone()[0]
        self.state['heartbeat_at'] = utc()
        atomic_json(self.out/'status.json', self.state)

    def stage(self, name, total):
        self.state.update(stage=name, done=0, total=total, eta_seconds=None)
        self.stage_start = time.monotonic()
        self.save()
        print(utc(), name, total, flush=True)

    def progress(self):
        self.state['done'] += 1
        self.state['last_progress_at'] = utc()
        if self.state['done']:
            self.state['eta_seconds'] = (time.monotonic()-self.stage_start)*(self.state['total']-self.state['done'])/self.state['done']

    @contextmanager
    def process(self, cmd, label, flags=None):
        env = environment()
        for key in STRICT:
            env.pop(key, None)
        env.update(flags or {})
        def start():
            return Process(cmd, env, self.out/(label+'.log'))
        self.worker = start()
        self.restart = start
        try:
            yield
        finally:
            self.worker.close()
            self.worker = None

    def request(self, ident, kind, request, files=(), meta=None):
        spec = dict(request=request, meta=meta or {}, files=list(map(str,files)))
        previous = self.ledger.get(ident, spec)
        if previous is not None:
            self.progress()
            return previous
        attempts = []
        for attempt in range(3):
            try:
                begin = time.monotonic()
                result = self.worker.request(dict(request, id=ident))
                if result.get('type') != 'result':
                    raise RuntimeError(str(result))
                if kind in ('generation', 'oracle'):
                    wav=Path(request['output'])
                    if result.get('raw_float'):
                        value=np.fromfile(wav,dtype='<f4')
                        sf.write(wav,value,48000,subtype='FLOAT',format='WAV')
                    audio,sr=sf.read(wav,dtype='float32')
                    if sr!=48000 or not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
                        raise ValueError('Empty, silent, non-finite or invalid-rate generated audio')
                    if meta and meta.get('frames_expected') and len(audio)!=meta['frames_expected']*1920:
                        raise ValueError('Unexpected generated sample count')
                result = dict(meta or {}, **{k:v for k,v in result.items() if k not in ('type','id')},
                              id=ident, seconds_elapsed=time.monotonic()-begin,
                              files={str(p):sha256(p) for p in files}, attempts=attempts)
                self.ledger.put(ident, kind, spec, result)
                self.progress()
                return result
            except (Exception,) as error:
                attempts.append(dict(error=f'{type(error).__name__}: {error}', at=utc()))
                text = str(error).lower()
                infrastructure = isinstance(error, TimeoutError) or any(t in text for t in
                    ['out of memory','worker exited','broken pipe','device lost','cuda error','timeout'])
                if infrastructure and attempt < 2:
                    self.worker.close()
                    self.worker = self.restart()
                    continue
                result = dict(meta or {}, id=ident, failed=True, error=attempts[-1]['error'], attempts=attempts)
                self.ledger.put(ident, kind, spec, result)
                self.state['failed'] += 1
                self.progress()
                if infrastructure:
                    raise RuntimeError('Three consecutive worker infrastructure failures: '+ident) from error
                print('FAILED', ident, result['error'], flush=True)
                return result

    def official(self, variant):
        return [str(ROOT/'.venvs/Irodori-TTS/bin/python'), str(HERE/'official_worker.py'), variant]

    def speech(self, variant):
        native, _ = command(dict(engine='speech', variant=variant, precision='fp32'))
        return [str(PREVIOUS/'speech-diagnostic'), native[1], native[2]]

    def paths(self, mode, rate, ref):
        base = self.out/'features'/mode/str(rate)/ref
        base.parent.mkdir(parents=True, exist_ok=True)
        return {k: str(base.with_suffix(s)) for k,s in [('pcm_out','.pcm.f32'), ('resampled_out','.resampled.f32'),
                 ('latent_out','.latent.f32'), ('latent_pt','.latent.pt'), ('state_out','.state.f32')]}

    def input(self, rate, ref):
        return self.out/'inputs'/str(rate)/(ref+'.wav')

    def prepare(self):
        self.stage('prepare-inputs', WAVE_COUNT)
        cached = self.out/'inputs.json'
        if cached.exists():
            index = json.loads(cached.read_text())
            for r in index:
                if sha256(r['path']) != r['sha256']:
                    raise RuntimeError('Input hash mismatch: '+r['path'])
                self.progress()
            return index
        archive = zipfile.ZipFile(ROOT/'wav_tsukuyomi.zip')
        members = sorted(i for i in archive.namelist() if i.lower().endswith('.wav'))
        if len(members) != 100:
            raise ValueError('Expected 100 source WAVs')
        records = [];audit=[]
        import io
        for member in members:
            payload = archive.read(member)
            info = sf.info(io.BytesIO(payload))
            audio, sr = sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
            ident = Path(member).stem
            audit.append(dict(id=ident,sample_rate=sr,subtype=info.subtype,channels=info.channels,frames=info.frames,
                              excluded=ident in EXCLUDED))
            if ident in EXCLUDED:
                if sr!=48000:raise ValueError('Excluded source header changed: '+member)
                continue
            if sr != 96000 or not np.isfinite(audio).all():
                raise ValueError('Unexpected source audio: '+member)
            mono = audio.mean(axis=1)
            for rate in RATES:
                path = self.input(rate,ident)
                path.parent.mkdir(parents=True,exist_ok=True)
                value = mono if rate == sr else soxr.resample(mono,sr,rate,quality='VHQ')
                sf.write(path,value,rate,subtype='FLOAT')
                records.append(dict(id=ident,rate=rate,path=str(path),sha256=sha256(path),samples=len(value),
                                    source=member,source_sha256=__import__('hashlib').sha256(payload).hexdigest(),
                                    source_subtype=info.subtype,source_channels=info.channels,source_frames=info.frames))
                self.progress()
        atomic_json(cached,records)
        atomic_json(self.out/'source-audit.json',audit)
        return records

    def waveform(self, index, binary):
        import torch
        import torchaudio
        torch.set_num_threads(8)
        self.stage('waveform-comparison',len(index))
        with self.process([str(binary)],'wave-worker'):
            for row in index:
                rate, ref = row['rate'],row['id']
                folder=self.out/'waveforms'/str(rate);folder.mkdir(parents=True,exist_ok=True)
                cpp=folder/(ref+'.speech.f32');official=folder/(ref+'.official.f32')
                result=self.request(f'wave:{rate}:{ref}','wave',dict(wav=row['path'],output=str(cpp)),[cpp],meta=dict(rate=rate,ref=ref))
                if result.get('failed'):continue
                audio,_=sf.read(row['path'],dtype='float32',always_2d=True);audio=torch.from_numpy(audio.mean(1))
                value=audio if rate==48000 else torchaudio.functional.resample(audio,rate,48000)
                value.numpy().astype('<f4').tofile(official)
                cpp_value=np.fromfile(cpp,dtype='<f4')
                oracle,sr=sf.read(self.input(48000,ref),dtype='float32')
                measure=dict(id=f'measure:{rate}:{ref}',rate=rate,ref=ref,
                             between=metrics(value.numpy(),cpp_value),
                             official_vs_soxr48=metrics(oracle,value.numpy()),speech_vs_soxr48=metrics(oracle,cpp_value),
                             files={str(official):sha256(official),str(cpp):sha256(cpp)})
                self.ledger.put(measure['id'],'wave_metric',dict(input_sha=row['sha256']),measure)

    def encode(self):
        self.stage('reference-encoding',180)
        modes=['official','speech','speech_no_f16','official_speech_pcm','speech_official_pcm','speech_no_f16_official_pcm']
        for mode in modes:
            official=mode in ('official','official_speech_pcm')
            cmd=self.official('mf') if official else self.speech('mf')
            flags=STRICT if 'no_f16' in mode else {}
            with self.process(cmd,'encode-'+mode,flags):
                for rate in RATES:
                    for ref in REFS:
                        paths=self.paths(mode,rate,ref)
                        req=dict(op='dump',reference=str(self.input(rate,ref)),**paths)
                        if mode.endswith('_official_pcm'):
                            req['pcm_in']=self.paths('official',rate,ref)['pcm_out']
                        elif mode=='official_speech_pcm':
                            req['pcm_in']=self.paths('speech',rate,ref)['pcm_out']
                        files=[paths['latent_out'],paths['state_out']]
                        if official:files.append(paths['latent_pt'])
                        # pcm_in omits the resampler dump; PCM values are already available in its source condition.
                        if not req.get('pcm_in'):files += [paths['pcm_out'],paths['resampled_out']]
                        result=self.request(f'encode:{mode}:{rate}:{ref}','encode',req,files,
                                            dict(mode=mode,rate=rate,ref=ref))
                        if result.get('failed'):raise RuntimeError('Reference encoding failed: '+result['id'])

    def targets(self):
        path=ROOT/'artifacts/quality/assets/jvs/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt'
        texts=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
        return {key:texts[key] for key in TARGET_IDS}

    def oracle(self):
        self.stage('official-length-oracle',30)
        for variant in ['mf','rf']:
            with self.process(self.official(variant),'length-'+variant):
                for ref in REFS:
                    for text_id,text in self.targets().items():
                        wav=self.out/'oracle'/(f'{variant}-{ref}-{text_id}.wav');wav.parent.mkdir(exist_ok=True)
                        # No injected noise here: official automatic duration chooses the shared frame count.
                        # Existing diagnostic requires its noise hook; an isolated worker overlay below supplies native RNG.
                        request=dict(op='synthesize',text=text,seed=0,output=str(wav),
                                     latent_pt=self.paths('official',48000,ref)['latent_pt'],native_noise=True)
                        result=self.request(f'oracle:{variant}:{ref}:{text_id}','oracle',request,[wav],
                                            dict(variant=variant,ref=ref,text_id=text_id))
                        if result.get('failed'):raise RuntimeError('Length oracle failed: '+result['id'])
                        if not 0<result['frames']<=750:raise ValueError('Predicted length outside shared 30-second range')

    def generate(self):
        self.stage('controlled-generation',3600)
        oracles={(r['variant'],r['ref'],r['text_id']):r for r in self.ledger.rows('oracle')}
        for variant in ['mf','rf']:
            for mode in MODES:
                official=mode in ('official','official_speech_pcm')
                with self.process(self.official(variant) if official else self.speech(variant),'generate-'+variant+'-'+mode,
                                  STRICT if mode=='speech_no_f16' else {}):
                    for rate in RATES:
                        for ref in REFS:
                            paths=self.paths(mode,rate,ref)
                            for text_id,text in self.targets().items():
                                frames=oracles[variant,ref,text_id]['frames']
                                for seed in range(5):
                                    group=digest([variant,ref,text_id,seed])[:24]
                                    noise=self.out/'noise'/(group+'.f32');noise.parent.mkdir(exist_ok=True)
                                    if not noise.exists():np.random.default_rng(int(group[:16],16)).standard_normal((frames,32),dtype=np.float32).tofile(noise)
                                    wav=self.out/'audio'/variant/mode/str(rate)/(group+'.wav');wav.parent.mkdir(parents=True,exist_ok=True)
                                    req=dict(op='synthesize',text=text,seed=seed,seconds=float(np.nextafter(frames/25,np.inf)),
                                             noise=str(noise),output=str(wav),latent_pt=paths['latent_pt'])
                                    if not official:req['latent_in']=paths['latent_out']
                                    result=self.request(f'generate:{variant}:{mode}:{rate}:{group}','generation',req,[wav],
                                                        dict(variant=variant,mode=mode,rate=rate,ref=ref,text_id=text_id,seed=seed,
                                                             frames_expected=frames,noise_sha256=sha256(noise),wav=str(wav)))
                                    if result.get('failed'):continue
                                    info=sf.info(wav)
                                    if info.samplerate!=48000 or info.frames!=frames*1920:raise ValueError('Unexpected generated sample count')
        self.save()

    def score(self):
        rows=self.ledger.rows('generation')
        self.stage('campplus-scoring',len(rows)+len(HELDOUT))
        with self.process([str(PYTHON),str(HERE.parent/'score_worker.py'),'campplus'],'campplus'):
            embeddings=[]
            for ref in HELDOUT:
                wav=self.input(48000,ref)
                result=self.request('centroid:'+ref,'centroid',dict(wav=str(wav)),meta=dict(ref=ref,wav_sha256=sha256(wav)))
                if result.get('failed'):raise RuntimeError('Held-out embedding failure')
                embeddings.append(result['embedding'])
            centroid=np.mean(embeddings,axis=0);centroid/=np.linalg.norm(centroid)
            atomic_json(self.out/'speaker-centroid.json',dict(heldout=HELDOUT,embedding=centroid.tolist()))
            for row in rows:
                meta={k:row[k] for k in ['variant','mode','rate','ref','text_id','seed','wav','frames_expected','noise_sha256']}
                meta['wav_sha256']=row['files'][row['wav']]
                result=self.request('score:'+row['id'],'score',dict(wav=row['wav']),meta=meta)
                # Analysis computes cosine from the saved embedding and fixed centroid.
        self.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    # A second invocation must never race a detached owner or change its database.
    ownership=(out/'run.lock').open('a')
    fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Experiment(out,args.retry_failed)
    def interrupted(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def heartbeat():
        while not run.stopped.wait(30):
            # sqlite connections belong to the main thread; heartbeat only writes in-memory state.
            atomic_json(out/'status.json',dict(run.state,heartbeat_at=utc()))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        binary=build(out)
        config=configuration(binary)
        manifest=out/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=config:raise RuntimeError('Resume manifest mismatch; use a new output directory')
        else:atomic_json(manifest,config)
        index=run.prepare()
        if args.prepare_only:
            run.state.update(status='prepared',stage='prepared',exit_code=0,finished_at=utc());run.save();return
        run.waveform(index,binary)
        run.stage('waiting-for-gpu-lock',0)
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run.encode();run.oracle();run.generate();run.score()
        run.stage('analysis',1)
        subprocess.run([str(PYTHON),str(HERE/'analyze.py'),'--out',str(out)],cwd=ROOT,check=True)
        run.progress();run.state.update(status='complete',stage='complete',exit_code=0,finished_at=utc());run.save()
    except BaseException as error:
        run.state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                         error=f'{type(error).__name__}: {error}',exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,finished_at=utc())
        run.save();traceback.print_exc();raise
    finally:
        if run.worker:run.worker.close()
        run.stopped.set();thread.join(timeout=2);run.ledger.conn.close()


if __name__=='__main__':main()
