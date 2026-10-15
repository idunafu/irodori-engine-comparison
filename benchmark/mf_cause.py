"""Small serial MF runtime experiments, isolated from every earlier benchmark."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import run as harness

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/mf-runtime-cause-20261015'
DEST = ROOT / 'benchmark/results/mf-runtime-cause-20261015'
PYTHONS = dict(old=ROOT/'.venvs/Irodori-TTS/bin/python',
               new=ROOT/'.venvs/runtime-ab-20261015/Irodori-TTS/bin/python')
BASE_COMMAND = harness.command

def start_telemetry():
    """Call after NVML initialization, and stop before shutting NVML down."""
    import speech_ab
    speech_ab.OUT=OUT
    stop=threading.Event()
    thread=threading.Thread(target=speech_ab.telemetry,args=(stop,),daemon=True)
    thread.start()
    return stop,thread

def snapshot():
    files = []
    for env in ('.venvs/Irodori-TTS', '.venvs/Irodori-TTS-Optimiz', '.venvs/onnx',
                '.venvs/runtime-ab-20261015/Irodori-TTS', '.venvs/runtime-ab-20261015/Irodori-TTS-Optimiz',
                '.venvs/runtime-ab-20261015/onnx', 'benchmark/.venv'):
        files.extend((ROOT/env).rglob('*.dist-info/RECORD'))
        files.extend((ROOT/env).glob('pyvenv.cfg'))
    for folder in ('benchmark/results', 'artifacts/runtime-ab-20261015'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and DEST not in p.parents)
    for name in ('engines.lock.json', 'models.lock.json', 'benchmark/config.json', 'benchmark/prompts.json',
                 'benchmark/python_worker.py', 'artifacts/benchmark/results.jsonl'):
        files.append(ROOT/name)
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def prepare():
    OUT.mkdir(parents=True, exist_ok=True); DEST.mkdir(parents=True, exist_ok=True)
    path = OUT/'preservation.json'
    current = snapshot()
    if path.exists():
        assert current == json.loads(path.read_text()), 'An existing environment/result changed'
    else:
        path.write_text(json.dumps(current, indent=2)+'\n')
    source = (ROOT/'benchmark/python_worker.py').read_text()
    source = source.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})')
    source = source.replace("torch.set_float32_matmul_precision('highest')", "torch.set_float32_matmul_precision('highest')\ntorch.backends.cudnn.benchmark = os.environ.get('MF_BENCHMARK') == '1'")
    source = source.replace("codec_precision='fp32', resident_allocated=", "codec_precision='fp32', torch_version=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark, resident_allocated=")
    source = source.replace("ROOT/'artifacts/benchmark/python-cache'", f'Path({str(OUT/"python-cache")!r})')
    # Optional capture is used only by an untimed setup request, before measured requests.
    anchor = "original = getattr(owner, name)\n"
    source = source.replace(anchor, anchor + "    if stage == 'codec_decode' and os.environ.get('MF_CAPTURE'):\n"
        "        def capture(latent):\n"
        f"            path=Path({str(OUT)!r})/('latent-'+str(int(latent.shape[1]/25))+'.pt')\n"
        "            if not path.exists(): torch.save(latent.detach().cpu(),path)\n"
        "            return original(latent)\n"
        "        setattr(owner,name,capture)\n"
        "        return\n")
    (OUT/'python_worker.py').write_text(source)

def command(case):
    cmd, env = BASE_COMMAND(case)
    cmd[0] = str(PYTHONS.get(case['stack'], ROOT/f'.venvs/mf-runtime-cause-20261015/{case["stack"]}/bin/python'))
    cmd[1] = str(ROOT/'benchmark/mf_cause_worker.py' if case['phase']=='codec' else OUT/'python_worker.py')
    env.update(PYTHONDONTWRITEBYTECODE='1', MF_CODEC_MODE=case['mode'],
               XDG_CACHE_HOME=str(ROOT/'.cache'), HF_HOME=str(ROOT/'.cache/huggingface'))
    env.pop('LD_LIBRARY_PATH', None)
    if case['mode']=='v7': env['TORCH_CUDNN_V8_API_DISABLED']='1'
    if case['mode']=='benchmark': env['MF_BENCHMARK']='1'
    if case.get('capture'): env['MF_CAPTURE']='1'
    return cmd, env

def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['capture','codec','e2e','check'])
    p.add_argument('--stacks', nargs='+', default=['old','new']); p.add_argument('--modes', nargs='+', default=['default'])
    p.add_argument('--rounds',type=int,default=3); p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--tag',default='initial')
    p.add_argument('--seconds',nargs='+',type=int,default=[2,8,24]); args=p.parse_args()
    prepare()
    if args.phase=='check': print('PRESERVED',len(snapshot())); return
    harness.command = command
    existing = OUT/'results.jsonl'
    done = {json.loads(s)['key'] for s in existing.read_text().splitlines()} if existing.exists() else set()
    cases = [(s,m,precision) for s in args.stacks for m in args.modes
             for precision in (['fp32','bf16'] if args.phase=='e2e' else ['fp32'])]
    with (ROOT/'artifacts/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); monitor=harness.Monitor()
        stop,thread=start_telemetry()
        try:
            for round_id in range(args.rounds if args.phase!='capture' else 1):
                for stack,mode,precision in (cases if round_id%2==0 else cases[::-1]):
                    case=dict(engine='Irodori-TTS',variant='mf',backend='cuda',precision=precision,
                        stack=stack,mode=mode,phase=args.phase,capture=args.phase=='capture',
                        experiment=args.tag,
                        case=f'mf-cause-{args.phase}-{stack}-{mode}-{precision}')
                    prefix=f'{args.phase}/{stack}/{mode}/{precision}/{round_id}'
                    if args.tag!='initial':prefix=args.tag+'/'+prefix
                    if prefix+'/complete' in done: continue
                    print('START',prefix,flush=True)
                    with harness.Worker(case,monitor) as worker:
                        harness.append(OUT/'startup.jsonl',dict(key=prefix,ready=worker.ready))
                        for seconds in args.seconds:
                            for cache in (['reencode','native_cache'] if args.phase=='e2e' else ['native_cache']):
                                req=dict(id=prefix,seconds=seconds,cache=cache,text=harness.PROMPTS[8]['text'])
                                for _ in range(4 if args.phase!='capture' else 1): worker.request(req)
                                if args.phase=='capture': continue
                                for i in range(args.repeats):
                                    key=f'{prefix}/{seconds}/{cache}/normal/{i}'
                                    output=OUT/f'{args.phase}-{stack}-{mode}-{precision}-{seconds}-{cache}.{("npy" if args.phase=="codec" else "wav")}'
                                    out=worker.request(dict(req,id=key,output=str(output) if i==0 else None))
                                    harness.append(existing,dict(key=key,round=round_id,case=case,seconds=seconds,cache=cache,kind='normal',result=out))
                                for kind in (['modules','operators'] if args.phase=='codec' else ['stages']):
                                    for i in range(1 if kind=='operators' else 3):
                                        key=f'{prefix}/{seconds}/{cache}/{kind}/{i}'
                                        out=worker.request(dict(req,id=key,profile=kind if args.phase=='codec' else True))
                                        harness.append(existing,dict(key=key,round=round_id,case=case,seconds=seconds,cache=cache,kind=kind,result=out))
                        harness.append(existing,dict(key=prefix+'/complete',kind='complete'))
                    print('DONE',prefix,flush=True)
        finally: stop.set();thread.join();monitor.close()
    assert snapshot()==json.loads((OUT/'preservation.json').read_text())

if __name__=='__main__': main()
