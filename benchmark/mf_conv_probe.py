"""Standalone reproducer for the largest regressing DACVAE Conv1d shape."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/mf-runtime-cause-20261015'

def worker(mode):
    import time
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=mode=='benchmark'
    torch.set_float32_matmul_precision('highest')
    rng=np.random.default_rng(42)
    x=torch.from_numpy(rng.standard_normal((1,768,2400),dtype=np.float32)).cuda()
    w=torch.from_numpy(rng.standard_normal((768,768,7),dtype=np.float32)/np.sqrt(np.float32(768*7))).cuda()
    def conv():return torch.nn.functional.conv1d(x,w,padding=3)
    rows=[]
    with torch.inference_mode():
        for i in range(26):
            torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();a.record();y=conv();b.record();torch.cuda.synchronize()
            rows.append(dict(ms=(time.perf_counter()-start)*1000,gpu_ms=a.elapsed_time(b),warmup=i<6))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:conv()
        torch.cuda.synchronize()
    kernels=[dict(name=e.name,us=e.device_time_total) for e in prof.events() if str(e.device_type).endswith('CUDA')]
    print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        mode=mode,input_shape=list(x.shape),weight_shape=list(w.shape),dilation=1,padding=3,tf32=False,
        samples=rows,kernels=kernels,finite=bool(y.isfinite().all()),sum=float(y.sum()),
        imported_optional=[n for n in ('torchao','torchcodec','torchaudio') if n in sys.modules])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--mode',default='default')
    args=parser.parse_args()
    if args.worker:worker(args.mode);return
    import fcntl
    import mf_cause
    mf_cause.prepare()
    with (ROOT/'artifacts/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for block in range(3):
            cases=[(s,m) for s in ('old','old924','new') for m in ('default','benchmark')]
            for stack,mode in (cases if block%2==0 else cases[::-1]):
                key=f'{block}/{stack}/{mode}'
                path=OUT/'conv-probe.jsonl'
                if path.exists() and key in {json.loads(s)['key'] for s in path.read_text().splitlines()}:continue
                python=mf_cause.PYTHONS.get(stack,ROOT/f'.venvs/mf-runtime-cause-20261015/{stack}/bin/python')
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
                env.pop('LD_LIBRARY_PATH',None)
                print('CONV',key,flush=True)
                result=subprocess.check_output([str(python),str(Path(__file__).resolve()),'--worker','--mode',mode],env=env,text=True)
                with path.open('a') as f:f.write(json.dumps(dict(key=key,round=block,stack=stack,result=json.loads(result)))+'\n')
    assert mf_cause.snapshot()==json.loads((OUT/'preservation.json').read_text())

if __name__=='__main__':main()
