"""Fixed-input DACVAE probes; never writes into an installed environment."""
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/mf-runtime-cause-20261015'
sys.path.insert(0, str(ROOT / 'engines/Irodori-TTS'))
import numpy as np
import torch
from irodori_tts.codec import DACVAECodec

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
mode = os.environ.get('MF_CODEC_MODE', 'default')
torch.backends.cudnn.benchmark = mode == 'benchmark'
torch.backends.cudnn.enabled = mode != 'nocudnn'
codec = DACVAECodec.load(str(ROOT / 'models/codec/weights.pth'), dtype=torch.float32)
sync = torch.cuda.synchronize

def emit(data):
    print('BENCH ' + json.dumps(data), flush=True)

def provenance():
    import importlib.metadata as m
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), mode=mode,
                benchmark=torch.backends.cudnn.benchmark,
                optional_imported=[n for n in ('torchao','torchcodec') if n in sys.modules],
                libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                  if any(x in line for x in ('libcudnn', 'libcudart', 'libcublas'))}),
                dacvae=m.version('dacvae'))

emit(dict(type='ready', **provenance()))
inputs = {}
for line in sys.stdin:
    req = json.loads(line)
    if req.get('type') == 'stop':
        break
    try:
        seconds = req.get('seconds', 8)
        if seconds not in inputs:
            latent = torch.load(OUT / f'latent-{seconds}.pt', map_location='cpu', weights_only=True)
            inputs[seconds] = latent.to('cuda', dtype=torch.float32)
        latent = inputs[seconds]
        events = []
        handles = []
        if req.get('profile') == 'modules':
            # Leaf GPU events don't fence every module. Inclusive decoder block events are separate.
            for name, module in codec.model.named_modules():
                if not name or not (not list(module.children()) or name in ('decoder', 'quantizer.out_proj')):
                    continue
                pending = []
                def before(mod, args, name=name, pending=pending):
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(); pending.append((start, end, [list(a.shape) for a in args if isinstance(a, torch.Tensor)]))
                def after(mod, args, out, name=name, pending=pending):
                    start, end, shapes = pending.pop(); end.record()
                    events.append((name, type(mod).__name__, start, end, shapes))
                handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        sync(); start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(); start = time.perf_counter(); start_event.record()
        with torch.inference_mode():
            if req.get('profile') == 'operators':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    audio = codec.decode_latent(latent)
                prof.export_chrome_trace(str(OUT / (req['id'].replace('/', '-') + '.trace.json')))
            else:
                audio = codec.decode_latent(latent)
        end_event.record(); sync(); elapsed = (time.perf_counter()-start)*1000
        for handle in handles:
            handle.remove()
        data = audio.detach().cpu().numpy().reshape(-1)
        if req.get('output'):
            np.save(req['output'], data)
        detail = [dict(name=n, kind=k, ms=a.elapsed_time(b), shapes=s) for n,k,a,b,s in events]
        operators = []
        if req.get('profile') == 'operators':
            for event in prof.key_averages(group_by_input_shape=True):
                operators.append(dict(name=event.key, calls=event.count, cpu_us=event.cpu_time_total,
                                      self_cpu_us=event.self_cpu_time_total, device_us=event.device_time_total,
                                      self_device_us=event.self_device_time_total, shapes=event.input_shapes))
        emit(dict(type='result', id=req['id'], ms=elapsed, gpu_span_ms=start_event.elapsed_time(end_event),
                  samples=len(data), sample_rate=codec.sample_rate, finite=bool(np.isfinite(data).all()),
                  rms=float(np.sqrt(np.mean(data**2))), modules=detail, operators=operators,
                  peak_allocated=torch.cuda.max_memory_allocated(), **provenance()))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(dict(type='error', error=str(exc)))
