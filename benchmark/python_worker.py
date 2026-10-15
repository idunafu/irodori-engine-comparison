"""Resident Python worker. Detailed hooks are disabled for primary wall-time runs."""
import functools
import json
import os
from pathlib import Path
import sys
import time
import ast
import inspect
import textwrap

ROOT = Path(__file__).resolve().parents[1]
engine, variant, precision = sys.argv[1:4]
sys.path.insert(0, str(ROOT/'engines'/engine))
import numpy as np
import soundfile as sf
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
import irodori_tts.inference_runtime as mod
from types import SimpleNamespace
# Uniformly exclude the watermark subsystem from load and synthesis.
mod.SilentCipherWatermarker = lambda **kw: SimpleNamespace(ready=False)

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
cfg = json.loads((ROOT/'benchmark/config.json').read_text())
sync = lambda: torch.cuda.synchronize()
started = time.perf_counter()
runtime = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ROOT/f'models/{variant}/model.safetensors'),
    model_device='cuda', model_precision=precision, codec_repo=str(ROOT/'models/codec/weights.pth'),
    codec_device='cuda', codec_precision='fp32', compile_model=False))
sync()
load_ms = (time.perf_counter()-started)*1000
events = []
profile = False
cache_enabled = False
cache = {}
reference_latent = None
condition_excluded_ms = 0.

def emit(value):
    print('BENCH '+json.dumps(value, ensure_ascii=False), flush=True)

def instrument(owner, name, stage, memo=False):
    if not hasattr(owner, name): return
    original = getattr(owner, name)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if profile: sync(); t = time.perf_counter()
        out = original(*args, **kwargs)
        if profile:
            sync()
            shapes = [list(a.shape) for a in args if isinstance(a, torch.Tensor)]
            shapes += [list(a.shape) for a in kwargs.values() if isinstance(a, torch.Tensor)]
            events.append(dict(stage=stage, ms=(time.perf_counter()-t)*1000, shapes=shapes, cache_hit=False))
        if stage == 'codec_encode':
            global reference_latent
            reference_latent = out.detach()
        return out
    setattr(owner, name, wrapped)

def instrument_cpu(owner,name,stage):
    original=getattr(owner,name)
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        if profile:start=time.perf_counter()
        out=original(*args,**kwargs)
        if profile:
            shape=list(out[0].shape) if stage.endswith('tokenize') and isinstance(out,tuple) else []
            events.append(dict(stage=stage,ms=(time.perf_counter()-start)*1000,shape=shape))
        return out
    setattr(owner,name,wrapped)

instrument_cpu(runtime.tokenizer,'batch_encode','text_tokenize')
if runtime.caption_tokenizer is not None:instrument_cpu(runtime.caption_tokenizer,'batch_encode','caption_tokenize')
instrument_cpu(mod,'_load_audio','reference_file_read')
instrument_cpu(runtime.codec,'_normalize_loudness','reference_preprocess')
instrument(runtime.codec.model.encoder,'forward','codec_encoder_network')
instrument(runtime.codec.model.quantizer.in_proj,'forward','codec_encoder_projection')

# encode_conditions calls these modules; several duration/sampler paths reuse them.
instrument(runtime.model.text_encoder, 'forward', 'text_encode')
if getattr(runtime.model, 'caption_encoder', None) is not None:
    instrument(runtime.model.caption_encoder, 'forward', 'caption_encode', memo=True)
if getattr(runtime.model, 'speaker_encoder', None) is not None:
    instrument(runtime.model.speaker_encoder, 'forward', 'speaker_encode', memo=True)
instrument(runtime.codec, 'encode_file', 'reference_encode', memo=True)
instrument(runtime.codec, 'encode_waveform', 'codec_encode')
instrument(runtime.codec, 'decode_latent', 'codec_decode')
instrument(runtime.model, 'forward_with_encoded_conditions', 'dit_forward')
if getattr(runtime.model, 'duration_predictor', None) is not None:
    instrument(runtime.model.duration_predictor, 'forward', 'duration_predict')
if engine == 'Irodori-TTS-Optimiz':
    instrument(runtime.model.pretrained_text_backbone,'forward','text_caption_backbone_fused')
    instrument(runtime.model.text_encoder,'project_state','text_project')
    instrument(runtime.model.caption_encoder,'project_state','caption_project')
    import inspect
    original_conditions=runtime.model.encode_conditions
    signature=inspect.signature(original_conditions)
    def conditions_with_probes(*args,**kwargs):
        global profile
        if profile:
            bound=signature.bind(*args,**kwargs).arguments
            profile=False
            try:
                for name in ('text','caption'):
                    ids=bound.get(name+'_input_ids');mask=bound.get(name+'_mask')
                    if ids is None or mask is None:continue
                    sync();start=time.perf_counter()
                    state=getattr(runtime.model,name+'_encoder')(runtime.model.pretrained_text_backbone,ids,mask)
                    getattr(runtime.model,name+'_norm')(state)
                    sync();events.append(dict(stage=name+'_encode_probe',ms=(time.perf_counter()-start)*1000,shapes=[list(ids.shape)]))
            finally:profile=True
        return original_conditions(*args,**kwargs)
    runtime.model.encode_conditions=conditions_with_probes
import irodori_tts.inference_runtime as mod
class BenchScope:
    def __init__(self,stage):self.stage=stage;self.enabled=profile
    def __enter__(self):
        if self.enabled:
            sync();self.start=time.perf_counter();self.excluded_start=condition_excluded_ms
        return self
    def __exit__(self,*exc):
        if self.enabled:
            sync();ms=(time.perf_counter()-self.start)*1000
            excluded=condition_excluded_ms-self.excluded_start if self.stage=='sampling' else 0.
            events.append(dict(stage=self.stage,ms=ms-excluded,excluded_conditions_ms=excluded))

unmeasured_conditions=runtime.model.encode_conditions
def conditions_boundary(*args,**kwargs):
    global condition_excluded_ms
    if profile:sync();start=time.perf_counter()
    out=unmeasured_conditions(*args,**kwargs)
    if profile:sync();condition_excluded_ms+=(time.perf_counter()-start)*1000
    return out
runtime.model.encode_conditions=conditions_boundary

def sampling_boundaries(fn):
    """Only insert scopes: preserve every original sampling operation and argument."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    definition=next(x for x in tree.body if isinstance(x,ast.FunctionDef))
    condition_index=next(i for i,n in enumerate(definition.body)
        if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='encode_conditions' for c in ast.walk(n)))
    def scoped(stage,body):
        return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_bench_scope',ctx=ast.Load()),args=[ast.Constant(stage)],keywords=[]))],body=body)
    for node in list(ast.walk(definition)):
        if hasattr(node,'body') and isinstance(node.body,list):
            for i,child in list(enumerate(node.body)):
                if isinstance(child,ast.For) and 'num_steps' in ast.unparse(child.iter):
                    node.body[i]=scoped('sampling_loop',[child])
    # Include noise initialization and KV preparation, excluding the complete condition call.
    definition.body=[scoped('sampling',definition.body)]
    namespace=dict(inspect.unwrap(fn).__globals__,_bench_scope=BenchScope)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(fn),'exec'),namespace)
    return namespace[fn.__name__]

for method in ('sample_euler_rf_cfg','sample_euler_meanflow'):
    if hasattr(mod,method):setattr(mod,method,sampling_boundaries(getattr(mod,method)))
emit(dict(type='ready', load_ms=load_ms, device=torch.cuda.get_device_name(), precision=precision,
          codec_precision='fp32', resident_allocated=torch.cuda.memory_allocated(), resident_reserved=torch.cuda.memory_reserved()))
for line in sys.stdin:
    req = json.loads(line)
    if req.get('type') == 'stop': break
    try:
        profile = req.get('profile', False)
        cache_enabled = req.get('cache') == 'native_cache'
        if not cache_enabled:
            for name in ('_speaker_condition_cache','_caption_condition_cache'):
                if hasattr(runtime, name): getattr(runtime,name).clear()
            if hasattr(runtime,'_reference_latent_cache_dir'):
                runtime._reference_latent_cache_dir=ROOT/'artifacts/benchmark/python-cache'/str(os.getpid())
                for f in runtime._reference_latent_cache_dir.glob('*.pt'): f.unlink()
        events.clear()
        condition_excluded_ms=0.
        options = dict(text=req['text'], caption=cfg['caption'], ref_wav=str(ROOT/'artifacts/benchmark/inputs/reference.wav'),
                       num_steps=cfg['steps'][variant], seed=cfg['seed'], seconds=req.get('seconds'),
                       trim_tail=False, cfg_scale_text=3., cfg_scale_caption=3., cfg_scale_speaker=5.,
                       cfg_guidance_mode='independent', cfg_min_t=.5, cfg_max_t=1., t_schedule_mode='linear',
                       context_kv_cache=True, num_candidates=1)
        hit=False
        if engine == 'Irodori-TTS' and cache_enabled and reference_latent is not None:
            # The public ref_latent API skips reference decoding/encoding. Speaker still runs normally.
            options['ref_wav']=None
            path=ROOT/'artifacts/benchmark/python-cache'/f'{os.getpid()}-reference.pt'
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): torch.save(reference_latent.cpu().squeeze(0),path)
            options['ref_latent']=str(path);hit=True
        sync(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
        result=runtime.synthesize(SamplingRequest(**options))
        sync(); ms=(time.perf_counter()-t)*1000
        # Returned PCM has crossed the GPU boundary already in the runtime.
        audio=result.audio
        if isinstance(audio, torch.Tensor): audio=audio.detach().cpu().numpy()
        audio=np.asarray(audio).reshape(-1)
        hit=hit or any('loaded cached reference latent:' in m for m in result.messages)
        if req.get('output'): sf.write(req['output'],audio,result.sample_rate,subtype='FLOAT')
        emit(dict(type='result', id=req.get('id'), ms=ms, samples=len(audio), sample_rate=result.sample_rate,
                  instrumentation_version=4,
                  frames=(len(audio)+1919)//1920, events=list(events),
                  reference_cache_hit=hit, messages=result.messages,
                  caption_cache_hit=any('reused cached caption encoded state' in m for m in result.messages),
                  speaker_cache_hit=any('reused cached speaker encoded state' in m for m in result.messages),
                  stages=dict(result.stage_timings), pcm_sha256=__import__('hashlib').sha256(audio.tobytes()).hexdigest(),
                  finite=bool(np.isfinite(audio).all()), rms=float(np.sqrt(np.mean(audio**2))),
                  peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(dict(type='error', id=req.get('id'), error=str(e)))
