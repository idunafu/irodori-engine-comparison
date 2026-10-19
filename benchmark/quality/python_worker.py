"""Quality-only official runtime worker; original checkpoints and runtime stay untouched."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from common import ROOT
sys.path.insert(0, str(ROOT / 'engines/Irodori-TTS'))
import numpy as np
import soundfile as sf
import torch
import irodori_tts.inference_runtime as ir

ir.SilentCipherWatermarker = lambda **kw: SimpleNamespace(ready=False)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
variant, precision = sys.argv[1:3]
runtime = ir.InferenceRuntime.from_key(ir.RuntimeKey(
    checkpoint=str(ROOT / f'models/{variant}/model.safetensors'), model_device='cuda',
    model_precision=precision, codec_repo=str(ROOT / 'models/codec/weights.pth'),
    codec_device='cuda', codec_precision='fp32', compile_model=False))
runtime._reference_latent_cache_dir = ROOT / 'artifacts/quality/reference-cache'
cache = {}
latent_out = None
last_references = None
original_decode = runtime.codec.decode_latent


def decode(latent, *args, **kwargs):
    global latent_out
    latent_out = latent.detach().float().cpu().numpy()
    return original_decode(latent, *args, **kwargs)


runtime.codec.decode_latent = decode


def emit(v):
    print('QUALITY ' + json.dumps(v, ensure_ascii=False, allow_nan=False), flush=True)


emit(dict(type='ready', device=torch.cuda.get_device_name(), precision=precision))
for line in sys.stdin:
    req = json.loads(line)
    if req.get('type') == 'stop':
        break
    try:
        started = time.perf_counter()
        references = tuple(req.get('references', []))
        if references != last_references:
            if hasattr(runtime,'_speaker_condition_cache'):
                runtime._speaker_condition_cache.clear()
            last_references = references
        # Cache independently encoded clips, then concatenate in the requested order. Bound at 120 s.
        latent_paths = []
        for path in references:
            if path not in cache:
                audio, sr = ir._load_audio(path)
                latent = runtime.codec.encode_waveform(audio, sr)
                if isinstance(latent, tuple):
                    latent = latent[0]
                from common import digest
                dest = runtime._reference_latent_cache_dir / (digest([path, precision, variant]) + '.pt')
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent.detach().cpu().squeeze(0), dest)
                cache[path] = str(dest)
            latent_paths.append(cache[path])
        options = dict(text=req['text'], caption=req.get('caption') or None,
            no_ref=not references, ref_latents=latent_paths or None,
            num_steps=4 if variant == 'mf' else 40, seed=req['seed'], seconds=req.get('seconds'),
            trim_tail=False, cfg_scale_text=3., cfg_scale_caption=3., cfg_scale_speaker=5.,
            cfg_guidance_mode='independent', cfg_min_t=.5, cfg_max_t=1.,
            t_schedule_mode='linear', context_kv_cache=True, num_candidates=1)
        original_randn = torch.randn
        used = False
        def draw(*args, **kwargs):
            global used
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            if req.get('noise') and not used and len(shape) == 3 and shape[-1] == 32:
                used = True
                noise = np.fromfile(req['noise'], dtype='<f4').reshape(shape)
                return torch.as_tensor(noise, device=kwargs.get('device'), dtype=kwargs.get('dtype'))
            return original_randn(*args, **kwargs)
        latent_out = None
        try:
            torch.randn = draw
            result = runtime.synthesize(ir.SamplingRequest(**options))
        finally:
            torch.randn = original_randn
        if req.get('noise') and not used:
            raise RuntimeError('Initial noise was not consumed')
        audio = result.audio.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(audio).all():
            raise ValueError('Non-finite audio')
        sf.write(req['output'], audio, result.sample_rate, subtype='FLOAT', format='WAV')
        if req.get('latent_output'):
            np.asarray(latent_out, dtype='<f4').tofile(req['latent_output'])
        emit(dict(type='result', id=req['id'], samples=len(audio), frames=len(audio) // 1920,
                  sample_rate=result.sample_rate, ms=(time.perf_counter()-started)*1000,
                  noise_used=used, messages=result.messages))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(dict(type='error', id=req.get('id'), error=f'{type(e).__name__}: {e}'))
