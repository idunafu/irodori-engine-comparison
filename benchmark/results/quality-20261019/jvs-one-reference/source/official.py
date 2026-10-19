"""Isolated FP32 reference/condition diagnostics; pinned runtime is unchanged."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torchaudio
import soundfile as sf

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[5]
sys.path.insert(0,str(ROOT/'benchmark/quality'));sys.path.insert(0,str(ROOT/'engines/Irodori-TTS'))
from common import digest
import irodori_tts.inference_runtime as ir
from irodori_tts.model import patch_sequence_with_mask
ir.SilentCipherWatermarker=lambda **kw:SimpleNamespace(ready=False)
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
variant=sys.argv[1]
runtime=ir.InferenceRuntime.from_key(ir.RuntimeKey(checkpoint=str(ROOT/f'models/{variant}/model.safetensors'),
 model_device='cuda',model_precision='fp32',codec_repo=str(ROOT/'models/codec/weights.pth'),codec_device='cuda',codec_precision='fp32',compile_model=False))
model=runtime.model;codec=runtime.codec
original_condition=runtime._load_speaker_embedding_condition
def emit(x):print('QUALITY '+json.dumps(x,ensure_ascii=False,allow_nan=False),flush=True)
emit(dict(type='ready'))
for line in sys.stdin:
 r=json.loads(line)
 if r.get('type')=='stop':break
 try:
  if r['op']=='dump':
   if r.get('pcm_in'):
    pcm=np.fromfile(r['pcm_in'],dtype='<f4');audio=torch.from_numpy(pcm)[None]
   else:
    audio,sr=ir._load_audio(r['reference'])
    if sr!=codec.sample_rate:audio=torchaudio.functional.resample(audio,sr,codec.sample_rate)
    if r.get('resampled_out'):audio.numpy().astype('<f4').tofile(r['resampled_out'])
    audio=codec._normalize_loudness(audio,codec.sample_rate,codec.normalize_db)[None]
    audio.numpy().astype('<f4').tofile(r['pcm_out'])
   latent=codec.encode_waveform(audio,codec.sample_rate,normalize_db=None,ensure_max=False)
   if r.get('latent_in'):latent=torch.from_numpy(np.fromfile(r['latent_in'],dtype='<f4').reshape(1,-1,32)).to('cuda')
   np.asarray(latent.detach().cpu(),dtype='<f4').tofile(r['latent_out'])
   torch.save(latent.detach().cpu().squeeze(0),r['latent_pt'])
   ref,mask=runtime._load_reference_latent(req=ir.SamplingRequest(text='あ',ref_latents=[r['latent_pt']]),batch_size=1,messages=[])
   ref,mask=patch_sequence_with_mask(ref,mask,model.cfg.speaker_patch_size)
   with torch.inference_mode():
    state=model.speaker_norm(model.speaker_encoder(ref,mask));state,mask=model._prepend_masked_mean_token(state,mask)
   np.asarray(state.detach().cpu(),dtype='<f4').tofile(r['state_out'])
   cache=ROOT/'artifacts/quality/reference-cache'/(digest([r.get('reference',''),'fp32',variant])+'.pt')
   cached_max_abs=None
   if cache.exists() and not r.get('pcm_in') and not r.get('latent_in'):
    old=torch.load(cache,weights_only=True,map_location='cpu').numpy();new=latent.detach().cpu().squeeze(0).numpy()
    cached_max_abs=float(np.max(np.abs(old-new))) if old.shape==new.shape else None
   emit(dict(type='result',id=r['id'],frames=latent.shape[1],speaker_tokens=state.shape[1],speaker_dim=state.shape[2],cached_latent_max_abs=cached_max_abs,normalize_db=codec.normalize_db))
  else:
   state_path=r.get('state_in')
   if state_path:
    state=torch.from_numpy(np.fromfile(state_path,dtype='<f4').reshape(1,-1,model.cfg.speaker_dim)).to('cuda')
    runtime._load_speaker_embedding_condition=lambda **kw:(state,torch.ones(state.shape[:2],device='cuda',dtype=torch.bool))
   else:runtime._load_speaker_embedding_condition=original_condition
   options=dict(text=r['text'],caption=None,no_ref=False,ref_latents=[r['latent_pt']],seed=r['seed'],seconds=r['seconds'],
                trim_tail=False,num_steps=4 if variant=='mf' else 40,cfg_scale_text=3.,cfg_scale_caption=0.,cfg_scale_speaker=5.,
                cfg_guidance_mode='independent',cfg_min_t=.5,cfg_max_t=1.,t_schedule_mode='linear',context_kv_cache=True,num_candidates=1)
   original=torch.randn
   used=False
   def noise(*args,**kw):
    global used
    shape=args[0] if len(args)==1 and isinstance(args[0],(tuple,list)) else args
    if not used and len(shape)==3 and shape[-1]==32:
     used=True
     return torch.from_numpy(np.fromfile(r['noise'],dtype='<f4').reshape(shape)).to(device=kw.get('device'),dtype=kw.get('dtype'))
    return original(*args,**kw)
   try:torch.randn=noise;out=runtime.synthesize(ir.SamplingRequest(**options))
   finally:torch.randn=original
   assert used
   audio=out.audio.detach().cpu().numpy().reshape(-1);sf.write(r['output'],audio,out.sample_rate,subtype='FLOAT')
   emit(dict(type='result',id=r['id'],samples=len(audio),frames=len(audio)//1920,noise_used=True))
 except Exception as e:
  import traceback;traceback.print_exc(file=sys.stderr)
  emit(dict(type='error',id=r.get('id'),error=f'{type(e).__name__}: {e}'))
