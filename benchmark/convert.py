"""Convert the pinned original FP32 checkpoints, never upcast a distribution GGUF."""
import json
import os
from pathlib import Path
import subprocess
import sys
import struct

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmark/models'

def run(argv, **kw):
    print(' '.join(map(str,argv)),flush=True)
    subprocess.run(list(map(str,argv)),check=True,**kw)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    py=ROOT/'.venvs/Irodori-TTS/bin/python'
    env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONPATH=str(ROOT/'engines/Irodori-TTS'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    codec=OUT/'codec-raw.safetensors'
    if not codec.exists():
        run([py,ROOT/'engines/audio.cpp/tests/irodori_tts/convert_dacvae_pth_to_safetensors.py',
             '--input',ROOT/'models/codec/weights.pth','--output',codec],env=env)
    converter=OUT/'speech-convert.py'
    code=(ROOT/'engines/speech.cpp/reference/irodori-tts/convert.py').read_text()
    code=code.replace('from pins import CODEC, MODELS, snapshot',
        "CODEC = {'local':'codec'}\nMODELS = "+repr({v:dict(local=v,repository='Aratako/Irodori-TTS-v4.1-Small'+('-MF' if v=='mf' else ''),revision=r) for v,r in [('rf','2b28324dc263ed5e6638b3cf3dd94c82ead07b4b'),('mf','ccc78f5d480b6e51b69b2d5042a14c4da04fea6e')]})+"\ndef snapshot(pin):\n    return "+repr(str(ROOT/'models'))+" + '/' + pin['local']")
    converter.write_text(code)
    sidecars=OUT/'audio-sidecars'
    sidecars.mkdir(exist_ok=True)
    # Model config is metadata of the very same FP32 checkpoint.
    with (ROOT/'models/rf/model.safetensors').open('rb') as f:
        header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    (sidecars/'model_config.json').write_text(header['__metadata__']['config_json'])
    (sidecars/'text_encoder_config.json').write_text(header['__metadata__']['text_encoder_config_json'])
    # Other codec/package sidecars contain configuration, never recovered model weights.
    import importlib.util
    spec=importlib.util.spec_from_file_location('audio_convert',ROOT/'engines/audio.cpp/tools/community_models/irodori_tts/convert_irodori_v4_1_anime_gguf.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.materialize_sidecars(ROOT/'models/audio-rf-runtime/irodori-tts-v4-small-f16.gguf',sidecars)
    (sidecars/'model_config.json').write_text(header['__metadata__']['config_json'])
    (sidecars/'text_encoder_config.json').write_text(header['__metadata__']['text_encoder_config_json'])
    for variant in ('rf','mf'):
        dest=OUT/f'speech-{variant}'
        if not list(dest.glob('*.gguf')):
            run([py,converter,variant,dest],env=env)
        f32=next(dest.glob('*.gguf'))
        f16=OUT/f'speech-{variant}-f16.gguf'
        if not f16.exists():run([ROOT/'artifacts/build/speech.cpp/cuda/speech','quantize',f32,f16,'--type','f16'],env=env)
        if variant=='rf':
            for typ in ('orig','f16'):
                path=OUT/f'audio-rf-{typ}.gguf'
                if not path.exists():
                    run([ROOT/'artifacts/build/audio.cpp/cuda/bin/audiocpp_gguf',
                        '--input',f'model_weights={ROOT}/models/rf/model.safetensors',
                        '--input',f'codec_weights={codec}', '--root',sidecars,'--family','irodori_tts',
                        '--type',typ,'--keep-type','codec_weights/*=orig', '--output',path,
                        '--sidecar',f'{ROOT}/models/rf/tokenizer/tokenizer.json=tokenizer/tokenizer.json',
                        '--sidecar',f'{ROOT}/models/rf/tokenizer/tokenizer_config.json=tokenizer/tokenizer_config.json'],env=env)

if __name__=='__main__': main()
