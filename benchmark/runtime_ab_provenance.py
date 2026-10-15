"""Hash actual fixed inputs and record candidate binary versions before timing."""
import hashlib
import json
import os
import platform
import subprocess
import time

from runtime_ab_setup import ROOT, OUT


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    paths = [ROOT / f'models/{v}/model.safetensors' for v in ('rf', 'mf')]
    paths += [ROOT / 'models/codec/weights.pth', ROOT / 'artifacts/benchmark/inputs/reference.wav']
    paths += list((ROOT / 'artifacts/onnx/export/ref').glob('*.onnx*'))
    paths += [ROOT / 'artifacts/build/irodori-onnx/release/examples/comparison_bench']
    paths += [ROOT / f'engines/{e}/irodori_tts/inference_runtime.py' for e in ('Irodori-TTS', 'Irodori-TTS-Optimiz')]
    paths += [ROOT / 'benchmark/config.json', ROOT / 'benchmark/prompts.json',
              ROOT / 'artifacts/benchmark/inputs/durations.json']
    data = dict(recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                platform=platform.platform(),
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,power.limit', '--format=csv,noheader'], text=True),
                hashes={str(p.relative_to(ROOT)): digest(p) for p in paths})
    runtimes = {}
    for engine in ('Irodori-TTS', 'Irodori-TTS-Optimiz', 'onnx'):
        python = ROOT / f'.venvs/runtime-ab-20261015/{engine}/bin/python'
        code = 'import torch,sys,json,torchaudio,torchcodec; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,torchaudio=torchaudio.__version__,torchcodec=torchcodec.__version__)))'
        info = json.loads(subprocess.check_output([str(python), '-c', code], text=True,
                                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')))
        assert info['cuda'] == '13.0', info
        runtimes[engine] = info
    data['candidates'] = runtimes
    (OUT / 'provenance.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(runtimes, indent=2))


if __name__ == '__main__':
    main()
