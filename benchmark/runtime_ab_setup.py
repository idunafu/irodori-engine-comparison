"""Build isolated runtime candidates without syncing any original project."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/runtime-ab-20261015'
ENV = dict(os.environ, UV_CACHE_DIR=str(ROOT / '.cache/uv'),
           XDG_DATA_HOME=str(ROOT / '.cache/data'), UV_LINK_MODE='hardlink')


def call(args):
    print(' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), env=ENV, cwd=ROOT, check=True,
                          text=True, stdout=subprocess.PIPE).stdout


def metadata():
    data = {}
    for name in ('torch', 'torchaudio', 'torchcodec', 'onnx', 'onnxruntime-gpu', 'onnxscript'):
        url = f'https://pypi.org/pypi/{name}/json'
        with urllib.request.urlopen(url) as response:
            d = json.load(response)
        data[name] = dict(version=d['info']['version'], requires_python=d['info']['requires_python'],
                          source=url, wheels=[x['filename'] for x in d['urls']])
    return data


def snapshot():
    files = [ROOT / n for n in ('engines.lock.json', 'models.lock.json', 'benchmark/config.json',
             'benchmark/prompts.json', 'artifacts/benchmark/results.jsonl', 'artifacts/benchmark/preflight.jsonl')]
    for engine in ('Irodori-TTS', 'Irodori-TTS-Optimiz', 'onnx'):
        files += list((ROOT / f'.venvs/{engine}').rglob('*.dist-info/RECORD'))
        files += list((ROOT / f'.venvs/{engine}').glob('pyvenv.cfg'))
    files += list((ROOT / 'benchmark/results').glob('*.*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = OUT / 'preservation.json'
    current = snapshot()
    if before.exists():
        assert json.loads(before.read_text()) == current, 'Original environment changed'
    else:
        before.write_text(json.dumps(current, indent=2) + '\n')
    info_path = OUT / 'latest.json'
    if not info_path.exists():
        pinned = ROOT / 'environments/runtime-ab-20261015/versions.json'
        info = json.loads(pinned.read_text()) if pinned.exists() else metadata()
        info_path.write_text(json.dumps(info, indent=2) + '\n')
    versions = {k: v['version'] for k, v in json.loads(info_path.read_text()).items()}
    for engine in ('Irodori-TTS', 'Irodori-TTS-Optimiz', 'onnx'):
        target = ROOT / f'.venvs/runtime-ab-20261015/{engine}'
        python = target / 'bin/python'
        if (OUT / f'{engine}-packages.txt').exists():
            continue
        old = call(['uv', 'pip', 'freeze', '--python', ROOT / f'.venvs/{engine}/bin/python'])
        (OUT / f'{engine}-original-packages.txt').write_text(old)
        if not python.exists():
            call(['uv', 'venv', '--python', '3.11' if engine == 'onnx' else '3.10.20', target])
        pinned = ROOT / f'environments/runtime-ab-20261015/{engine}-requirements.txt'
        if pinned.exists():
            call(['uv', 'pip', 'sync', '--python', python, '--index',
                  'https://download.pytorch.org/whl/cu130', '--index-strategy',
                  'unsafe-best-match', pinned])
            (OUT / f'{engine}-packages.txt').write_text(call(['uv', 'pip', 'freeze', '--python', python]))
            print('READY', engine, flush=True)
            continue
        if engine != 'onnx':
            skip = ('torch==', 'torchaudio==', 'torchcodec==', 'triton==', 'nvidia-',
                    '-e ', 'irodori-tts', 'cuda-')
            frozen = '\n'.join(line for line in old.splitlines() if not line.startswith(skip)) + '\n'
            req = OUT / f'{engine}-unchanged.txt'
            req.write_text(frozen)
            call(['uv', 'pip', 'install', '--python', python, '--no-deps', '-r', req])
        call(['uv', 'pip', 'install', '--python', python, '--index',
              'https://download.pytorch.org/whl/cu130', '--index-strategy', 'unsafe-best-match',
              f'torch=={versions["torch"]}+cu130', f'torchaudio=={versions["torchaudio"]}',
              f'torchcodec=={versions["torchcodec"]}'])
        if engine == 'onnx':
            call(['uv', 'pip', 'install', '--python', python,
                  f'onnx=={versions["onnx"]}', f'onnxscript=={versions["onnxscript"]}',
                  f'onnxruntime-gpu[cuda,cudnn]=={versions["onnxruntime-gpu"]}'])
        (OUT / f'{engine}-packages.txt').write_text(call(['uv', 'pip', 'freeze', '--python', python]))
        print('READY', engine, flush=True)
    assert snapshot() == json.loads(before.read_text()), 'Original environment changed'


if __name__ == '__main__':
    main()
