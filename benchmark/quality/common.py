"""Paths, immutable inputs and durable records shared by quality evaluation tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/quality'
ASSETS = BASE / 'assets'
CONFIG = json.loads((HERE / 'config.json').read_text())
PYTHON = ROOT / '.venvs/quality/bin/python'
DATASETS = ('jkyb', 'jvs', 'practical')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, cwd=ROOT, **kwargs)


def cases(profile='practical'):
    for engine, variants, precision in [('official', ['mf', 'rf'], 'bf16'),
                                         ('onnx', ['mf'], 'fp16'),
                                         ('speech', ['mf', 'rf'], 'f16')]:
        for variant in variants:
            p = 'fp32' if profile == 'fp32' else precision
            yield dict(engine=engine, variant=variant, precision=p,
                       profile=profile, id=f'{engine}-{variant}-{p}')


def stratified(items, count, group, salt):
    """Deterministic round-robin sampling; never inspect generated outputs."""
    from collections import defaultdict
    groups = defaultdict(list)
    for item in items:
        groups[group(item)].append(item)
    groups = {k: sorted(v, key=lambda x: digest([salt, x['id']])) for k, v in groups.items()}
    chosen = []
    while len(chosen) < min(count, len(items)):
        for key in sorted(groups):
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def environment():
    env = dict(os.environ)
    env.update(HF_HOME=str(ROOT / '.cache/huggingface'),
               HF_HUB_CACHE=str(ROOT / '.cache/huggingface/hub'),
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               RAYON_NUM_THREADS='8',
               NVIDIA_TF32_OVERRIDE='0', PYTHONUNBUFFERED='1')
    return env


def failure_category(error):
    text=str(error).lower()
    for category,patterns in [
        ('reference_insufficient',['insufficient reference']),
        ('non_finite',['non-finite','nonfinite']),('empty',['empty audio']),
        ('silent',['silent audio']),
        ('duration_limit',['at most 30','over 30','exceeds 30','length exceeds','seconds is outside']),
        ('infrastructure',['out of memory','timeout','worker exited','broken pipe','cuda error','device lost','startup failed']),
    ]:
        if any(pattern in text for pattern in patterns):return category
    return 'other'
