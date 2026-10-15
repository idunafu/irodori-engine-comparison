"""Build a frozen upstream snapshot in isolation; never update the benchmark baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import build as instrument

ROOT=Path(__file__).resolve().parents[1]
AB=ROOT/'artifacts/speech-ab'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(sha):
    dest=AB/sha
    upstream=dest/'engines/speech.cpp'
    if not upstream.exists():
        dest.mkdir(parents=True,exist_ok=True)
        archive=dest/'upstream.tar.gz'
        if not archive.exists():
            urllib.request.urlretrieve(f'https://codeload.github.com/nyosegawa/speech.cpp/tar.gz/{sha}',archive)
        stage=dest/'unpack';stage.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(stage,filter='data')
        source=next(stage.glob('speech.cpp-*'))
        upstream.parent.mkdir(exist_ok=True)
        shutil.move(source,upstream)
        shutil.rmtree(stage)
        # The current upstream tree pins exactly the baseline's ggml SHA.
        request=urllib.request.Request(f'https://api.github.com/repos/nyosegawa/speech.cpp/git/trees/{sha}',
            headers={'User-Agent':'irodori-engine-comparison'})
        with urllib.request.urlopen(request,timeout=45) as r:tree=json.load(r)
        nested=next(e['sha'] for e in tree['tree'] if e['path']=='ggml')
        actual=subprocess.check_output(['git','-C',str(ROOT/'engines/speech.cpp/ggml'),'rev-parse','HEAD'],text=True).strip()
        if nested!=actual:raise RuntimeError('New ggml SHA needs its own snapshot: '+nested)
        shutil.copytree(ROOT/'engines/speech.cpp/ggml',upstream/'ggml',dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.git','build','target','__pycache__'))
        (dest/'source-lock.json').write_text(json.dumps(dict(repository='https://github.com/nyosegawa/speech.cpp',
            commit=sha,ggml=nested,archive_sha256=digest(archive)),indent=2)+'\n')
    for name in ('profile.hpp','speech_worker.cpp'):
        (dest/'benchmark').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'benchmark'/name,dest/'benchmark'/name)
    instrument.ROOT=dest
    instrument.SOURCE=dest/'artifacts/benchmark/source'
    instrument.SOURCE.mkdir(parents=True,exist_ok=True)
    return dest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sha',required=True)
    args=parser.parse_args()
    dest=prepare(args.sha)
    for backend in ('cuda','vulkan'):
        stamp=instrument.SOURCE/f'speech.cpp/.bench-built-{backend}.json'
        if stamp.exists():continue
        instrument.cpp('speech.cpp',backend)
    print('ISOLATED BUILDS READY',dest,flush=True)

if __name__=='__main__':main()
