"""Clone pinned package sets into new environments, then change only cuDNN."""
import json
import os
from pathlib import Path
import subprocess
import argparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/mf-runtime-cause-20261015'
DEST=ROOT/'environments/mf-runtime-cause-20261015'

def main():
    import mf_cause
    mf_cause.prepare()
    DEST.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,UV_CACHE_DIR=str(ROOT/'.cache/uv'),XDG_DATA_HOME=str(ROOT/'.cache/data'),
             UV_LINK_MODE='hardlink',PYTHONDONTWRITEBYTECODE='1')
    original=(ROOT/'environments/runtime-ab-20261015/Irodori-TTS-requirements.txt').read_text()
    parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+',choices=['cudnn9241','old924'],default=['cudnn9241','old924'])
    args=parser.parse_args()
    commands=[]
    for name in args.names:
        target=ROOT/f'.venvs/mf-runtime-cause-20261015/{name}'
        requirements=DEST/f'{name}-requirements.txt'
        if name=='old924':
            source=requirements.read_text() if requirements.exists() else (ROOT/'artifacts/runtime-ab-20261015/Irodori-TTS-original-packages.txt').read_text()
            source='\n'.join(line for line in source.splitlines() if not line.startswith(('-e ','irodori-tts')))+'\n'
            source=source.replace('nvidia-cudnn-cu12==9.10.2.21','nvidia-cudnn-cu12==9.24.0.43')
        else:
            version={'cudnn9241':'9.24.1.1'}[name]
            source=original.replace('nvidia-cudnn-cu13==9.24.0.43',f'nvidia-cudnn-cu13=={version}')
        requirements.write_text(source)
        cmd=['uv','venv','--python','3.10.20',str(target)]
        if not (target/'bin/python').exists():
            subprocess.run(cmd,env=env,check=True);commands.append(cmd)
        # Intentional dependency override in the new diagnostic environment only.
        # All listed packages, including torch, remain exactly frozen otherwise.
        cmd=['uv','pip','install','--python',str(target/'bin/python'),'--no-deps','--index',
             'https://download.pytorch.org/whl/'+('cu128' if name=='old924' else 'cu130'),'--index-strategy','unsafe-best-match','-r',str(requirements)]
        subprocess.run(cmd,env=env,check=True);commands.append(cmd)
        # An unqualified == pin also accepts CUDA-local versions (PEP 440).
        # Force the original PyPI artifacts for packages whose original version had no + suffix.
        plain=[line for line in source.splitlines() if line.startswith(('torchao==','torchcodec==')) and '+' not in line]
        if plain:
            cmd=['uv','pip','install','--python',str(target/'bin/python'),'--no-deps','--index','https://pypi.org/simple']
            for line in plain:cmd.extend(['--reinstall-package',line.split('==')[0]])
            cmd.extend(plain);subprocess.run(cmd,env=env,check=True);commands.append(cmd)
        freeze=subprocess.check_output(['uv','pip','freeze','--python',str(target/'bin/python')],env=env,text=True)
        (OUT/f'{name}-packages.txt').write_text(freeze)
        print('READY',name,flush=True)
    with (OUT/'setup-commands.jsonl').open('a') as f:
        for command in commands:f.write(json.dumps(command)+'\n')
    assert mf_cause.snapshot()==json.loads((OUT/'preservation.json').read_text())

if __name__=='__main__':main()
