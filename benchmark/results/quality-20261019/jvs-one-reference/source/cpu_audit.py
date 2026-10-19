from pathlib import Path
import json
import sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[5]
sys.path.insert(0,str(ROOT/'benchmark/quality'))
from common import atomic_json,environment
from worker import Process,command

cmd,_=command(dict(engine='speech',variant='mf',precision='fp32'))
w=Process([str(OUT/'codec-audit'),cmd[1],'cpu'],environment(),OUT/'codec-cpu.log')
rows=[]
try:
    for sp in json.loads((OUT/'manifest.json').read_text())['speakers']:
        d=OUT/'features/mf'/sp
        req=dict(id=sp,pcm=str(d/'official-pcm.f32'),windowed=str(d/'cpu-windowed.f32'),whole=str(d/'cpu-whole.f32'))
        result=w.request(req)
        if result.get('type')!='result':raise RuntimeError(result)
        rows.append(dict(**req,result=result));atomic_json(OUT/'cpu-codec-audit.json',rows)
finally:w.close()
