"""Audit coverage, GPU preflight, lengths and actual forward instrumentation."""
import json
from collections import Counter
from pathlib import Path
from report import latest_records,aggregate
from run import matrix,CFG,PROMPTS,OUT

def main():
    records=latest_records(OUT/'results.jsonl')
    rows=aggregate(records)
    preflight={r['key']:r for r in map(json.loads,(OUT/'preflight.jsonl').read_text().splitlines())}
    errors=[]
    cases=matrix()
    failures=[r for r in records if r['status']=='failed']
    for case in cases:
        name=case['case']
        check=preflight.get(name+'/preflight')
        if not check or check['status']!='ok':errors.append(name+': GPU preflight missing/failed')
        cold=[r for r in records if r.get('kind')=='cold' and r['case']['case']==name and r['status']=='ok']
        if len(cold)!=3:errors.append(name+f': cold count {len(cold)} != 3')
        complete=[r for r in rows if r['case']==name and r['complete']]
        rejected=[r for r in failures if r['case']['case']==name and r.get('mode')]
        if len(complete)+len(rejected)!=130:errors.append(name+f': coverage {len(complete)}+{len(rejected)} != 130')
        for cache in CFG['cache_profiles']:
            main=[r for r in complete if r['cache']==cache and r['mode']=='matched']
            if len(main)!=20:errors.append(name+'/'+cache+f': matched count {len(main)} != 20')
    for r in records:
        if r['status']!='ok' or r.get('kind') not in ('wall','profile','cold','shape_first'):continue
        result=r['result']
        if result.get('sample_rate')!=48000 or not result.get('finite') or result.get('rms',0)<=1e-8:
            errors.append(r['key']+': invalid PCM')
        if r.get('seconds') and abs(result['samples']-round(r['seconds']*48000))>1:
            errors.append(r['key']+': fixed length mismatch')
        if r.get('kind')=='profile':
            events=result.get('events',[])
            names={e['stage'] for e in events}
            required={'sampling','sampling_loop','text_tokenize'}
            if not required<=names:errors.append(r['key']+': missing '+','.join(required-names))
            count=sum(e['stage']=='dit_forward' for e in events)
            if count<CFG['steps'][r['case']['variant']]:errors.append(r['key']+': insufficient forward calls')
            if any(e['ms']<0 for e in events):errors.append(r['key']+': negative interval')
    data=dict(expected_groups=3120,complete_groups=sum(r['complete'] for r in rows),
              failed_groups=sum(bool(r.get('mode')) for r in failures),errors=errors,
              statuses=dict(Counter(r['status'] for r in records)))
    (OUT/'validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
