"""Validate A/B coverage, input preservation, GPU selection, and saved PCM (no quality scoring)."""
import collections
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import speech_ab as ab
from speech_ab_build import digest

def main():
    records=ab.read_records();rows=ab.summarize(records);errors=[]
    cfg=json.loads((ab.OUT/'config.json').read_text())
    decision=json.loads((ab.OUT/'decision.json').read_text())
    expected=32+len(decision['affected'])*80
    if len(rows)!=expected or not all(r['complete'] for r in rows):errors.append('Incomplete input coverage')
    counts=collections.Counter(r['kind'] for r in records)
    if counts['wall']!=expected*30 or counts['profile']!=expected*30:errors.append('Wrong request counts')
    if counts['preflight']!=8 or counts['cold']!=24+6*len(decision['affected']):errors.append('Wrong independent worker counts')
    for r in records:
        if r['status']!='ok':errors.append(r['key']+': failure');continue
        if r['kind']=='preflight':
            if r['profile_relative_error']>1e-4:errors.append(r['key']+': instrumentation changed PCM')
        if r['kind'] in ('cold','preflight') and 'RTX 5090' not in r['load']['device']:errors.append(r['key']+': unexpected GPU')
        if r['kind'] not in ('wall','profile'):continue
        v=r['result'];seconds=r['seconds']
        if not v['finite'] or v['rms']<=1e-8 or v['sample_rate']!=48000 or v['samples']!=round(seconds*48000):errors.append(r['key']+': invalid PCM')
        try:ab.validate_result(v,seconds,r['kind']=='profile',r['case']['variant'])
        except Exception as e:errors.append(r['key']+': '+str(e))
        if any(e['ms']<0 for e in v.get('events',[])):errors.append(r['key']+': negative timing')
    preserved=ab.preservation()==json.loads((ab.OUT/'preservation.json').read_text())
    if not preserved:errors.append('Baseline artifacts changed')
    for name,sha in cfg['inputs_sha256'].items():
        if digest(ab.ROOT/name)!=sha:errors.append('Input changed: '+name)
    head=subprocess.check_output(['git','-C',str(ab.ROOT/'engines/speech.cpp'),'rev-parse','HEAD'],text=True).strip()
    if head!=ab.BASELINE:errors.append('Pinned submodule changed')
    pairs=[]
    for phase,models,prompts in [('screen',[(m,b) for m in ('rf','mf') for b in ('cuda','vulkan')],cfg['screen_prompts']),
                                ('expanded',decision['affected'],ab.harness.PROMPTS)]:
        for m,b in models:
            for p in prompts:
                for seconds in (8,24):
                    arrays=[]
                    for version in ('old','new'):
                        file=ab.OUT/'wav'/f'{phase}-{version}-{m}-{b}-{p["id"]}-{seconds}.wav'
                        info=sf.info(file);audio,rate=sf.read(file,dtype='float32')
                        if info.channels!=1 or rate!=48000 or len(audio)!=seconds*48000 or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2))<=1e-8:errors.append(str(file)+': invalid WAV')
                        arrays.append(audio)
                    same=bool(np.array_equal(arrays[0].view(np.uint32),arrays[1].view(np.uint32)))
                    pairs.append(dict(phase=phase,variant=m,backend=b,prompt=p['id'],seconds=seconds,bit_equal=same,
                        max_abs=float(np.max(np.abs(arrays[0]-arrays[1]))),
                        relative_error=0. if same else float(np.linalg.norm(arrays[0]-arrays[1])/max(np.linalg.norm(arrays[0]),1e-12))))
    (ab.OUT/'pcm-parity.json').write_text(json.dumps(pairs,indent=2)+'\n')
    source=json.loads((ab.SOURCE/'source-lock.json').read_text())
    binaries={v+'-'+b:digest(ab.command(ab.case_for(v,'rf',b))[0][0]) for v in ('old','new') for b in ('cuda','vulkan')}
    build_settings={b:json.loads((ab.SOURCE/f'artifacts/benchmark/source/speech.cpp/.bench-built-{b}.json').read_text()) for b in ('cuda','vulkan')}
    (ab.OUT/'provenance.json').write_text(json.dumps(dict(baseline=ab.BASELINE,latest=source,binary_sha256=binaries,
        latest_builds=build_settings,measurement_config=cfg,
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv,noheader'],text=True).strip()),ensure_ascii=False,indent=2)+'\n')
    validation=dict(expected_conditions=expected,complete_conditions=sum(r['complete'] for r in rows),request_counts=dict(counts),
        original_results_preserved=preserved,original_submodule_preserved=head==ab.BASELINE,
        wav_count=len(pairs)*2,pcm_pairs=len(pairs),bit_equal_pairs=sum(p['bit_equal'] for p in pairs),errors=errors)
    (ab.OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
