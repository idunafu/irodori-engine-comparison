"""Freeze all splits before generation, and enumerate deterministic task identities."""
from __future__ import annotations
import json
from common import ASSETS, CONFIG, DATASETS, ROOT, cases, digest, stratified


def inputs(datasets=None):
    enabled=set(DATASETS if datasets is None else datasets)
    if not enabled or enabled-set(DATASETS):
        raise ValueError('Select at least one known evaluation dataset')
    jkyb = []
    source = ASSETS / 'jkyb/data/common_kanji_source.jsonl'
    for line in source.read_text().splitlines() if 'jkyb' in enabled else []:
        b = json.loads(line)
        jkyb.append(dict(id='jkyb:' + b['key'], dataset='jkyb', text=b['text'],
            kind=b['reading_category'], cluster=b['key'].rsplit('_', 1)[0],
            references=[], caption='', benchmark=b))
    if 'jkyb' in enabled and len(jkyb) != 13536:
        raise ValueError(f'Expected 13536 JKYB sentences, got {len(jkyb)}')
    diagnostic_jkyb = stratified(jkyb, CONFIG['diagnostic']['jkyb'],
        lambda x: x['kind'], 'jkyb-diagnostic-v1')
    pilot_jkyb = stratified(diagnostic_jkyb, CONFIG['pilot']['jkyb'],
        lambda x: x['kind'], 'jkyb-pilot-v1')
    speakers = json.loads((ASSETS / 'jvs-index.json').read_text()) if 'jvs' in enabled else []
    # Prefer the corpus's gender metadata; ID deciles are a deterministic fallback.
    diagnostic_sp = stratified(speakers, CONFIG['diagnostic']['jvs'],
        lambda x: x.get('gender') or str((int(x['id'][3:])-1)//10), 'jvs-diagnostic-v1')
    pilot_sp = stratified(diagnostic_sp, CONFIG['pilot']['jvs'],
        lambda x: x.get('gender') or str((int(x['id'][3:])-1)//10), 'jvs-pilot-v1')
    jvs = []
    for sp in speakers:
        targets, held = sp['parallel'][:5], sp['parallel'][5:15]
        if set(x['path'] for x in targets+held) & set(x['path'] for x in sp['references']):
            raise ValueError('JVS leakage')
        for length in ['one', '30', '60', '120']:
            selected = []
            for clip in sp['references']:
                selected.append(clip)
                if length == 'one' or sum(c['seconds'] for c in selected) >= int(length):
                    break
            actual = sum(c['seconds'] for c in selected)
            missing = length != 'one' and actual < int(length)
            for target in targets:
                jvs.append(dict(id=f"jvs:{sp['id']}:{length}:{target['id']}", dataset='jvs',
                    text=target['text'], kind=length, cluster=sp['id'], speaker=sp['id'],
                    references=[str(ROOT / c['path']) for c in selected], caption='',
                    reference_seconds=min(actual,120), reference_missing=missing,
                    held_out=held))
    sb = json.loads((ASSETS / 'speech-bench/prompts/speak-ja-JP.json').read_text())['prompts']
    if len(sb) != 20:
        raise ValueError('The pinned speech-bench prompt count changed')
    reference = str(ROOT / 'artifacts/benchmark/inputs/reference.wav')
    practical = [dict(id='speech-bench:'+p['id'], dataset='practical', text=p['text'],
                     kind=p['kind'], cluster='speech-bench:'+p['id'],
                     references=[reference], caption=CONFIG['caption']) for p in sb]
    for p in json.loads((ROOT / 'benchmark/prompts.json').read_text()):
        practical.append(dict(id='existing:'+p['id'], dataset='practical', text=p['text'],
            kind='existing:'+p['bucket'], cluster='existing:'+p['id'],
            references=[reference], caption=CONFIG['caption']))
    if 'practical' not in enabled:
        practical=[]
    split = dict(diagnostic_jkyb=[x['id'] for x in diagnostic_jkyb],
                 pilot_jkyb=[x['id'] for x in pilot_jkyb],
                 diagnostic_jvs=[x['id'] for x in diagnostic_sp],
                 pilot_jvs=[x['id'] for x in pilot_sp])
    return jkyb, jvs, practical, split


def task_id(item, case, seed, mode='native'):
    return digest([item['id'], case['id'], seed, mode])[:24]


def tasks(datasets=None):
    jkyb, jvs, practical, split = inputs(datasets)
    pilot_j = set(split['pilot_jkyb']); pilot_s = set(split['pilot_jvs'])
    diag_j = set(split['diagnostic_jkyb']); diag_s = set(split['diagnostic_jvs'])
    for profile in ['practical', 'fp32']:
        selected = jkyb+jvs+practical if profile == 'practical' else (
            [x for x in jkyb if x['id'] in diag_j] +
            [x for x in jvs if x['speaker'] in diag_s] + practical)
        for case in cases(profile):
            for item in selected:
                pilot = item['dataset']=='practical' or item['id'] in pilot_j or item.get('speaker') in pilot_s
                for seed in CONFIG['seeds']:
                    spec = dict(item=item, case=case, seed=seed, mode='native')
                    yield task_id(item, case, seed), spec, profile=='practical' and pilot
    # Paired fixed-frame noise diagnostics include both weight profiles, with the same F32 oracle per variant.
    for profile in ['practical', 'fp32']:
        for case in cases(profile):
            oracle = next(c for c in cases('fp32') if c['engine']=='official' and c['variant']==case['variant'])
            for item in practical:
                for seed in CONFIG['seeds']:
                    spec = dict(item=item, case=case, seed=seed, mode='common',
                                length_source=task_id(item, oracle, seed))
                    yield task_id(item, case, seed, 'common'), spec, False
