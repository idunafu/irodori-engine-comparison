"""Rebuild automatic metrics and paired cluster intervals from the saved campaign."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from common import ASSETS, BASE, CONFIG, HERE, ROOT, atomic_json, digest, environment, sha256
from dataset import task_id
from store import Store
from worker import Process


def centroids(directory):
    """L2-normalize every held-out embedding, average ten, then normalize the centroid."""
    import sherpa_onnx
    import soxr
    index=json.loads((ASSETS/'jvs-index.json').read_text())
    model=ASSETS/'models/campplus'/CONFIG['models']['campplus']['file']
    signature=digest([sha256(model),[(s['id'],[(c['path'],c['sha256']) for c in s['parallel'][5:15]]) for s in index]])
    path=Path(directory)/'speaker-centroids.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['signature']!=signature:
            raise ValueError('Speaker scoring inputs changed')
        return saved
    extractor=sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model),num_threads=8,provider='cpu',debug=False))
    natural={}
    for speaker in index:
        embeddings=[]
        for clip in speaker['parallel'][5:15]:
            wav=ROOT/clip['path']
            if sha256(wav)!=clip['sha256']:
                raise ValueError('Natural scoring audio changed')
            audio,sr=sf.read(wav,dtype='float32',always_2d=True)
            audio=audio.mean(axis=1)
            if sr!=16000:
                audio=soxr.resample(audio,sr,16000,quality='HQ')
            stream=extractor.create_stream();stream.accept_waveform(16000,audio);stream.input_finished()
            if not extractor.is_ready(stream):
                raise ValueError('Held-out reference too short')
            emb=np.asarray(extractor.compute(stream));emb=emb/np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        natural[speaker['id']]=embeddings
    centers={s:(lambda x:(x/np.linalg.norm(x)).tolist())(np.mean(v,axis=0)) for s,v in natural.items()}
    value=dict(signature=signature,centroids=centers,natural_embeddings=natural)
    atomic_json(path,value)
    return value


def bootstrap_difference(a,b,replicates=2000):
    """Resample paired clusters, preserving each metric's numerator and denominator."""
    keys=sorted(a.keys()&b.keys())
    if not keys:
        return None
    left=np.asarray([a[k] for k in keys],dtype=float)
    right=np.asarray([b[k] for k in keys],dtype=float)
    if left[:,1].sum()==0 or right[:,1].sum()==0:
        return None
    rng=np.random.default_rng(20261016)
    draws=[]
    for offset in range(0,replicates,64):
        indices=rng.integers(0,len(keys),(min(64,replicates-offset),len(keys)))
        la=left[indices].sum(axis=1); rb=right[indices].sum(axis=1)
        valid=(la[:,1]>0)&(rb[:,1]>0)
        draws.extend((la[valid,0]/la[valid,1]-rb[valid,0]/rb[valid,1]).tolist())
    return dict(difference=float(left[:,0].sum()/left[:,1].sum()-right[:,0].sum()/right[:,1].sum()),
                low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),clusters=len(keys))


def write_csv(path,rows):
    rows=list(rows)
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def report(directory):
    directory=Path(directory)
    store=Store(directory);db=store.db
    dest=directory/'report';dest.mkdir(exist_ok=True)
    sys.path.insert(0,str(ASSETS/'jkyb-evaluator/src'))
    from jkyb_eval.models import BenchmarkRow
    from jkyb_eval.metrics import score_row
    from jkyb_eval.normalization import canonicalize_text
    from rapidfuzz.distance import Levenshtein
    has_jvs=db.execute("SELECT 1 FROM tasks WHERE dataset='jvs' LIMIT 1").fetchone() is not None
    centers=centroids(directory)['centroids'] if has_jvs else {}
    speaker_ids=sorted(centers)
    center_matrix=np.asarray([centers[x] for x in speaker_ids])
    metrics=defaultdict(lambda:defaultdict(lambda:np.zeros(2)))
    clusters=defaultdict(lambda:defaultdict(lambda:np.zeros(2)))
    consistency=defaultdict(list)
    split=json.loads((directory/'splits.json').read_text())
    diag_j=set(split['diagnostic_jkyb']);diag_s=set(split['diagnostic_jvs'])
    def add(spec,name,value,weight=1,success_only=False):
        item,case=spec['item'],spec['case']
        if not np.isfinite(value) or not np.isfinite(weight):
            raise ValueError('Non-finite report metric')
        for kind in ['all',item['kind']]:
            key=(item['dataset'],case['id'],spec['mode'],kind,name,success_only)
            metrics[key][spec['seed']]+=np.array([value,weight])
        if not success_only:
            key=(item['dataset'],case['id'],spec['mode'],name,'all')
            clusters[key][item['cluster']]+=np.array([value,weight])
            if item['id'] in diag_j or item.get('speaker') in diag_s or item['dataset']=='practical':
                key=(*key[:-1],'diagnostic')
                clusters[key][item['cluster']]+=np.array([value,weight])
    node=Process(['node',str(HERE/'speech_bench_worker.mjs')],environment(),directory/'report-node.log')
    failures=[]
    detail_path=dest/'per-input.jsonl'
    try:
        with detail_path.open('w') as detail:
            for row in db.execute("SELECT * FROM tasks WHERE status!='pending' ORDER BY rowid"):
                spec=json.loads(row['spec']);item=spec['item'];case=spec['case']
                scores={s['scorer']:(s['status'],json.loads(s['result'])) for s in db.execute(
                    'SELECT scorer,status,result FROM scores WHERE task_id=?',(row['id'],))}
                good=row['status']=='ok'
                out=json.loads(row['result'])
                add(spec,'generation_failure_rate',int(not good))
                values={}
                if not good:
                    failures.append(dict(id=row['id'],dataset=item['dataset'],case=case['id'],
                                         seed=spec['seed'],mode=spec['mode'],category=out.get('category','other'),error=out.get('error')))
                if item['dataset']=='jkyb' and all(s in scores and scores[s][0]=='ok' for s in ['kana','whisper']):
                    scored=score_row(BenchmarkRow.from_dict(item['benchmark'],location=item['id']),
                        scores['kana'][1].get('transcript',''),text_prediction=scores['whisper'][1].get('transcript',''),score_text=True)
                    values={k:float(scored[k]) for k in ['target_exact','relaxed_target_exact',
                        'target_kana_cer','target_kana_cer_clipped','sentence_kana_cer','text_cer']}
                    for name,value in values.items():
                        add(spec,name,value)
                        if good:add(spec,name,value,success_only=True)
                if item['dataset']=='jvs':
                    embedding=scores.get('campplus',('',{}))[1].get('embedding')
                    top1=0
                    if embedding is not None:
                        similarities=center_matrix@np.asarray(embedding)
                        values['camplus_cosine']=float(similarities[speaker_ids.index(item['speaker'])])
                        add(spec,'camplus_cosine',values['camplus_cosine'])
                        top1=int(speaker_ids[int(np.argmax(similarities))]==item['speaker'])
                    values['speaker_top1']=top1
                    add(spec,'speaker_top1',top1)
                    if 'whisper' in scores and scores['whisper'][0]=='ok':
                        ref=canonicalize_text(item['text'])
                        pred=canonicalize_text(scores['whisper'][1].get('transcript',''))
                        values['whisper_cer']=Levenshtein.distance(ref,pred)/max(1,len(ref))
                        add(spec,'whisper_cer',values['whisper_cer'])
                if item['dataset']=='practical':
                    pitch=None
                    if good:
                        acoustic=node.request(dict(id=row['id'],wav=row['wav']))
                        if acoustic['type']!='result':raise RuntimeError(str(acoustic))
                        pitch=acoustic.get('median_pitch_hz')
                        values.update(acoustic)
                        for name in ['seconds_per_character','leading_silence_seconds',
                                     'trailing_silence_seconds','clipping_fraction']:
                            add(spec,name,out[name])
                    for scorer in ['qwen','reazon']:
                        if scorer not in scores or scores[scorer][0]!='ok':continue
                        heard=node.request(dict(id=row['id'],text=item['text'],transcript=scores[scorer][1].get('transcript','')))
                        if heard['type']!='result':raise RuntimeError(str(heard))
                        length=heard['heard_reference_length']
                        add(spec,scorer+'_heard_error_rate',heard['heard_errors'],length)
                        add(spec,scorer+'_raw_cer',heard['heard_raw_errors'],length)
                        add(spec,scorer+'_heard_as_said',int(heard['heard_as_said']))
                        add(spec,scorer+'_breakdown_rate',int(heard['heard_errors']/max(1,length)>.3))
                        values[scorer]=heard
                    embedding=scores.get('eres2net',('',{}))[1].get('embedding')
                    consistency[(case['id'],spec['mode'],spec['seed'])].append((item['id'],embedding,pitch))
                if spec['mode']=='common' and good:
                    oracle=dict(engine='official',variant=case['variant'],precision='fp32',id=f"official-{case['variant']}-fp32")
                    reference=db.execute('SELECT wav FROM tasks WHERE id=? AND status=?',
                        (task_id(item,oracle,spec['seed'],'common'),'ok')).fetchone()
                    if reference:
                        a,_=sf.read(row['wav'],dtype='float32');b,_=sf.read(reference['wav'],dtype='float32')
                        if len(a)==len(b) and np.std(a)>0 and np.std(b)>0:
                            values['pcm_correlation']=float(np.corrcoef(a,b)[0,1])
                            values['pcm_relative_l2']=float(np.linalg.norm(a-b)/max(1e-12,np.linalg.norm(b)))
                            add(spec,'pcm_correlation',values['pcm_correlation'])
                            add(spec,'pcm_relative_l2',values['pcm_relative_l2'])
                detail.write(json.dumps(dict(id=row['id'],item=item['id'],dataset=item['dataset'],kind=item['kind'],
                    case=case['id'],profile=case['profile'],mode=spec['mode'],seed=spec['seed'],status=row['status'],
                    wav_sha256=row['wav_sha256'],metrics=values),ensure_ascii=False,allow_nan=False)+'\n')
        for (case,mode,seed),takes in consistency.items():
            sim=[]
            for i,(item,a,_) in enumerate(takes):
                if a is None:continue
                for other,b,_ in takes[i+1:]:
                    if other!=item and b is not None:sim.append(float(np.dot(a,b)))
            pitch=[12*np.log2(p) for _,_,p in takes if p is not None and p>0]
            if sim:metrics[('practical',case,mode,'all','same_voice_cosine',False)][seed]+=np.array([np.mean(sim),1])
            if len(pitch)>1:metrics[('practical',case,mode,'all','pitch_spread_semitones',False)][seed]+=np.array([np.std(pitch),1])
        per_seed=[];summaries=[]
        for key,seeds in sorted(metrics.items()):
            dataset,case,mode,kind,name,success=key
            base=dict(dataset=dataset,case=case,mode=mode,kind=kind,metric=name,success_only=success)
            values=[]
            for seed,(numerator,denominator) in sorted(seeds.items()):
                if denominator==0:continue
                value=float(numerator/denominator);values.append(value)
                per_seed.append(dict(**base,seed=seed,value=value,denominator=denominator))
            if values:summaries.append(dict(**base,mean=float(np.mean(values)),population_std=float(np.std(values)),seed_count=len(values)))
        comparisons=[]
        for (dataset,case,mode,metric,subset),sample in sorted(clusters.items()):
            engine,variant,precision=case.split('-')
            official='official-'+variant+('-fp32' if precision=='fp32' else '-bf16')
            contrasts=[('vs_official',official,subset)] if case!=official and subset=='all' else []
            if precision=='fp32' and subset=='diagnostic':
                practical={'official':'bf16','onnx':'fp16','speech':'f16'}[engine]
                contrasts.append(('fp32_minus_practical',f'{engine}-{variant}-{practical}','diagnostic'))
            for name,baseline,target_subset in contrasts:
                other=clusters.get((dataset,baseline,mode,metric,target_subset),{})
                interval=bootstrap_difference(sample,other,CONFIG['bootstrap_replicates'])
                if interval:comparisons.append(dict(dataset=dataset,case=case,baseline=baseline,mode=mode,
                    metric=metric,comparison=name,**interval))
        write_csv(dest/'per-seed.csv',per_seed);write_csv(dest/'summary.csv',summaries)
        write_csv(dest/'comparisons.csv',comparisons);write_csv(dest/'failures.csv',failures)
        atomic_json(dest/'coverage.json',dict(generation=store.counts(),failure_categories=dict(Counter(r['category'] for r in failures)),scores=[dict(x) for x in db.execute(
            'SELECT scorer,status,COUNT(*) AS count FROM scores GROUP BY scorer,status')]))
        manifest=directory/'manifest.json'
        selection=json.loads(manifest.read_text()).get('datasets') if manifest.exists() else None
        lines=['# Irodori-TTS 自動品質評価','',
            '評価対象: '+', '.join(selection or sorted({r['dataset'] for r in summaries}))+'.',
            'JSUT・試聴・caption指示追従は対象外。固定版の実装を同一評価器で比較。',
            '値は5 seedの平均と母標準偏差。生成失敗は読み評価で空出力、話者Top-1で不正解として扱う。',
            'CAM++類似度は採点可能な音声のみ。自動指標だけで聴感上の自然さや同等性は判定しない。','',
            '| データ | 実装 | モード | 指標 | 平均 | 母標準偏差 |','|---|---|---|---|---:|---:|']
        primary={'target_exact','relaxed_target_exact','target_kana_cer','sentence_kana_cer','text_cer',
            'camplus_cosine','speaker_top1','qwen_heard_error_rate','reazon_heard_error_rate',
            'qwen_breakdown_rate','reazon_breakdown_rate','same_voice_cosine','pitch_spread_semitones','generation_failure_rate'}
        for r in summaries:
            if r['kind']=='all' and not r['success_only'] and r['metric'] in primary:
                lines.append(f"| {r['dataset']} | {r['case']} | {r['mode']} | {r['metric']} | {r['mean']:.6f} | {r['population_std']:.6f} |")
        lines+=['','公式版との差・95% paired cluster bootstrap区間は comparisons.csv、種別別・成功例のみの集計は summary.csv を参照。',
                'JKYBは漢字読みの組、JVSは話者を再標本化する。実用文は文章を単位とし、異なる文章間の声の一貫性を測る。',
                '生成失敗の分類・件数は failures.csv、採点可能件数は coverage.json と per-seed.csv に保存。',
                '既存速度比較とコード・モデル・runtimeを共有するが、生成条件や入力集合が異なる速度値は直接結合しない。','']
        (dest/'RESULTS.ja.md').write_text('\n'.join(lines))
    finally:
        node.close();store.db.close()


if __name__=='__main__':
    report(Path(sys.argv[1]))
