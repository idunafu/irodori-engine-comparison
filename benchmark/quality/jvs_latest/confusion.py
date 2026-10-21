"""Reconstruct all speaker decisions and paired errors from saved CAM++ embeddings."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import ROOT,ASSETS,atomic_json,sha256
from report import write_csv

RUN=ROOT/'artifacts/quality/runs/20261019-jvs-one-latest-eacabcd'
OUT=RUN/'analysis/confusion'


def classify(embedding,truth,ids,matrix):
    scores=matrix@np.asarray(embedding,dtype=float)
    i=ids.index(truth);rank=np.argsort(-scores,kind='stable');other=scores.copy();other[i]=-np.inf
    rival=int(np.argmax(other));pred=int(np.argmax(scores))
    return dict(predicted=ids[pred],correct=pred==i,true_similarity=float(scores[i]),
                rival=ids[rival],rival_similarity=float(scores[rival]),margin=float(scores[i]-scores[rival]),
                true_rank=int(1+np.count_nonzero(scores>scores[i])),
                top5=';'.join(ids[int(j)] for j in rank[:5]),
                top5_scores=';'.join(f'{scores[j]:.9f}' for j in rank[:5])),scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    centers=json.loads((RUN/'speaker-centroids.json').read_text())['centroids'];ids=sorted(centers)
    matrix=np.array([centers[s] for s in ids]);index={r['id']:r for r in json.loads((ASSETS/'jvs-index.json').read_text())}
    conn=sqlite3.connect(f'file:{RUN}/results.sqlite3?mode=ro',uri=True)
    records=[];score_arrays=defaultdict(list);meta_arrays=defaultdict(list)
    query='''SELECT t.id,t.case_id,t.spec,t.result,t.wav_sha256,s.result,s.wav_sha256
             FROM tasks t JOIN scores s ON s.task_id=t.id WHERE s.scorer='campplus' AND s.status='ok' ORDER BY t.rowid'''
    for key,case,spec,result,wav_hash,score,score_hash in conn.execute(query):
        assert wav_hash==score_hash
        spec=json.loads(spec);item=spec['item'];result=json.loads(result);score=json.loads(score)
        assert item['kind']=='one' and len(item['references'])==1
        decision,values=classify(score['embedding'],item['speaker'],ids,matrix)
        records.append(dict(id=key,case=case,variant=spec['case']['variant'],speaker=item['speaker'],
                            gender=index[item['speaker']]['gender'],item=item['id'],seed=spec['seed'],
                            reference_seconds=item['reference_seconds'],frames=result['frames'],
                            **decision))
        score_arrays[case].append(values);meta_arrays[case].append(key)
    conn.close()
    assert len(records)==15000
    lookup={(r['case'],r['item'],r['seed']):r for r in records}
    write_csv(out/'decisions.csv',records)
    np.savez_compressed(out/'all-centroid-scores.npz',speaker_ids=np.array(ids),
                        **{case:np.array(v) for case,v in score_arrays.items()},
                        **{case+'-task-ids':np.array(v) for case,v in meta_arrays.items()})
    speaker_rows=[];pair_rows=[];transitions=[];paired=[];summaries=[];rank_summaries=[]
    for case in sorted(score_arrays):
        group=[r for r in records if r['case']==case]
        summaries.append(dict(case=case,takes=len(group),correct=sum(r['correct'] for r in group),
                              top1=float(np.mean([r['correct'] for r in group])),
                              top3=float(np.mean([r['true_rank']<=3 for r in group])),
                              top5=float(np.mean([r['true_rank']<=5 for r in group])),
                              true_similarity=float(np.mean([r['true_similarity'] for r in group])),
                              mean_margin=float(np.mean([r['margin'] for r in group]))))
        for speaker in ids:
            takes=[r for r in group if r['speaker']==speaker];assert len(takes)==25
            errors=Counter(r['predicted'] for r in takes if not r['correct'])
            speaker_rows.append(dict(case=case,speaker=speaker,gender=index[speaker]['gender'],
                                     reference_seconds=takes[0]['reference_seconds'],correct=25-sum(errors.values()),errors=sum(errors.values()),
                                     mean_similarity=float(np.mean([r['true_similarity'] for r in takes])),
                                     mean_margin=float(np.mean([r['margin'] for r in takes])),
                                     wrong_destinations=';'.join(f'{s}:{n}' for s,n in errors.most_common())))
            for predicted,n in errors.items():pair_rows.append(dict(case=case,speaker=speaker,predicted=predicted,count=n))
        bad=[r for r in group if not r['correct']]
        for threshold in [.005,.01,.02,.05]:
            rank_summaries.append(dict(case=case,threshold_pp=threshold*100,errors=len(bad),
                                       true_within_threshold_of_winner=sum(r['margin']>=-threshold for r in bad)))
    speaker_lookup={(r['case'],r['speaker']):r for r in speaker_rows}
    gaps=[];confusions=[];concentration=[]
    for variant in ['mf','rf']:
        official='official-'+variant+'-bf16';cpp='speech-'+variant+'-f16';previous='previous-'+variant+'-f16'
        counts=Counter();cases=defaultdict(list)
        for r in [r for r in records if r['case']==official]:
            other=lookup[cpp,r['item'],r['seed']]
            label=('both_correct' if other['correct'] else 'cpp_only_error') if r['correct'] else ('official_only_error' if other['correct'] else 'both_wrong')
            counts[label]+=1;cases[label].append((r,other))
            paired.append(dict(variant=variant,speaker=r['speaker'],item=r['item'],seed=r['seed'],outcome=label,
                               official_predicted=r['predicted'],cpp_predicted=other['predicted'],
                               official_true_similarity=r['true_similarity'],cpp_true_similarity=other['true_similarity'],
                               true_change=other['true_similarity']-r['true_similarity'],
                               rival_change=other['rival_similarity']-r['rival_similarity'],
                               official_margin=r['margin'],cpp_margin=other['margin'],
                               official_frames=r['frames'],cpp_frames=other['frames']))
        for label,group in cases.items():
            transitions.append(dict(variant=variant,outcome=label,count=len(group),
                                    mean_true_change_pp=float(np.mean([b['true_similarity']-a['true_similarity'] for a,b in group]))*100,
                                    mean_rival_change_pp=float(np.mean([b['rival_similarity']-a['rival_similarity'] for a,b in group]))*100,
                                    mean_cpp_margin_pp=float(np.mean([b['margin'] for a,b in group]))*100,
                                    cpp_within_1pp=sum(b['margin']>=-.01 for a,b in group)))
        for speaker in ids:
            a=speaker_lookup[official,speaker];b=speaker_lookup[cpp,speaker];c=speaker_lookup[previous,speaker]
            affected=[r for r in paired if r['variant']==variant and r['speaker']==speaker]
            gaps.append(dict(variant=variant,speaker=speaker,gender=a['gender'],reference_seconds=a['reference_seconds'],
                             official_correct=a['correct'],cpp_correct=b['correct'],previous_correct=c['correct'],
                             excess_errors=b['errors']-a['errors'],cpp_only_errors=sum(r['outcome']=='cpp_only_error' for r in affected),
                             official_only_errors=sum(r['outcome']=='official_only_error' for r in affected),
                             official_similarity=a['mean_similarity'],cpp_similarity=b['mean_similarity'],
                             official_wrong=a['wrong_destinations'],cpp_wrong=b['wrong_destinations']))
        destinations={(r['speaker'],r['predicted']) for r in pair_rows if r['case'] in [official,cpp,previous]}
        pc=Counter({(r['case'],r['speaker'],r['predicted']):r['count'] for r in pair_rows})
        for speaker,predicted in destinations:
            a,b,c=(pc[k,speaker,predicted] for k in [official,cpp,previous])
            confusions.append(dict(variant=variant,speaker=speaker,predicted=predicted,
                                   same_gender=index[speaker]['gender']==index[predicted]['gender'],
                                   official_count=a,cpp_count=b,previous_count=c,excess=b-a))
        ordered=sorted([r for r in gaps if r['variant']==variant],key=lambda r:(-r['excess_errors'],r['speaker']))
        concentration.append(dict(variant=variant,net_excess=sum(r['excess_errors'] for r in ordered),
                                  positive_excess=sum(max(r['excess_errors'],0) for r in ordered),
                                  improvement_offset=sum(min(r['excess_errors'],0) for r in ordered),
                                  speakers_worse=sum(r['excess_errors']>0 for r in ordered),speakers_better=sum(r['excess_errors']<0 for r in ordered),
                                  top2_excess=sum(r['excess_errors'] for r in ordered[:2]),top5_excess=sum(r['excess_errors'] for r in ordered[:5])))
    write_csv(out/'summary.csv',summaries);write_csv(out/'per-speaker.csv',speaker_rows)
    write_csv(out/'pairs-by-case.csv',pair_rows);write_csv(out/'paired-outcomes.csv',paired)
    write_csv(out/'paired-outcome-summary.csv',transitions);write_csv(out/'error-margins.csv',rank_summaries)
    gaps.sort(key=lambda r:(r['variant'],-r['excess_errors'],r['speaker']))
    confusions.sort(key=lambda r:(r['variant'],-r['excess'],-r['cpp_count'],r['speaker'],r['predicted']))
    write_csv(out/'speaker-gaps.csv',gaps);write_csv(out/'confusion-comparison.csv',confusions)
    write_csv(out/'concentration.csv',concentration)
    total=Counter()
    for row in gaps:total[row['speaker']]+=row['excess_errors']
    affected=[s for s,_ in sorted(total.items(),key=lambda v:(-v[1],v[0]))[:6]]
    control=[]
    for gender in ['M','F']:
        control.append(next(s for s in ids if index[s]['gender']==gender and s not in affected and
                            all(speaker_lookup[case,s]['correct']==25 for case in ['official-mf-bf16','official-rf-bf16','speech-mf-f16','speech-rf-f16'])))
    selection=dict(affected=affected,controls=control,speakers=affected+control,
                   criterion='Top six speakers by summed MF/RF excess errors over official, plus first perfect male/female controls; fixed before intervention outputs')
    atomic_json(out/'selection.json',selection)
    natural=out/'natural-scores.json'
    natural_rows=[]
    if natural.exists():
        for r in json.loads(natural.read_text()):
            if r.get('failed'):continue
            result,_=classify(r['embedding'],r['speaker'],ids,matrix)
            natural_rows.append(dict(role=r['role'],speaker=r['speaker'],clip=r['clip'],**result))
        write_csv(out/'natural-decisions.csv',natural_rows)
    metadata=dict(source_run=str(RUN),rows=len(records),source_sha256={str(p):sha256(p) for p in
                   [RUN/'results.sqlite3',RUN/'speaker-centroids.json',ASSETS/'jvs-index.json',Path(__file__)]},
                  selection=selection,concentration=concentration,summary=summaries)
    atomic_json(out/'analysis.json',metadata)
    lines=['# 最新cppと公式版の話者混同の比較','',
           '保存済みCAM++特徴量と同じ100話者centroidから、全15,000音声の予測話者・順位・スコア差を再計算。',
           '各話者は各実装・方式につき5文章×5 seed＝25音声。話者IDはJVSのID。','']
    for variant in ['mf','rf']:
        counts={r['outcome']:r['count'] for r in transitions if r['variant']==variant}
        lines += [variant.upper(),'',f"公式のみ正解 {counts.get('cpp_only_error',0)}、cppのみ正解 {counts.get('official_only_error',0)}、両方正解 {counts.get('both_correct',0)}、両方不正解 {counts.get('both_wrong',0)}。",'',
                  '| 正解話者 | 公式正解/25 | cpp正解/25 | cppの誤認先（件数） | 誤り純増 |','|---|---:|---:|---|---:|']
        for r in [r for r in gaps if r['variant']==variant][:12]:
            lines.append(f"| {r['speaker']} | {r['official_correct']} | {r['cpp_correct']} | {r['cpp_wrong']} | {r['excess_errors']:+d} |")
        lines+=['','| 正解→誤認先 | 公式の誤認数 | cppの誤認数 | 増減 |','|---|---:|---:|---:|']
        for r in [r for r in confusions if r['variant']==variant][:15]:
            lines.append(f"| {r['speaker']} → {r['predicted']} | {r['official_count']} | {r['cpp_count']} | {r['excess']:+d} |")
        stat=next(r for r in concentration if r['variant']==variant)
        lines += ['',f"誤り純増 {stat['net_excess']}件。上位2話者の純増は{stat['top2_excess']}件、上位5話者は{stat['top5_excess']}件。悪化{stat['speakers_worse']}話者、改善{stat['speakers_better']}話者。",'']
    if natural_rows:
        lines += ['入力・実音声の識別（生成音声と同じ100話者centroid）','', '| 音声 | 正解 | 件数 | 識別率 |','|---|---:|---:|---:|']
        for role in ['reference','natural_target']:
            group=[r for r in natural_rows if r['role']==role]
            lines.append(f"| {role} | {sum(r['correct'] for r in group)} | {len(group)} | {np.mean([r['correct'] for r in group])*100:.2f}% |")
    lines += ['', '混同表は観測結果であり、実装差の原因は生成長・ノイズ・リサンプル・重み精度を揃える追加診断で検証する。',
              '全件の誤認先はdecisions.csv、話者単位はspeaker-gaps.csv、全混同ペアはconfusion-comparison.csv。','']
    (out/'RESULTS.ja.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selection=selection,concentration=concentration),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
