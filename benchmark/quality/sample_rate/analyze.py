"""Rebuild descriptive paired tables from saved waveforms, embeddings and SQLite."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import soundfile as sf

from run import DEFAULT_OUT, RATES, REFS, MODES, WAVE_COUNT, metrics


def write(out, name, rows):
    if not rows:return
    with (out/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def interval(values):
    """Resample reference clips, never treat takes as independent speakers."""
    grouped=defaultdict(list)
    for ref,value in values:grouped[ref].append(value)
    means=np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    draw=np.random.default_rng(20261019).integers(0,len(means),(2000,len(means)))
    boot=means[draw].mean(1)
    return dict(mean=float(means.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    out=args.out.resolve();conn=sqlite3.connect(f'file:{out}/results.sqlite3?mode=ro',uri=True)
    def rows(kind):return [json.loads(r[0]) for r in conn.execute('SELECT result FROM tasks WHERE kind=? AND status="ok" ORDER BY id',(kind,))]
    waveform=rows('wave_metric');encoded=rows('encode');generated=rows('generation');scores=rows('score')
    config=json.loads((out/'manifest.json').read_text())
    wave_rows=[]
    for rate in RATES:
        selected=[r for r in waveform if r['rate']==rate]
        for stage in ['between','official_vs_soxr48','speech_vs_soxr48']:
            group=[r[stage] for r in selected]
            wave_rows.append(dict(rate=rate,comparison=stage,clips=len(group),
                                  relative_l2=float(np.mean([r['relative_l2'] for r in group])),
                                  cosine=float(np.mean([r['cosine'] for r in group])),
                                  max_length_difference=max(abs(r['length_delta']) for r in group)))
    write(out,'waveform-summary.csv',wave_rows)
    flattened=[dict(rate=r['rate'],ref=r['ref'],comparison=k,**r[k]) for r in waveform
               for k in ['between','official_vs_soxr48','speech_vs_soxr48']]
    write(out,'waveform-per-clip.csv',flattened)
    def feature(mode,rate,ref,ext):return out/'features'/mode/str(rate)/(ref+ext)
    tensor=[]
    pairs=[('native','speech'),('no_f16','speech_no_f16'),('same_pcm_native','speech_official_pcm'),
           ('same_pcm_no_f16','speech_no_f16_official_pcm'),('preprocessing_only','official_speech_pcm')]
    for rate in RATES:
        for ref in REFS:
            a=np.fromfile(feature('official',rate,ref,'.latent.f32'),dtype='<f4')
            for label,mode in pairs:
                b=np.fromfile(feature(mode,rate,ref,'.latent.f32'),dtype='<f4')
                tensor.append(dict(rate=rate,ref=ref,comparison=label,**metrics(a,b)))
    write(out,'latent-per-clip.csv',tensor)
    tensor_summary=[]
    for rate in RATES:
        for label,_ in pairs:
            group=[r for r in tensor if r['rate']==rate and r['comparison']==label]
            tensor_summary.append(dict(rate=rate,comparison=label,clips=len(group),
                                       relative_l2=float(np.mean([r['relative_l2'] for r in group])),
                                       cosine=float(np.mean([r['cosine'] for r in group])),
                                       max_length_difference=max(abs(r['length_delta']) for r in group)))
    write(out,'latent-summary.csv',tensor_summary)
    centroid=np.array(json.loads((out/'speaker-centroid.json').read_text())['embedding'])
    individual=[]
    def key(r):return r['variant'],r['mode'],r['rate'],r['ref'],r['text_id'],r['seed']
    generated_index={key(r):r for r in generated}
    for r in scores:
        raw,sr=sf.read(r['wav'],dtype='float32')
        base=generated_index.get((r['variant'],'official',r['rate'],r['ref'],r['text_id'],r['seed']))
        correlation=None
        if base:
            audio,_=sf.read(base['wav'],dtype='float32')
            correlation=metrics(audio,raw)['cosine']
        individual.append({**{k:r[k] for k in ['variant','mode','rate','ref','text_id','seed','wav']},
                           'cosine':float(centroid@np.array(r['embedding'])),
                           'waveform_cosine_to_official':correlation,
                           'seconds':len(raw)/sr,'near_zero_fraction':float(np.mean(np.abs(raw)<1e-4)),
                           'clipping_fraction':float(np.mean(np.abs(raw)>=1)),
                           'noise_sha256':r['noise_sha256']})
    score_index={key(r):r for r in individual}
    summary=[];contrasts=[]
    for variant in ['mf','rf']:
        for mode in MODES:
            for rate in RATES:
                group=[r for r in individual if r['variant']==variant and r['mode']==mode and r['rate']==rate]
                if not group:continue
                seed_means=[np.mean([r['cosine'] for r in group if r['seed']==seed]) for seed in range(5)]
                summary.append(dict(variant=variant,mode=mode,rate=rate,takes=len(group),
                                    cosine_mean=float(np.mean(seed_means)),seed_population_std=float(np.std(seed_means)),
                                    clipping_fraction=float(np.mean([r['clipping_fraction'] for r in group])),
                                    near_zero_fraction=float(np.mean([r['near_zero_fraction'] for r in group]))))
                for contrast in ['vs_official_same_rate','vs_same_mode_48000']:
                    pairs0=[]
                    for r in group:
                        baseline=(variant,'official',rate,r['ref'],r['text_id'],r['seed']) if contrast=='vs_official_same_rate' else (variant,mode,48000,r['ref'],r['text_id'],r['seed'])
                        if baseline in score_index:pairs0.append((r['ref'],r['cosine']-score_index[baseline]['cosine']))
                    if pairs0:contrasts.append(dict(variant=variant,mode=mode,rate=rate,contrast=contrast,takes=len(pairs0),**interval(pairs0)))
    write(out,'generation-per-input.csv',individual);write(out,'speaker-summary.csv',summary);write(out,'speaker-paired-differences.csv',contrasts)
    failures=[dict(json.loads(r[2]),id=r[0],kind=r[1]) for r in conn.execute('SELECT id,kind,result FROM tasks WHERE status="failed" ORDER BY id')]
    (out/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    groups=defaultdict(list)
    for row in generated:groups[row['variant'],row['ref'],row['text_id'],row['seed']].append(row)
    # Every controlled take must share generation length and the exact noise bytes.
    bad=[list(k) for k,group in groups.items() if len({r['noise_sha256'] for r in group})!=1 or len({r['frames_expected'] for r in group})!=1]
    validation=dict(waveforms=len(waveform),expected_waveforms=WAVE_COUNT,encodings=len(encoded),expected_encodings=180,
                    generations=len(generated),expected_generations=3600,scores=len(scores),failed=len(failures),
                    paired_groups=len(groups),expected_groups=150,noise_or_length_mismatch=bad,
                    expected_conditions_per_group=24,incomplete_groups=[list(k) for k,g in groups.items() if len(g)!=24],
                    source_subtypes=sorted({r['source_subtype'] for r in json.loads((out/'inputs.json').read_text())}))
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    if bad:raise ValueError('Controlled lengths or noises mismatch')
    if not failures and (len(generated)!=3600 or len(scores)!=3600 or len(waveform)!=WAVE_COUNT or len(encoded)!=180):
        raise ValueError('Missing planned outputs without failure records')
    lines=['# つくよみちゃん音源による入力サンプルレート比較','',
           f'波形比較 {len(waveform)}/{WAVE_COUNT}件、参照エンコード {len(encoded)}/180件、生成 {len(generated)}/3600件、採点 {len(scores)}/3600件。失敗 {len(failures)}件。','',
           '同じ1話者の96kHz原音から、libsoxr VHQで96・48・44.1・32・24・16kHzのfloat32 WAVを作成した。',
           '提供された元WAVは99件が96kHz・float32、ID050のみ48kHz・float32。050は波形比較から除外し、96kHzへのアップサンプルで原音を偽造しない。',
           '元96kHzの99クリップで公式とspeech.cppの48kHz変換波形を比較した。生成参照はID001〜005、照合centroidは別のID011〜020、合成文章はID021〜023。',
           '生成はMF/RF、seed 0〜4、全条件で公式48kHz入力の予測生成長と同一初期ノイズを使用。透かし・末尾トリムなし。全モデル・codec重みはFP32。',
           '通常Vulkan、内部F16無効Vulkan、公式版、および公式版へspeech.cppの正規化済みPCMを渡す介入条件を比較する。最後の条件は前処理の差を分離する。','',
           '| 入力kHz | 波形の相対L2差（99クリップ平均） | 波形cosine |','|---|---:|---:|']
    for rate in RATES:
        r=next(x for x in wave_rows if x['rate']==rate and x['comparison']=='between')
        lines.append(f"| {rate/1000:g} | {r['relative_l2']*100:.5f}% | {r['cosine']:.9f} |")
    lines+=['','この波形差は数値差であり、品質低下率ではない。公式Hannとspeech.cpp Kaiserの差であり、一方の方式の優劣を直接示さない。',
            '48kHzへのsoxr変換波形との差もCSVに保存したが、低レート化で失われた帯域も含むので実装差とは区別する。','',
            '| 入力kHz | 通常Vulkan latent差 | F16無効 latent差 | 同じPCM・F16無効 latent差 |','|---|---:|---:|---:|']
    for rate in RATES:
        def v(label):return next(r['relative_l2']*100 for r in tensor_summary if r['rate']==rate and r['comparison']==label)
        lines.append(f"| {rate/1000:g} | {v('native'):.5f}% | {v('no_f16'):.5f}% | {v('same_pcm_no_f16'):.6f}% |")
    for variant in ['mf','rf']:
        lines+=['',f'{variant.upper()}：CAM++ cosine（各セル75音声）','',
                '| 入力kHz | 公式 | speech.cpp通常 | speech.cpp F16無効 | 公式＋speech.cpp前処理 |','|---|---:|---:|---:|---:|']
        for rate in RATES:
            values=[]
            for mode in MODES:
                row=next((r for r in summary if r['variant']==variant and r['mode']==mode and r['rate']==rate),None)
                values.append(f"{row['cosine_mean']:.6f}" if row else '欠測')
            lines.append('| '+f'{rate/1000:g}'+' | '+' | '.join(values)+' |')
        lines+=['','同じ実装の48kHz入力に対する差','',
                '| 入力kHz | 公式 | speech.cpp通常 | speech.cpp F16無効 | 公式＋speech.cpp前処理 |','|---|---:|---:|---:|---:|']
        for rate in RATES:
            values=[]
            for mode in MODES:
                row=next((r for r in contrasts if r['variant']==variant and r['mode']==mode and r['rate']==rate and r['contrast']=='vs_same_mode_48000'),None)
                values.append(f"{row['mean']:+.6f}" if row else '欠測')
            lines.append('| '+f'{rate/1000:g}'+' | '+' | '.join(values)+' |')
    lines+=['','1話者の実験であり、話者Top-1は評価しない。95%区間は参照5クリップ単位の対応bootstrapで、話者集団への信頼区間ではない。',
            '自動話者類似度の診断であり、聴感・読みの自然さの同等性は検証していない。','',
            '保存済みデータからの再集計：',
            f'`.venvs/quality/bin/python benchmark/quality/sample_rate/analyze.py --out {out}`','']
    (out/'RESULTS.ja.md').write_text('\n'.join(lines))
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    conn.close()


if __name__=='__main__':main()
