"""Paired analysis of native precision, reference tensors, and path replacements."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np

OUT=Path(__file__).resolve().parent;RUN=OUT.parents[1]


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    x,y=a.astype(float),b.astype(float)
    return dict(relative_l2=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
                cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
                max_abs=float(np.max(np.abs(y-x))))


def interval(left,right):
    keys=sorted(left);assert set(keys)==set(right)
    a,b=np.array([np.mean(left[s]) for s in keys]),np.array([np.mean(right[s]) for s in keys])
    d=b-a;draws=np.random.default_rng(20261019).integers(0,len(keys),(2000,len(keys)))
    values=d[draws].mean(1)
    return dict(difference=float(d.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)))


def main():
    speakers=set(json.loads((RUN/'splits.json').read_text())['diagnostic_jvs'])
    original=defaultdict(list)
    for line in (RUN/'report/per-input.jsonl').open():
        r=json.loads(line)
        if r['kind']=='one' and r['item'].split(':')[1] in speakers:
            original[r['case']].append(r)
    saved=[];original_intervals=[]
    for case,rows in original.items():
        for metric in ['camplus_cosine','speaker_top1']:
            saved.append(dict(case=case,metric=metric,mean=float(np.mean([r['metrics'][metric] for r in rows])),takes=len(rows)))
    for variant in ['mf','rf']:
        for precision in ['practical','fp32']:
            a=f'official-{variant}-'+('bf16' if precision=='practical' else 'fp32')
            b=f'speech-{variant}-'+('f16' if precision=='practical' else 'fp32')
            for metric in ['camplus_cosine','speaker_top1']:
                by=[]
                for ca in [a,b]:
                    ds=defaultdict(list)
                    for r in original[ca]:ds[r['item'].split(':')[1]].append(r['metrics'][metric])
                    by.append(ds)
                original_intervals.append(dict(variant=variant,precision=precision,metric=metric,**interval(*by)))
    write_csv('original-matched20.csv',saved);write_csv('original-matched20-comparisons.csv',original_intervals)
    dumps_path=OUT/'dump-results.json'
    if not dumps_path.exists():return
    ds={r['id']:r for r in json.loads(dumps_path.read_text())}
    comparisons=[];normalization=[]
    for variant in ['mf','rf']:
        for sp in sorted(speakers):
            def load(mode,key):return np.fromfile(ds[f'{variant}:{sp}:{mode}'][key],dtype='<f4')
            if f'{variant}:{sp}:speech_official_latent' not in ds:continue
            for label,aa,bb in [
                ('resampler','official:resampled_out','speech_native:resampled_out'),
                ('preprocessed_pcm','official:pcm_out','speech_native:pcm_out'),
                ('reference_latent_total','official:latent_out','speech_native:latent_out'),
                ('codec_same_pcm','official:latent_out','speech_official_pcm:latent_out'),
                ('speaker_total','official:state_out','speech_native:state_out'),
                ('speaker_same_pcm','official:state_out','speech_official_pcm:state_out'),
                ('speaker_same_latent','official:state_out','speech_official_latent:state_out')]:
                m,k=aa.split(':');n,l=bb.split(':')
                comparisons.append(dict(variant=variant,speaker=sp,stage=label,**compare(load(m,k),load(n,l))))
            a=ds[f'{variant}:{sp}:official']['result'];b=ds[f'{variant}:{sp}:speech_native']['result']
            assert a['frames']==b['frames'] and a['speaker_tokens']==b['speaker_tokens']==1+a['frames']//4
            normalization.append(dict(variant=variant,speaker=sp,frames=a['frames'],speaker_tokens=a['speaker_tokens'],
                                      official_db=a['normalize_db'],speech_db=b['lufs'],cached_latent_max_abs=a['cached_latent_max_abs'],
                                      official_gain=float(np.linalg.norm(load('official','pcm_out'))/np.linalg.norm(load('official','resampled_out'))),
                                      speech_gain=float(np.linalg.norm(load('speech_native','pcm_out'))/np.linalg.norm(load('speech_native','resampled_out')))))
    write_csv('tensor-comparisons.csv',comparisons);write_csv('preprocessing-audit.csv',normalization)
    backend=[]
    cpu=OUT/'cpu-codec-audit.json'
    if cpu.exists():
        for r in json.loads(cpu.read_text()):
            a=np.fromfile(Path(r['pcm']).parent/'official-latent.f32',dtype='<f4')
            b=np.fromfile(r['windowed'],dtype='<f4');c=np.fromfile(r['whole'],dtype='<f4')
            assert r['result']['codec_input_dtype']==0
            backend.append(dict(mode='cpu',speaker=r['id'],relative_l2=compare(a,b)['relative_l2'],
                                whole_relative_l2=compare(a,c)['relative_l2'],window_vs_whole=compare(b,c)['relative_l2']))
    gpu=OUT/'gpu-codec-audit.json'
    if gpu.exists():
        for r in json.loads(gpu.read_text()):
            assert r['result']['codec_input_dtype']==0
            backend.append({k:r[k] for k in ['mode','speaker','relative_l2','whole_relative_l2','window_vs_whole']})
    write_csv('backend-codec-comparisons.csv',backend)
    score_path=OUT/'scores.jsonl'
    if not score_path.exists():return
    rows=[json.loads(line) for line in score_path.read_text().splitlines()]
    summary=[];contrasts=[]
    modes=sorted({r['mode'] for r in rows})
    for variant in ['mf','rf']:
        for mode in modes:
            selected=[r for r in rows if r['variant']==variant and r['mode']==mode]
            if not selected:continue
            for metric in ['cosine','top1']:
                seeds=[np.mean([r[metric] for r in selected if r['seed']==seed]) for seed in range(5)]
                summary.append(dict(variant=variant,mode=mode,metric=metric,mean=float(np.mean(seeds)),
                                    population_std=float(np.std(seeds)),takes=len(selected),speakers=len({r['speaker'] for r in selected})))
        for mode in modes:
            if mode=='official':continue
            for metric in ['cosine','top1']:
                by=[]
                for m in ['official',mode]:
                    ds0=defaultdict(list)
                    for r in rows:
                        if r['variant']==variant and r['mode']==m:ds0[r['speaker']].append(r[metric])
                    by.append(ds0)
                if by[0] and by[1]:contrasts.append(dict(variant=variant,mode=mode,metric=metric,**interval(*by)))
    write_csv('controlled-summary.csv',summary);write_csv('controlled-comparisons.csv',contrasts)
    lines=['# 1クリップ参照の再現性差の調査','',
           '確認できた主な差は、参照音声のリサンプル方式と、Vulkan codec内のF16演算経路。',
           'モデル・codecの重みをFP32にするだけでは、Vulkanの内部F16への丸めは除けない。',
           '同じ公式PCMでのlatent誤差は、既定Vulkanで相対L2約0.09379、内部F16経路を無効にすると約0.00000310。',
           '同じlatentから作る話者特徴量のcosineは約0.999996で、パッチ化や平均tokenの扱いは公式と対応している。',
           'MFの生成長・ノイズ共通実験では、公式との差は通常参照-0.003280、公式PCM-0.001248、公式latent-0.000182、公式話者特徴量-0.000013。',
           '数値はこの固定20話者での点推定。元の100話者の低下量と直接混ぜず、区間も参照する。','',
           '既存の固定診断20話者で比較。MF・RFとも通常条件は5文×5 seed。',
           '追加の統制実験はすべてFP32重み・公式の固定生成長・共通FP32初期ノイズ。既定Vulkanの内部演算経路は維持。',
           'MFは20話者×5文×5 seed（各500音声）、RFは同じ20話者×先頭1文×5 seed（各100音声）。',
           '保存済み100話者centroidを共通に用い、採点器・元結果は変更しない。','',
           '| 通常条件・同じ20話者 | CAM++類似度 | 話者Top-1 |','|---|---:|---:|']
    lookup={(r['case'],r['metric']):r['mean'] for r in saved}
    for case in ['official-mf-bf16','speech-mf-f16','official-mf-fp32','speech-mf-fp32','official-rf-bf16','speech-rf-f16','official-rf-fp32','speech-rf-fp32']:
        lines.append(f"| {case} | {lookup[case,'camplus_cosine']:.6f} | {lookup[case,'speaker_top1']*100:.2f}% |")
    lines += ['', '| 中間値の比較 | 平均cosine | 平均相対L2誤差 |','|---|---:|---:|']
    for variant in ['mf','rf']:
        for stage in sorted({r['stage'] for r in comparisons}):
            group=[r for r in comparisons if r['variant']==variant and r['stage']==stage]
            lines.append(f"| {variant} / {stage} | {np.mean([r['cosine'] for r in group]):.8f} | {np.mean([r['relative_l2'] for r in group]):.6f} |")
    lines += ['', 'speaker_same_latentは同一latentを話者encoderへ渡した比較。codec_same_pcmは同一前処理波形をcodecへ渡した比較。',
              'reference_latent_totalとspeaker_totalは各実装の独自前処理を含む。','',
              'ラウドネス目標は双方-16。フレーム数と話者token数も一致し、先頭の平均tokenと4フレーム単位のpatch化は同じ。',
              'リサンプルは公式がtorchaudioの既定Hann（幅6・rolloff 0.99）、speech.cppがKaiser（幅64・rolloff 0.9475937）。',
              '公式の再計算latentと、元の評価で保存したlatentの最大差は全20話者で0。',
              '同じlatentからの話者特徴量はほぼ一致し、参照波形からlatentまでの段階で差が拡大している。','',
              '| 同一公式PCMを使ったcodec比較 | latent相対L2誤差 |','|---|---:|']
    for mode in sorted({r['mode'] for r in backend}):
        group=[r for r in backend if r['mode']==mode]
        lines.append(f"| {mode} ({len(group)}話者) | {np.mean([r['relative_l2'] for r in group]):.8f} |")
    lines += ['', 'codecの重み型は全経路でF32。CPUの100frame分割と一括encodeは同値。',
              'GPUのno_coopmatはcooperative matrixを無効化、no_f16はさらにF16演算経路も無効化する派生診断。',
              '元のベンチマーク設定やsubmoduleにこれらの変更は適用していない。','',
              '| 統制条件（全てFP32重み） | 類似度 | Top-1 |','|---|---:|---:|']
    lookup={(r['variant'],r['mode'],r['metric']):r['mean'] for r in summary}
    for variant in ['mf','rf']:
        for mode in ['official','speech_native','speech_official_pcm','speech_official_latent','speech_official_state']:
            if (variant,mode,'cosine') in lookup:
                lines.append(f"| {variant} / {mode} | {lookup[variant,mode,'cosine']:.6f} | {lookup[variant,mode,'top1']*100:.2f}% |")
    lines += ['', 'speech_native: 独自前処理・codec・話者encoder。speech_official_pcm: 公式前処理波形に交換。',
              'speech_official_latent: 公式codec latentに交換。speech_official_state: 公式話者特徴量に交換。',
              '信頼区間は話者単位の対応ありbootstrap（2,000回）。詳細はcontrolled-comparisons.csv。',
              'FP32重みでの通常条件の差は点推定として残るが、20話者の95%区間は0を含む。',
              'リサンプル・codec・話者encoderの中間値と交換実験で差の発生箇所を確認したもので、全100話者での補正後スコアは未測定。',
              '短い参照で影響が強い理由を情報量の少なさと推測できるが、参照長別の中間値交換までは実施していない。',
              '自動話者評価であり、聴感上の自然さ・同一性を直接保証しない。','',
              '確認した固定版ソース：',
              '- engines/Irodori-TTS/irodori_tts/codec.py: encode_waveform内の既定resample、normalize、deterministic mean',
              '- engines/speech.cpp/src/common/resample.cpp: Kaiserの幅・rolloff',
              '- engines/speech.cpp/src/families/irodori-tts/codec.cpp: codec encoderと窓分割',
              '- engines/speech.cpp/src/families/irodori-tts/speaker-encoder.cpp: 4フレームpatch、1/6、RMSNorm、先頭mean token',
              '- engines/speech.cpp/src/common/graph.h: 行列積はF32 accumulationを指定',
              '- engines/speech.cpp/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp: FLOAT_TYPEの内部F16選択',
              '参照・モデルのrevisionは元runのmanifest、診断コード等のhashはsource-audit.jsonに保存。','']
    (OUT/'RESULTS.ja.md').write_text('\n'.join(lines))
    print('\n'.join(lines[-18:]))


if __name__=='__main__':main()
