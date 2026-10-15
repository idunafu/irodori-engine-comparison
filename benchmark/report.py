"""Aggregate primary wall measurements separately from synchronized component probes."""
import collections
import csv
import json
import os
import re
from pathlib import Path
import statistics as stats
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/benchmark/results.jsonl'
DEST=ROOT/'benchmark/results'

def median(xs):return float(stats.median(xs)) if xs else None
def total_events(result,stage):return sum(e['ms'] for e in result.get('events',[]) if e['stage']==stage)
def stages(result):
    s=result.get('stages',{})
    return dict(s) if isinstance(s,(list,dict)) else {}

def forward_batch(e):
    if e.get('batch'):return e['batch']
    if e.get('shape'):return e['shape'][0]
    if e.get('shapes'):return e['shapes'][0][0]
    return None

def components(result):
    events=result.get('events',[]); names={e['stage'] for e in events};s=stages(result)
    def get(name,probe=None):
        if name in names:return total_events(result,name)
        if probe and probe in names:return total_events(result,probe)
        return 0.0
    speaker=get('speaker_encode')
    if 'reference_total' in names:reference=get('reference_total')
    elif 'prepare_reference' in s:reference=s['prepare_reference']*1000+speaker
    elif 'reference_encode' in names:reference=get('reference_encode')+speaker
    else:reference=get('codec_encode')+speaker
    decode=s.get('decode_latent',None)
    decode=decode*1000 if decode is not None else get('codec_decode')
    if not decode and 'codec' in s:decode=s['codec']*1000
    forwards=[e for e in events if e['stage']=='dit_forward']
    codec_encode=get('codec_encode')
    if 'codec_encoder_network' in names:
        codec_encode=get('codec_encoder_network')+get('codec_encoder_projection')
    else:codec_encode=max(0.,codec_encode-get('reference_preprocess_in_codec'))
    return dict(sampling_ms=get('sampling'),sampling_loop_ms=get('sampling_loop'),
        text_tokenize_ms=get('text_tokenize') if 'text_tokenize' in names else (s['tokenize_text']*1000 if 'tokenize_text' in s else None),
        caption_tokenize_ms=get('caption_tokenize') if 'caption_tokenize' in names else None,
        reference_preprocess_ms=get('reference_file_read')+get('reference_preprocess')+get('reference_preprocess_in_codec'),
        reference_codec_load_ms=get('reference_codec_load'),
        text_encode_ms=get('text_encode','text_encode_probe'),caption_encode_ms=get('caption_encode','caption_encode_probe'),
        encoder_probe=('text_encode_probe' in names),shared_encoder_ms=get('text_caption_duration_fused')+get('text_caption_backbone_fused'),
        reference_encode_ms=reference,reference_codec_ms=codec_encode,speaker_encode_ms=speaker,decode_ms=decode,
        duration_ms=get('duration_predict'),segments=sum(e['stage']=='sampling' for e in events),forward_calls=len(forwards),
        forward_first_ms=forwards[0]['ms'] if forwards else None,
        forward_b1_ms=median([e['ms'] for e in forwards[1:] if forward_batch(e)==1]),
        forward_b4_ms=median([e['ms'] for e in forwards[1:] if forward_batch(e)==4]))

def latest_records(path):
    latest={}
    if not path.exists():return []
    with path.open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('kind')=='profile':
                version=max(r.get('instrumentation_version',0),r.get('result',{}).get('instrumentation_version',0))
                required=3 if r['case']['engine'] in ('Irodori-TTS','Irodori-TTS-Optimiz') else 2
                if version<required:continue
            latest[r['key']]=r
    return list(latest.values())

def aggregate(records):
    groups=collections.defaultdict(list);cold=collections.defaultdict(list)
    for r in records:
        if r.get('kind')=='cold' and r['status']=='ok':cold[r['case']['case']].append(r)
        if r.get('kind') in ('wall','profile','idle') and r['status']=='ok':
            parts=2 if r['kind'] in ('wall','profile') else 1
            group=r['key'].rsplit('/',parts)[0]
            groups[group].append(r)
    rows=[]
    for key,xs in groups.items():
        wall=[r['result'] for r in xs if r['kind']=='wall'];profile=[components(r['result']) for r in xs if r['kind']=='profile']
        idle=[r['result'] for r in xs if r['kind']=='idle']
        if not wall:continue
        r=next(r for r in xs if r['kind']=='wall');c=r['case'];cs=cold[c['case']]
        row=dict(case=c['case'],engine=c['engine'],variant=c['variant'],backend=c['backend'],precision=c['precision'],
                 codec_precision='fp16' if c['engine']=='irodori-onnx' and c['precision']=='fp16' else 'fp32',
                 precision_profile=c['precision_profile'],cache=r['cache'],mode=r['mode'],prompt=r['prompt']['id'],
                 tokens=r['prompt']['tokens'],characters=r['prompt']['characters'],bucket=r['prompt']['bucket'],
                 requested_seconds=r.get('seconds'),wall_samples=len(wall),profile_samples=len(profile),
                 warm_ms=median([x['ms'] for x in wall]),min_ms=min(x['ms'] for x in wall),max_ms=max(x['ms'] for x in wall),
                 stdev_ms=stats.pstdev(x['ms'] for x in wall),audio_seconds=median([x['samples']/x['sample_rate'] for x in wall]),
                 frames=median([x['frames'] for x in wall]),rtf=median([x['rtf'] for x in wall]),
                 resident_mib=median([x['resident_vram']/2**20 for x in idle if x.get('resident_vram') is not None]),
                 peak_mib=max(x['peak_vram'] for x in wall)/2**20,
                 allocator_peak_allocated_mib=max((x['peak_allocated']/2**20 for x in wall if x.get('peak_allocated') is not None),default=None),
                 allocator_peak_reserved_mib=max((x['peak_reserved']/2**20 for x in wall if x.get('peak_reserved') is not None),default=None),
                 load_ms=median([x['load']['load_ms'] for x in cs]),cold_ms=median([x['result']['ms'] for x in cs]),
                 startup_ms=median([x['load']['startup_wall_ms'] for x in cs]),
                 loaded_resident_mib=median([x['load']['resident_vram']/2**20 for x in cs if x['load'].get('resident_vram') is not None]),
                 load_peak_mib=max((x['load']['peak_vram']/2**20 for x in cs),default=None),
                 vram_method=','.join(sorted({x['vram_method'] for x in wall})))
        for metric in ('sampling_ms','sampling_loop_ms','text_encode_ms','caption_encode_ms','shared_encoder_ms',
                       'text_tokenize_ms','caption_tokenize_ms','reference_preprocess_ms','reference_codec_load_ms',
                       'reference_encode_ms','reference_codec_ms','speaker_encode_ms','decode_ms','duration_ms',
                       'segments','forward_calls','forward_first_ms','forward_b1_ms','forward_b4_ms'):
            row[metric]=median([p[metric] for p in profile if p[metric] is not None])
        row['encoder_probe']=any(p['encoder_probe'] for p in profile)
        def padding_tokens():
            for sample in xs:
                if sample['kind']!='profile':continue
                for event in sample['result'].get('events',[]):
                    if event['stage'] in ('text_encode','text_encode_probe','text_tokenize'):
                        shape=event.get('shape') or (event.get('shapes') or [None])[0]
                        if shape and len(shape)==2:return shape[1]
            if c['engine']=='sttts-gui':
                return next(n for n in (16,24,32,48,64,96,128,192,256) if n>=r['prompt']['tokens'])
            return 256 if c['engine'] in ('Irodori-TTS','Irodori-TTS-Optimiz','audio.cpp') else r['prompt']['tokens']
        row['text_padded_tokens']=padding_tokens()
        row['at_30_second_limit']=row['audio_seconds']>=30.
        shapes=[]
        hits=[]
        for sample in xs:
            if sample['kind']!='profile':continue
            result=sample['result'];events=result.get('events',[])
            for event in events:
                if event['stage']!='dit_forward':continue
                shape=event.get('shape') or (event.get('shapes') or [None])[0] or [event.get('batch'),event.get('frames')]
                if shape not in shapes:shapes.append(shape)
            hit=result.get('reference_cache_hit')
            if c['engine']=='audio.cpp':hit=not any(e['stage']=='codec_encode' for e in events)
            if hit is not None:hits.append(int(hit))
        row['dit_shapes_json']=json.dumps(shapes,separators=(',',':'))
        row['reference_cache_hit_rate']=sum(hits)/len(hits) if hits else None
        row['complete']=len(wall)==5 and len(profile)==5
        rows.append(row)
    return rows

def fmt(value,places=1):return '—' if value is None else f'{value:.{places}f}'
def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    DEST.mkdir(exist_ok=True)
    records=latest_records(DATA);rows=aggregate(records);write_csv(DEST/'per-input.csv',rows)
    case_rows=collections.defaultdict(list)
    for r in rows:
        if r['mode']=='matched':case_rows[(r['case'],r['cache'])].append(r)
    summary=[]
    for key,xs in sorted(case_rows.items()):
        row={k:xs[0][k] for k in ('case','engine','variant','backend','precision','codec_precision','precision_profile','cache')}
        row.update(prompts_complete=sum(x['complete'] for x in xs),prompts_measured=len(xs),
                   peak_mib=max(x['peak_mib'] for x in xs),load_peak_mib=xs[0]['load_peak_mib'])
        for name in ('allocator_peak_allocated_mib','allocator_peak_reserved_mib'):
            row[name]=max((x[name] for x in xs if x[name] is not None),default=None)
        for name in ('load_ms','startup_ms','loaded_resident_mib','cold_ms','warm_ms','sampling_ms','sampling_loop_ms','text_encode_ms','caption_encode_ms','text_tokenize_ms',
                     'shared_encoder_ms','reference_encode_ms','reference_codec_ms','reference_preprocess_ms','reference_codec_load_ms','caption_tokenize_ms','speaker_encode_ms','decode_ms',
                     'forward_calls','forward_first_ms','forward_b1_ms','forward_b4_ms','rtf','resident_mib'):
            row[name]=median([x[name] for x in xs if x[name] is not None])
        row['encoder_probe']=any(x['encoder_probe'] for x in xs)
        summary.append(row)
    write_csv(DEST/'summary.csv',summary)
    failed=[r for r in records if r['status']=='failed']
    for r in failed:
        log=ROOT/'logs'/f'benchmark-{r["case"]["case"]}.log'
        r['stderr_log']=str(log.relative_to(ROOT))
        if r['case']['engine']=='speech.cpp' and r.get('mode')=='auto' and log.exists():
            messages=re.findall(r"the text's speech is predicted to last [^\n]+",log.read_text(errors='replace'))
            if messages:
                r['upstream_limit']='Auto duration exceeds the 30 second per-request limit; upstream rejects the request.'
                r['example_backend_message']=messages[0]
    failed_groups=[r for r in failed if r.get('mode')]
    (DEST/'failures.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2)+'\n')
    cfg=json.loads((ROOT/'benchmark/config.json').read_text())
    text=['# Irodori-TTS 性能比較結果','',
          '同一の公式モデルrevision・20入力・TYC054.wav（10.990秒、48kHz、24bit）を使用。RFは40、MFは4ステップ。読み上げ品質は評価していない。',
          '',f'検証した入力条件: **{sum(r["complete"] for r in rows)+len(failed_groups)}/3120**（測定成功 {sum(r["complete"] for r in rows)}、拒否・失敗 {len(failed_groups)}）。通常速度と詳細計測は各5回。表の時間はms、VRAMはMiB。'+('未完了の値は暫定値。' if sum(r['complete'] for r in rows)+len(failed_groups)<3120 else '全条件の測定・拒否理由の記録が完了。'),
          '', 'ロード・初回推論は代表文章p09で3回。warmなどは文章ごとの中央値をさらに20文章で中央値に集計する。ピークVRAMは通常計測の最大観測値。',
          '', 'DiT B1/B4は実際のbatch別のforward時間。CFGを別々のB1 forwardで実行する実装があるため、呼び出し数も併記する。個別encoderに † のある実装は独立probeの値で、共有処理時間とは加算できない。',
          '', '詳細条件と測定範囲は [README](../README.ja.md)、全入力の集計は [CSV](per-input.csv)。']
    for variant in ('rf','mf'):
        for profile_name in ('fp32','practical'):
            for cache in ('reencode','native_cache'):
                xs=[r for r in summary if r['variant']==variant and r['cache']==cache and
                    (r['precision_profile']==profile_name or r['engine'] in ('sttts-gui','Irodori-TTS-burn'))]
                text += ['',f'## {variant.upper()} / {profile_name} / {cache}','',
                    '| 実装 | backend | モデル / codec | 完了入力 | load | 初回 | warm | sample | loop | DiT B1 / B4 | calls | DACVAE | RTF | 常駐 | peak |',
                    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
                for r in xs:
                    text.append(f'| {r["engine"]} | {r["backend"]} | {r["precision"]} / {r["codec_precision"]} | {r["prompts_complete"]}/20 | '+
                        ' | '.join(fmt(r[n]) for n in ('load_ms','cold_ms','warm_ms','sampling_ms','sampling_loop_ms'))+
                        f' | {fmt(r["forward_b1_ms"],2)} / {fmt(r["forward_b4_ms"],2)} | {fmt(r["forward_calls"],0)} | {fmt(r["decode_ms"])} | {fmt(r["rtf"],3)} | {fmt(r["resident_mib"],0)} | {fmt(r["peak_mib"],0)} |')
                text += ['', '| 実装 / backend | text tokenize | caption tokenize | text encode | caption encode | 共有encoder | 参照音声 合計 | 前処理 | encoder load | DACVAE encode | speaker encode |',
                         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
                for r in xs:
                    label=r['engine']+' / '+r['backend']+(' †' if r['encoder_probe'] else '')
                    text.append('| '+label+' | '+' | '.join(fmt(r[n]) for n in ('text_tokenize_ms','caption_tokenize_ms','text_encode_ms','caption_encode_ms','shared_encoder_ms','reference_encode_ms','reference_preprocess_ms','reference_codec_load_ms','reference_codec_ms','speaker_encode_ms'))+' |')
                text += ['', '| 実装 / backend | プロセス起動からready | ロード後VRAM | ロード中peak | allocator allocated / reserved peak | DiT先頭step |',
                         '|---|---:|---:|---:|---:|---:|']
                for r in xs:
                    text.append('| '+r['engine']+' / '+r['backend']+' | '+' | '.join(fmt(r[n]) for n in ('startup_ms','loaded_resident_mib','load_peak_mib'))+
                        ' | '+fmt(r['allocator_peak_allocated_mib'])+' / '+fmt(r['allocator_peak_reserved_mib'])+' | '+fmt(r['forward_first_ms'])+' |')
    text += ['', '## 自動予測とスケーリング','',
        '固定8秒のグラフでは入力トークン数だけを変化させ、出力長のグラフでは文章p09を固定する。モデルのpadding・bucket処理は維持するため、計算形状が変わらない区間や段階的な変化がある。',
        '', '自動予測が上限を超える場合の処理（clamp、拒否、分割）は実装差として記録する。失敗した入力を成功例の速度に混ぜない。']
    colors={e:c for e,c in zip(sorted({r['engine'] for r in rows}),plt.cm.tab10.colors)}
    for variant in ('rf','mf'):
        for profile_name in ('fp32','practical'):
            for mode in ('text_scaling','audio_scaling','auto'):
                selected=[r for r in rows if r['variant']==variant and r['cache']=='native_cache' and r['mode']==mode and r['complete'] and
                          (r['precision_profile']==profile_name or r['engine'] in ('sttts-gui','Irodori-TTS-burn'))]
                if not selected:continue
                fig,axes=plt.subplots(1,2,figsize=(13,4.5));cases=collections.defaultdict(list)
                for r in selected:cases[r['case']].append(r)
                for case,xs in cases.items():
                    for ax,metric in zip(axes,('warm_ms','sampling_ms')):
                        bins=collections.defaultdict(list)
                        for r in xs:bins[r['requested_seconds'] if mode=='audio_scaling' else r['tokens']].append(r[metric])
                        points=sorted((x,median([v for v in vals if v is not None])) for x,vals in bins.items())
                        ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',markersize=3,
                                color=colors[xs[0]['engine']],linestyle='-' if xs[0]['backend']=='cuda' else '--',
                                label=xs[0]['engine']+' / '+xs[0]['backend'])
                        ax.set_xlabel('Output seconds' if mode=='audio_scaling' else 'Input tokens (including BOS)')
                        ax.set_ylabel(metric+' (ms)');ax.grid(alpha=.25)
                for ax in axes:
                    ax.set_ylim(bottom=0)
                axes[0].legend(fontsize=7);fig.suptitle(f'{variant.upper()} / {profile_name} / native cache / {mode}')
                fig.tight_layout();name=f'{variant}-{profile_name}-{mode}.png';fig.savefig(DEST/name,dpi=160);plt.close(fig)
                text+=['',f'![{variant} {profile_name} {mode}]({name})']
    text += ['', '## 自動長予測（native cache）','',
        '| 実装 / model / backend / 精度 | 完了入力 | warm ms | 音声秒数 | RTF | 生成segment数 | 30秒以上の入力数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    automatic=collections.defaultdict(list)
    for r in rows:
        if r['mode']=='auto' and r['cache']=='native_cache' and r['complete']:automatic[r['case']].append(r)
    for _,xs in sorted(automatic.items()):
        r=xs[0];label=f'{r["engine"]} / {r["variant"]} / {r["backend"]} / {r["precision"]}'
        text.append('| '+label+f' | {len(xs)}/20 | '+
            ' | '.join(fmt(median([x[k] for x in xs]),3 if k=='rtf' else 1) for k in ('warm_ms','audio_seconds','rtf'))+
            ' | '+fmt(median([x['segments'] for x in xs]),0)+f' | {sum(x["at_30_second_limit"] for x in xs)} |')
    text += ['', '## 制限・確認結果','',
        'audio.cppはMF、ONNXはRF、OptimizはMFに上流非対応。stttsとBurnはVulkanのみ。FP32共通基準は重みとcodecを揃えたもので、内部attention・積算の精度まで同一とは限らない。実用設定のONNX decoderはFP16、他のcodecはFP32。',
        '', 'speech.cppの自動長予測は30秒を超える要求を拒否する。該当入力は失敗一覧にstderrログとともに保存し、固定生成長の主比較とは区別する。',
        '', '前処理はファイル読取り・正規化などのscopeを集計。codecの外側scopeに含まれる前処理は内側scopeの実測値を差し引き、重複計上しない。Burnはvoice登録のたびにcodec encoderを読み込むため、その時間をencoder load列で分離する。実際のpadding長・DiT形状・キャッシュhit率は入力別CSVに保存する。',
        '', 'NVMLピークは10ms間隔の観測値。allocatorのピークやGPU全体の値と同じ定義として扱わない。OS・ドライバのディスクキャッシュを維持し、GPUのクロック・電力制限は変更していない。',
        '', f'記録された失敗: {len(failed)}件。[詳細](failures.json)。']
    (DEST/'RESULTS.ja.md').write_text('\n'.join(text)+'\n')
    print('complete',sum(r['complete'] for r in rows),'of 3120; rows',len(rows),'failures',len(failed))

if __name__=='__main__':main()
