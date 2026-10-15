"""Summarize independent wall measurements and diagnostic probes separately."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np

from mf_cause import ROOT, OUT, DEST, snapshot
from report import components

LOCAL_REPORT = OUT / 'report'

def records(path):
    latest={}
    for line in path.read_text().splitlines():
        row=json.loads(line);latest[row['key']]=row
    return list(latest.values())

def interval(a,b,paired=True):
    """Resample independent worker blocks, pairing only contemporaneous experiments."""
    a=np.array(a);b=np.array(b)
    assert a.shape[0]==b.shape[0]==3 and a.ndim==b.ndim==2
    rng=np.random.default_rng(42);block=rng.integers(0,3,(5000,3))
    av=a[block[:,:,None],rng.integers(0,a.shape[1],(5000,3,a.shape[1]))]
    block_b=block if paired else rng.integers(0,3,(5000,3))
    bv=b[block_b[:,:,None],rng.integers(0,b.shape[1],(5000,3,b.shape[1]))]
    delta=(np.median(np.median(bv,axis=2),axis=1)/np.median(np.median(av,axis=2),axis=1)-1)*100
    return np.percentile(delta,[2.5,97.5]).tolist()

def write_csv(name,rows,directory=DEST):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (directory/name).open('w') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)

def group_key(r):
    c=r['case']
    return (c.get('experiment','initial'),c['phase'],c['stack'],c['mode'],c['precision'],r['seconds'],r['cache'])

def main():
    LOCAL_REPORT.mkdir(parents=True,exist_ok=True)
    rs=records(OUT/'results.jsonl');groups=collections.defaultdict(list)
    for r in rs:
        if r['kind']!='complete':groups[group_key(r)].append(r)
    rows=[];lookup={};blocks={};module_rows=[];operator_rows=[]
    fields=('experiment','phase','stack','mode','precision','seconds','cache')
    for key,items in sorted(groups.items()):
        row=dict(zip(fields,key));normal=[r for r in items if r['kind']=='normal']
        by_round=[[r['result']['ms'] for r in normal if r['round']==i] for i in range(3)]
        assert all(by_round),f'Incomplete blocks: {key}'
        assert len({len(b) for b in by_round})==1
        row.update(n=len(normal),ms=st.median(st.median(v) for v in by_round),
                   min_ms=min(r['result']['ms'] for r in normal),max_ms=max(r['result']['ms'] for r in normal),
                   peak_mib=max(r['result']['peak_vram']/2**20 for r in normal))
        probes=[components(r['result']) for r in items if r['kind']=='stages']
        for metric in ('sampling_ms','decode_ms','reference_encode_ms','reference_codec_ms','reference_preprocess_ms','text_encode_ms','caption_encode_ms','speaker_encode_ms'):
            if probes:row[metric]=st.median(p[metric] for p in probes)
        lookup[key]=row;blocks[key]=by_round;rows.append(row)
        modules=collections.defaultdict(list)
        for r in items:
            if r['kind']=='modules':
                for m in r['result']['modules']:modules[(m['name'],m['kind'],str(m['shapes']))].append(m['ms'])
            if r['kind']=='operators':
                for op in r['result']['operators']:
                    operator_rows.append(dict(zip(fields,key))|dict(round=r['round'],**op))
        for (name,kind,shapes),values in modules.items():
            module_rows.append(dict(zip(fields,key))|dict(name=name,kind=kind,shapes=shapes,n=len(values),gpu_ms=st.median(values)))
    write_csv('measurements.csv',rows);write_csv('modules.csv',module_rows);write_csv('operators.csv',operator_rows,LOCAL_REPORT)
    comparisons=[]
    def compare(label,a,b):
        paired=a[0] in ('confirmation','patch-confirmation') or (a[1]=='e2e' and a[3]==b[3]=='default')
        x,y=lookup[a],lookup[b];lo,hi=interval(blocks[a],blocks[b],paired)
        comparisons.append(dict(comparison=label,phase=a[1],precision=a[4],seconds=a[5],cache=a[6],
            old_stack=a[2],old_mode=a[3],new_stack=b[2],new_mode=b[3],old_ms=x['ms'],new_ms=y['ms'],
            delta_pct=(y['ms']/x['ms']-1)*100,ci_low_pct=lo,ci_high_pct=hi,significant=lo>0 or hi<0,
            round_delta_pct=json.dumps([(st.median(n)/st.median(o)-1)*100 for o,n in zip(blocks[a],blocks[b])]),
            temporally_paired=paired))
    def k(experiment,phase,stack,mode,precision,seconds,cache='native_cache'):
        return (experiment,phase,stack,mode,precision,seconds,cache)
    for sec in (8,24):
        for other,label in [('old924','cuDNN only'),('new','latest bundle')]:
            compare(label,k('confirmation','codec','old','default','fp32',sec),k('confirmation','codec',other,'default','fp32',sec))
        compare('framework/CUDA residual',k('confirmation','codec','old924','default','fp32',sec),k('confirmation','codec','new','default','fp32',sec))
        for stack,mode,label in [('new','benchmark','algorithm search'),('new','v7','legacy API')]:
            compare(label,k('initial','codec','new','default','fp32',sec),k('initial','codec',stack,mode,'fp32',sec))
        compare('cuDNN patch',k('patch-confirmation','codec','new','default','fp32',sec),k('patch-confirmation','codec','cudnn9241','default','fp32',sec))
        for precision in ('fp32','bf16'):
            for cache in ('reencode','native_cache'):
                for stack,mode,label in [('new','default','latest bundle'),('old924','default','cuDNN only'),('new','benchmark','latest with search')]:
                    compare(label,k('initial','e2e','old','default',precision,sec,cache),k('initial','e2e',stack,mode,precision,sec,cache))
    write_csv('comparisons.csv',comparisons)
    # Identical latent PCM isolates the codec; MF parity also includes DiT numeric changes.
    parity=[]
    import soundfile as sf
    for base in sorted(OUT.glob('codec-old-default-fp32-*.npy')):
        if 'native_cache' in base.stem:continue
        sec=base.stem.split('-')[-1];a=np.load(base)
        for stack,mode in [('new','default'),('new','benchmark'),('old924','default'),('cudnn9241','default')]:
            candidates=[OUT/f'codec-{stack}-{mode}-fp32-{sec}-native_cache.npy',OUT/f'codec-{stack}-{mode}-fp32-{sec}.npy']
            path=next((p for p in candidates if p.exists()),None)
            if path is None:continue  # Algorithm-search arms measured 8/24 seconds only.
            b=np.load(path)
            assert a.shape==b.shape and np.isfinite(b).all()
            parity.append(dict(phase='codec',stack=stack,mode=mode,seconds=sec,precision='fp32',cache='native_cache',
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),max_abs=float(np.max(np.abs(a-b)))))
    for base in sorted(OUT.glob('e2e-old-default-*.wav')):
        _,_,_,precision,seconds,cache=base.stem.split('-');a,sr=sf.read(base,dtype='float32')
        for stack,mode in [('new','default'),('old924','default'),('new','benchmark')]:
            b,sr2=sf.read(OUT/f'e2e-{stack}-{mode}-{precision}-{seconds}-{cache}.wav',dtype='float32')
            assert sr==sr2==48000 and a.shape==b.shape and np.isfinite(b).all()
            parity.append(dict(phase='e2e',stack=stack,mode=mode,seconds=seconds,precision=precision,cache=cache,
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),max_abs=float(np.max(np.abs(a-b)))))
    write_csv('pcm-parity.csv',parity)
    conv=[]
    for r in records(OUT/'conv-probe.jsonl'):
        result=r['result'];values=[s for s in result['samples'] if not s['warmup']]
        assert result['finite'] and not result['imported_optional']
        conv.append(dict(stack=r['stack'],mode=result['mode'],round=r['round'],torch=result['torch'],cuda=result['cuda'],
            cudnn=result['cudnn'],ms=st.median(s['ms'] for s in values),gpu_ms=st.median(s['gpu_ms'] for s in values),
            first_call_ms=result['samples'][0]['ms'],
            fft_kernels=sum('fft1d' in e['name'] for e in result['kernels']),
            gemv_kernels=sum('gemvx' in e['name'] for e in result['kernels']),
            sgemm_kernels=sum('convolve_sgemm' in e['name'] for e in result['kernels']),
            kernel_count=len(result['kernels']),
            kernels=json.dumps(result['kernels'])))
    assert len(conv)==18
    write_csv('conv-probe.csv',conv)
    preservation=snapshot()==json.loads((OUT/'preservation.json').read_text())
    assert preservation
    for r in rs:
        if r['kind']=='complete':continue
        result=r['result'];assert result['finite'] and result['rms']>1e-8 and result['samples']==r['seconds']*48000
        if r['case']['phase']=='e2e':
            assert result['reference_cache_hit']==(r['cache']=='native_cache')
            if r['kind']=='stages':assert sum(e['stage']=='dit_forward' for e in result['events'])==4
    telemetry=[json.loads(s) for s in (OUT/'telemetry.jsonl').read_text().splitlines()]
    concurrent=[t for t in telemetry if sum('compute' in p['kinds'] for p in t['processes'])>1]
    differences={}
    for stack,baseline,removed,added in [('old924','Irodori-TTS-original-packages.txt','nvidia-cudnn-cu12==9.10.2.21','nvidia-cudnn-cu12==9.24.0.43'),
            ('cudnn9241','Irodori-TTS-packages.txt','nvidia-cudnn-cu13==9.24.0.43','nvidia-cudnn-cu13==9.24.1.1')]:
        a=set((ROOT/'artifacts/runtime-ab-20261015'/baseline).read_text().splitlines())
        b=set((OUT/f'{stack}-packages.txt').read_text().splitlines())
        assert a-b=={removed} and b-a=={added}
        differences[stack]=dict(removed=sorted(a-b),added=sorted(b-a))
    (DEST/'environment-differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    provenance={}
    for r in records(OUT/'startup.jsonl'):
        parts=r['key'].split('/')
        if parts[0]=='codec' and parts[2]=='default' and parts[1] not in provenance:
            provenance[parts[1]]=r['ready']
    provenance['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT/'benchmark').glob('mf_*.py'))+list((ROOT/'environments/mf-runtime-cause-20261015').glob('*requirements.txt'))}
    provenance['hardware']=(OUT/'hardware.csv').read_text()
    earlier=json.loads((ROOT/'artifacts/runtime-ab-20261015/provenance.json').read_text())['hashes']
    model_hashes={}
    for name in ('models/mf/model.safetensors','models/codec/weights.pth','artifacts/benchmark/inputs/reference.wav'):
        digest=hashlib.sha256()
        with (ROOT/name).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        model_hashes[name]=digest.hexdigest()
        assert model_hashes[name]==earlier[name]
    provenance['model_and_reference_sha256']=model_hashes
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    validation=dict(preserved=preservation,preserved_files=len(snapshot()),records=len(rs),measurements=len(rows),
        model_and_reference_preserved=True,
        normal=sum(r['kind']=='normal' for r in rs),profiles=sum(r['kind'] in ('stages','modules','operators') for r in rs),
        conv_processes=len(conv),telemetry_samples=len(telemetry),concurrent_compute_samples=len(concurrent),
        temperature_range=[min(t['temperature'] for t in telemetry),max(t['temperature'] for t in telemetry)],
        pcm_max_relative_l2=max(p['relative_l2'] for p in parity),
        codec_max_relative_l2=max(p['relative_l2'] for p in parity if p['phase']=='codec'),
        fp32_max_relative_l2=max(p['relative_l2'] for p in parity if p['phase']=='e2e' and p['precision']=='fp32'),
        bf16_max_relative_l2=max(p['relative_l2'] for p in parity if p['phase']=='e2e' and p['precision']=='bf16'),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('latent-*.pt')})
    assert not concurrent
    (DEST/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (LOCAL_REPORT/'summary.json').write_text(json.dumps(dict(measurements=rows,comparisons=comparisons,validation=validation),indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    colors=['#4c78a8','#e45756','#f2a541','#54a24b']
    variants=[('old','default'),('old924','default'),('new','default'),('new','benchmark')]
    labels=['Original','cuDNN only','Latest','Latest + search']
    vals=[lookup[k('confirmation' if mode=='default' else 'initial','codec',stack,mode,'fp32',8)]['ms'] for stack,mode in variants]
    axes[0].bar(range(4),vals,color=colors);axes[0].set_title('Fixed latent: 8 s DACVAE decode')
    for i,v in enumerate(vals):axes[0].text(i,v+.6,f'{v:.1f}',ha='center')
    vals=[st.median(r['gpu_ms'] for r in conv if r['stack']==s and r['mode']==m) for s,m in variants]
    axes[1].bar(range(4),vals,color=colors);axes[1].set_title('Single FP32 Conv1d: GPU time')
    for i,v in enumerate(vals):axes[1].text(i,v+.05,f'{v:.2f}',ha='center')
    x=np.arange(2);width=.19
    for i,((stack,mode),label) in enumerate(zip(variants,labels)):
        vals=[lookup[k('initial','e2e',stack,mode,p,24)]['ms'] for p in ('fp32','bf16')]
        axes[2].bar(x+(i-1.5)*width,vals,width,label=label,color=colors[i])
    axes[2].set_xticks(x,['MF FP32','MF BF16']);axes[2].set_title('24 s MF: reference cached')
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:ax.set_xticks(range(4),labels,rotation=22,ha='right')
    for ax in axes:ax.set_ylabel('Latency (ms), lower is faster');ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(DEST/'cause.png',dpi=160);plt.close(fig)
    markdown(lookup,comparisons,conv,validation)
    print(json.dumps(validation,indent=2))

def markdown(lookup,comparisons,conv,validation):
    def get(phase,stack,mode,precision,seconds,cache='native_cache',experiment='initial'):
        return lookup[(experiment,phase,stack,mode,precision,seconds,cache)]
    def comparison(label,seconds):
        return next(c for c in comparisons if c['phase']=='codec' and c['comparison']==label and c['seconds']==seconds)
    labels={'old':'旧環境','old924':'旧環境＋cuDNNのみ更新','new':'最新版','cudnn9241':'最新版＋cuDNN 9.24.1'}
    variants=[('old','default'),('old924','default'),('new','default'),('new','benchmark')]
    lines=['# 公式MF：最新版の速度低下の原因切り分け（2026-10-15）','',
        '**主因は、cuDNN 9.24のFP32 Conv1dにおける既定アルゴリズム選択。** '
        'PyTorch 2.10・CUDA 12.8を固定し、cuDNNだけを9.10.2→9.24.0に変更して速度低下を再現した。'
        '最大悪化層ではFFT＋GEMV経路を選択し、畳み込みの実測選択を有効にするとSGEMM系の経路で改善した。', '',
        '![切り分け結果](cause.png)','',
        '## 比較環境','',
        '| 条件 | PyTorch | CUDAランタイム | cuDNN | benchmark |','|---|---|---|---|---|',
        '| 旧環境 A | 2.10.0 | 12.8 | 9.10.2.21 | False |',
        '| cuDNNのみ更新 B | 2.10.0 | 12.8 | 9.24.0.43 | False |',
        '| 最新版 C | 2.14.1 | 13.0 | 9.24.0.43 | False |',
        '| パッチ比較 P | 2.14.1 | 13.0 | 9.24.1.1 | False |',
        '| 実測選択 C＋search | 2.14.1 | 13.0 | 9.24.0.43 | True |','',
        'RTX 5090、Python 3.10.20、CPUスレッド8。cuDNNはPyTorch最新版wheelの依存指定を基準とした。'
        'BはAからcuDNNのみ、PはCからcuDNNのみのパッケージ差であることをfreezeで確認した。'
        'B/Pは専用venvを作成し、診断目的の依存上書きとして`--no-deps`で構築した。'
        '実際にロードされたライブラリと実行時cuDNNバージョンも記録した。','',
        '## 1. 同一潜在入力でデコードを比較','',
        '旧環境の公式MF（p09、seed 42）で作った潜在入力を保存し、全環境で同じテンソルをデコード。'
        'A/B/Cは順序を反転する3ラウンド、各形状4回ウォームアップ後9回計測した。'
        '通常計測と演算・モジュールのプロファイルは分けている。以下は各ラウンド中央値の中央値（ms）。','',
        '| 条件 | 8秒音声 | 24秒音声 |','|---|---:|---:|']
    for stack,mode in variants:
        exp='confirmation' if mode=='default' else 'initial'
        vals=[get('codec',stack,mode,'fp32',sec,experiment=exp)['ms'] for sec in (8,24)]
        lines.append(f'| {labels[stack]}'+('＋実測選択' if mode=='benchmark' else '')+f' | {vals[0]:.2f} | {vals[1]:.2f} |')
    lines+=['','| 比較 | 8秒の差 | 24秒の差 | 24秒の95% CI |','|---|---:|---:|---|']
    for label,title in [('cuDNN only','A→B：cuDNNのみ'),('latest bundle','A→C：最新版全体'),('framework/CUDA residual','B→C：cuDNN版を揃えた残差'),('algorithm search','C→C＋search'),('cuDNN patch','C→P：9.24.1パッチ'),('legacy API','C→cuDNN v7 API')]:
        a,b=comparison(label,8),comparison(label,24)
        lines.append(f'| {title} | {a["delta_pct"]:+.2f}% | {b["delta_pct"]:+.2f}% | [{b["ci_low_pct"]:+.2f}, {b["ci_high_pct"]:+.2f}]% |')
    lines+=['',
        f'実測選択の行は初期測定（3ラウンド×7回）を使用した。search/v7比較の初期C基準値は'
        f'8秒{get("codec","new","default","fp32",8)["ms"]:.2f}ms、'
        f'24秒{get("codec","new","default","fp32",24)["ms"]:.2f}ms。'
        'A/B/Cの追加確認とは測定時刻が異なる。', '',
        'cuDNN単独更新と最新版全体で同程度の低下を再現した。9.24.1とv7 APIへの切り替えは回避にならなかった。'
        'NVIDIAは9.24.1で`cudnnBackendExecute`のCPU側オーバーヘッド修正を公表しているが、'
        '今回のGPU畳み込み経路の遅延はそのパッチでは解消しなかった。'
        '[NVIDIA公式リリースノート](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-24-1)。','',
        'CIは3プロセスブロックとブロック内試行を階層bootstrap（5,000回、seed 42）した参考区間。'
        'A/B/Cおよびパッチ確認は時間対応のあるブロック、それ以外の別実験間比較は独立ブロックとして再標本化した。'
        '同一GPU・固定入力の小規模測定であり、母集団全体の性能保証ではない。','',
        '## 2. Conv1d単層まで縮小','',
        '最大悪化層は`decoder.model.1.block.4.block.1`。8秒音声で入力`[1,768,2400]`、'
        '重み`[768,768,7]`、stride=1、padding=3、dilation=1、FP32。'
        'この形状を固定乱数テンソルと`torch.nn.functional.conv1d`だけで再現した。'
        'MF、DACVAE、TorchAudio、TorchCodec、TorchAOを読み込まない独立プロセスで、'
        '3ラウンド×ウォームアップ6回＋通常20回。GPU eventで測り、演算プロファイルは別実行した。','',
        '| 条件 | GPU時間 ms | 1回のConvで起動されるカーネル |','|---|---:|---|']
    for stack,mode in variants:
        group=[r for r in conv if r['stack']==stack and r['mode']==mode]
        value=st.median(r['gpu_ms'] for r in group)
        names='FFT 21＋GEMV 10＝31' if group[0]['fft_kernels'] else 'SGEMM系 1'
        lines.append(f'| {labels[stack]}'+('＋実測選択' if mode=='benchmark' else '')+f' | {value:.3f} | {names} |')
    lines+=['',
        'GPUトレースでは、旧版の`implicit_convolve_sgemm`から、9.24の`fft1d_r2c`／`fft1d_c2r`と'
        '`gemvx::kernel`へ変化した。実測選択で`implicit_convolve_sgemm`系に切り替わった。'
        '入力形状を揃えた単層でも再現するため、テキスト処理やMFのサンプリングを経由しない原因と確認できる。'
        'cuDNN内部のヒューリスティクスの判断理由や、最初に変化した中間バージョンの特定までは行っていない。','',
        '## 3. 公式MF全体で確認','',
        'p09、生成時間8/24秒、4ステップ、seed 42、モデル精度FP32/BF16、**codecは両方FP32**。'
        '参照再エンコード／参照潜在キャッシュの2条件で、3ラウンド×通常5回、別途段階プローブ3回を計測した。'
        'A/B/Cは順序を反転した同じ実験、C＋searchは別実験。以下は24秒音声の通常レイテンシ（ms）。','',
        '| モデル精度 | 参照条件 | 旧 A | cuDNNのみ B | 最新 C | C＋search |','|---|---|---:|---:|---:|---:|']
    for precision in ('fp32','bf16'):
        for cache in ('reencode','native_cache'):
            vals=[get('e2e',s,m,precision,24,cache)['ms'] for s,m in variants]
            lines.append(f'| {precision.upper()} | {cache} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['',
        '参照キャッシュ有効・24秒の段階別計測（ms）：','',
        '| モデル精度 | 条件 | MF sampling | DACVAE decode | 参照準備＋speaker |','|---|---|---:|---:|---:|']
    for precision in ('fp32','bf16'):
        for stack,mode in variants:
            r=get('e2e',stack,mode,precision,24)
            lines.append(f'| {precision.upper()} | {labels[stack]}'+('＋search' if mode=='benchmark' else '')+f' | {r["sampling_ms"]:.2f} | {r["decode_ms"]:.2f} | {r["reference_encode_ms"]:.2f} |')
    lines+=['',
        'MF samplingの差は小さく、デコードの増加が支配的。参照再エンコードでも、'
        'FP32条件の参照codecネットワークが24.62→29.25msへ増加する一方、音声読み込み・正規化は約20msのままだった。'
        'MFは4ステップで生成が短く、共通のFP32 codecの遅延が全体に占める割合が大きい。'
        'モデルをBF16にしてもcodecはFP32なので、同じデコードの遅延が残る。','',
        '段階プローブはGPU同期を挿入する補助計測であり、合計を通常レイテンシとはみなさない。'
        'モジュールGPU eventも実行を多少変えるため、原因の効果量には通常計測と単層再現を優先した。','',
        '## 回避設定の評価と範囲','',
        '`torch.backends.cudnn.benchmark = True`を専用ワーカー内だけで有効化すると、'
        '通常レイテンシはFP32/BF16とも旧環境以下になった。'
        '[PyTorch公式仕様](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cudnn.benchmark)では、'
        '複数の畳み込みアルゴリズムを実測して最速を選ぶ設定。'
        '形状初回の探索コストがあり、単層の初回呼び出しは最新版既定の約51msに対し実測選択で約112msだった。'
        'MF全体でのコールド時間や多様な動的形状への総合適用性は今回の判定対象に含めない。'
        '元環境の設定・エンジン実装には適用していない。','',
        f'固定潜在入力のcodec出力の相対L2最大値は{validation["codec_max_relative_l2"]:.2e}、'
        f'MF全体のFP32は{validation["fp32_max_relative_l2"]:.2e}。'
        f'BF16は最大{validation["bf16_max_relative_l2"]:.3f}で波形の同一性は確認できていない。'
        '波形差は位相差にも影響されるため、音質劣化の判定には使っていない。聴感/MOSは未評価。'
        '原因の判定は、同一潜在入力・FP32の比較と単層再現を主な証拠としている。','',
        '## 検証・保存','',
        f'- 通常{validation["normal"]}件、段階／モジュール／演算プローブ{validation["profiles"]}件。別途、Conv単層18プロセス×20回。',
        '- 全出力で48kHz、指定サンプル長、有限・非無音を確認。公式MFの段階プローブは全て4 forward。',
        f'- 元の3環境、前回の最新版3環境、制御用環境のmetadataと前回結果など計{validation["preserved_files"]}ファイルのSHA-256が前後一致。',
        '- MFモデル、codec重み、参照音声のSHA-256も前回の記録と一致。',
        f'- GPU監視{validation["telemetry_samples"]}サンプルでcomputeプロセスの重複0、温度{validation["temperature_range"][0]}–{validation["temperature_range"][1]}℃。監視は初期codec試験の後半から開始。初期試験も共通GPUロックで直列実行。',
        '- 生データ・固定潜在入力・GPUトレース・freeze・監視ログは`artifacts/mf-runtime-cause-20261015/`。追加環境は`.venvs/mf-runtime-cause-20261015/`。',
        '- 大きい演算別プロファイル`operators.csv`と重複集計`summary.json`は`artifacts/mf-runtime-cause-20261015/report/`に保存し、Git管理の対象外とする。',
        '- 集計は[measurements.csv](measurements.csv)、差とCIは[comparisons.csv](comparisons.csv)、層別は[modules.csv](modules.csv)、単層は[conv-probe.csv](conv-probe.csv)、検証は[validation.json](validation.json)。','',
        '## 再現手順','',
        'このワークスペースの元環境・モデルと、前回のruntime-ab環境を利用する。元環境へのinstall/syncは行わない。'
        '同じタグの完了済みラウンドは再利用するため、再測定時は別の`--tag`か専用の結果保存先を使う。','',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause_setup.py',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py capture --stacks old --seconds 2 8 24',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py codec --stacks old new --modes default',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py codec --stacks old new --modes benchmark v7 --seconds 8 24',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py codec --stacks old924 cudnn9241 --modes default benchmark',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py codec --stacks old old924 new --seconds 8 24 --repeats 9 --tag confirmation',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py codec --stacks new cudnn9241 --seconds 8 24 --repeats 9 --tag patch-confirmation',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py e2e --stacks old new old924 --seconds 8 24 --repeats 5',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause.py e2e --stacks new --modes benchmark --seconds 8 24 --repeats 5',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_conv_probe.py',
        'PYTHONDONTWRITEBYTECODE=1 benchmark/.venv/bin/python benchmark/mf_cause_report.py',
        '```','']
    (DEST/'RESULTS.ja.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
