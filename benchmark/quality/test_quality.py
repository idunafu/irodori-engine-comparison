"""Durability, split isolation and scoring tests that do not require a GPU or JVS."""
import json
import sys
from pathlib import Path
import subprocess
import time
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ASSETS,HERE,ROOT,digest,environment,stratified
from store import Store
from worker import Process
from run import audio_metrics,infrastructure
from run import Campaign,ensure_manifest
from report import bootstrap_difference


def spec():
    return dict(item=dict(id='a',dataset='jkyb'),case=dict(id='official-mf-bf16',profile='practical'),seed=0,mode='native')


def test_resume_verifies_saved_audio_and_invalidates_derived_scores(tmp_path):
    s=Store(tmp_path);s.populate([('a',spec(),True)])
    wav=tmp_path/'audio.wav';wav.write_bytes(b'first')
    s.outcome('a','ok',{'frames':25},wav)
    h=s.db.execute('SELECT wav_sha256 FROM tasks').fetchone()[0]
    s.score('a','kana','ok',{'transcript':'ア'},h);s.verify()
    s.db.close();s=Store(tmp_path);s.verify()
    s.populate([('a',spec(),True)])
    assert s.db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==1
    wav.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):s.verify()
    s.outcome('a','ok',{},wav)
    assert s.db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]==0
    s.verify();s.db.close()


def test_splits_are_deterministic_and_do_not_depend_on_input_order():
    items=[dict(id=str(i),group=str(i%3)) for i in range(99)]
    a=stratified(items,12,lambda x:x['group'],'frozen')
    b=stratified(list(reversed(items)),12,lambda x:x['group'],'frozen')
    assert a==b and len({x['id'] for x in a})==12
    assert [sum(x['group']==g for x in a) for g in ['0','1','2']]==[4,4,4]


def test_jvs_index_uses_recordings_without_shifting_fixed_targets(tmp_path,monkeypatch):
    import prepare
    import soundfile as sf
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    sp=tmp_path/'jvs001'
    for subset,count in [('parallel100',16),('nonpara30',5)]:
        path=sp/subset
        (path/'wav24kHz16bit').mkdir(parents=True)
        (path/'transcripts_utf8.txt').write_text(''.join(f'p{i:03}:テスト{i}\n' for i in range(count)))
        ids=range(15) if subset=='parallel100' else [0,2,4]
        for i in ids:
            sf.write(path/'wav24kHz16bit'/f'p{i:03}.wav',np.zeros(240,dtype=np.float32),24000,subtype='PCM_16')
    result=prepare.speaker_record(sp,'M')
    assert [x['id'] for x in result['parallel'][:15]]==[f'p{i:03}' for i in range(15)]
    assert [x['id'] for x in result['references']]==['p000','p002','p004']
    assert result['transcript_only']=={'parallel100':['p015'],'nonpara30':['p001','p003']}
    (sp/'parallel100/wav24kHz16bit/p003.wav').unlink()
    with pytest.raises(FileNotFoundError,match='fixed target/held-out'):
        prepare.speaker_record(sp,'M')


def test_resume_refuses_changed_conditions_without_overwriting_manifest(tmp_path):
    original={'format':1,'files':{'model':'first'},'fingerprint':'first'}
    ensure_manifest(tmp_path,original);ensure_manifest(tmp_path,original)
    with pytest.raises(ValueError,match='refusing resume'):
        ensure_manifest(tmp_path,dict(original,fingerprint='changed'))
    assert json.loads((tmp_path/'manifest.json').read_text())==original


def test_campaign_recovers_worker_crash_and_skips_finished_requests(tmp_path,monkeypatch):
    import run as runner
    script=tmp_path/'fake-worker.py'
    script.write_text('''import json,os,sys
import numpy as np,soundfile as sf
print('QUALITY '+json.dumps(dict(type='ready')),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['id']=='b':os._exit(2)
 sf.write(r['output'],np.full(48000,.1,dtype=np.float32),48000,subtype='FLOAT')
 print('QUALITY '+json.dumps(dict(type='result',id=r['id'],frames=25,samples=48000,ms=1)),flush=True)
''')
    monkeypatch.setattr(runner,'command',lambda case:([sys.executable,str(script)],environment()))
    c=Campaign(tmp_path)
    try:
        enumeration=[]
        for key in ['a','b','c']:
            s=spec();s['item'].update(text='テスト',references=[],caption='')
            enumeration.append((key,s,True))
        c.store.populate(enumeration)
        c.generate('1','test')
        assert c.store.counts()=={'failed':1,'ok':2}
        before=list(c.store.db.execute('SELECT id,wav_sha256 FROM tasks ORDER BY id'))
        c.generate('1','resume')
        assert c.state['phase_total']==0
        assert [tuple(x) for x in before]==[tuple(x) for x in c.store.db.execute('SELECT id,wav_sha256 FROM tasks ORDER BY id')]
        c.store.verify()
    finally:
        c.close_worker();c.finished.set();c.thread.join(timeout=2);c.store.db.close()


def test_audio_guardrails_and_failure_classification():
    with pytest.raises(ValueError,match='Empty'):audio_metrics(np.zeros(0))
    with pytest.raises(ValueError,match='Silent'):audio_metrics(np.zeros(48000))
    with pytest.raises(ValueError,match='Non-finite'):audio_metrics(np.full(48000,np.nan))
    audio=np.concatenate([np.zeros(960),np.full(4800,.1),np.zeros(1920)])
    m=audio_metrics(audio)
    assert m['leading_silence_seconds']==.02
    assert m['trailing_silence_seconds']==.04
    assert infrastructure('CUDA out of memory')
    assert not infrastructure('predicted length exceeds 30 seconds')


def test_cluster_interval_preserves_paired_clusters_and_weights():
    a={'x':np.array([1.,2.]),'y':np.array([2.,4.])}
    b={'x':np.array([0.,2.]),'y':np.array([0.,4.])}
    r=bootstrap_difference(a,b,100)
    assert r['difference']==.5 and r['low']==.5 and r['high']==.5 and r['clusters']==2
    assert bootstrap_difference(a,{'z':np.array([1.,1.])},100) is None


@pytest.mark.skipif(not (ASSETS/'jkyb/data/common_kanji_source.jsonl').exists(),reason='prepare dataset first')
def test_full_matrix_nested_references_and_held_out_separation(tmp_path,monkeypatch):
    import dataset
    (tmp_path/'jkyb/data').mkdir(parents=True)
    (tmp_path/'jkyb/data/common_kanji_source.jsonl').symlink_to(ASSETS/'jkyb/data/common_kanji_source.jsonl')
    (tmp_path/'speech-bench/prompts').mkdir(parents=True)
    (tmp_path/'speech-bench/prompts/speak-ja-JP.json').symlink_to(ASSETS/'speech-bench/prompts/speak-ja-JP.json')
    speakers=[]
    for i in range(100):
        name=f'jvs{i+1:03}'
        def clip(prefix,n):return dict(id=f'{prefix}{n:03}',path=f'{name}/{prefix}{n:03}.wav',text='テスト',seconds=4.1,sha256='fixture')
        speakers.append(dict(id=name,gender='M' if i%2 else 'F',
                             parallel=[clip('p',n) for n in range(15)],references=[clip('r',n) for n in range(30)]))
    (tmp_path/'jvs-index.json').write_text(json.dumps(speakers))
    monkeypatch.setattr(dataset,'ASSETS',tmp_path)
    j,jvs,p,split=dataset.inputs()
    assert len(j)==13536 and len(jvs)==2000 and len(p)==40
    assert len(split['diagnostic_jkyb'])==500 and len(split['diagnostic_jvs'])==20
    for speaker in speakers:
        rows=[r for r in jvs if r['speaker']==speaker['id']]
        refs=[next(r['references'] for r in rows if r['kind']==k) for k in ['one','30','60','120']]
        assert all(short==long[:len(short)] for short,long in zip(refs,refs[1:]))
        assert not set(refs[-1]) & {str(ROOT/x['path']) for x in speaker['parallel'][:15]}
    from collections import Counter
    counts=Counter();pilot=0
    for key,s,is_pilot in dataset.tasks():
        counts[(s['case']['profile'],s['mode'])]+=1
        pilot+=is_pilot
    assert counts=={('practical','native'):389400,('fp32','native'):23500,
                   ('practical','common'):1000,('fp32','common'):1000}
    assert pilot==8500
    # A separate reading campaign must not touch a missing JVS index or reference audio.
    (tmp_path/'jvs-index.json').unlink()
    j,voices,p,selection=dataset.inputs(['jkyb','practical'])
    assert len(j)==13536 and not voices and len(p)==40
    assert selection['diagnostic_jvs']==[] and selection['pilot_jvs']==[]
    selected=Counter()
    for key,s,is_pilot in dataset.tasks(['jkyb','practical']):
        assert s['item']['dataset']!='jvs'
        selected[(s['case']['profile'],s['mode'])]+=1
    assert selected=={('practical','native'):339400,('fp32','native'):13500,
                     ('practical','common'):1000,('fp32','common'):1000}
    practical=list(dataset.tasks(['practical']))
    assert len(practical)==4000
    assert {s['item']['dataset'] for _,s,_ in practical}=={'practical'}


def test_worker_timeout_exit_and_child_cleanup(tmp_path):
    script=tmp_path/'worker.py'
    script.write_text('import time\nprint(\'QUALITY {"type":"ready"}\',flush=True)\ntime.sleep(30)\n')
    w=Process([sys.executable,str(script)],environment(),tmp_path/'worker.log',timeout=.2)
    with pytest.raises(TimeoutError):w.receive()
    w.close()
    assert w.proc.poll() is not None
    script.write_text('print(\'QUALITY {"type":"ready"}\',flush=True)\n')
    w=Process([sys.executable,str(script)],environment(),tmp_path/'exit.log',timeout=2)
    assert w.receive()['infrastructure'];w.close()


@pytest.mark.skipif(not (ASSETS/'jkyb-evaluator/src').exists(),reason='prepare evaluator first')
def test_upstream_reading_scores_empty_and_repeated_outputs():
    sys.path.insert(0,str(ASSETS/'jkyb-evaluator/src'))
    from jkyb_eval.models import BenchmarkRow
    from jkyb_eval.metrics import score_row
    raw=json.loads((ASSETS/'jkyb/data/common_kanji_source.jsonl').read_text().splitlines()[0])
    row=BenchmarkRow.from_dict(raw,location='test')
    right=score_row(row,row.yomi,text_prediction=row.text,score_text=True)
    missing=score_row(row,None,text_prediction=None,score_text=True)
    repeated=score_row(row,row.yomi*3,text_prediction=row.text*3,score_text=True)
    assert right['target_exact'] and right['sentence_kana_cer']==0 and right['text_cer']==0
    assert missing['sentence_kana_cer']==1 and missing['text_cer']==1
    assert repeated['sentence_kana_cer']>0 and repeated['text_cer']>0


@pytest.mark.skipif(not (ASSETS/'speech-bench/src/measure/quality-heard.ts').exists(),reason='build helper first')
def test_upstream_heard_spelling_numbers_and_repetition(tmp_path):
    w=Process(['node',str(HERE/'speech_bench_worker.mjs')],environment(),tmp_path/'node.log')
    try:
        right=w.request(dict(id='right',text='午前十時です。',transcript='午前10時です'))
        assert right['heard_errors']==0 and right['heard_as_said']
        spelling=w.request(dict(id='spelling',text='あー。',transcript='ああ'))
        assert spelling['heard_as_said']
        repeat=w.request(dict(id='repeat',text='はい。',transcript='はい'*20))
        assert repeat['heard_raw_errors']>repeat['heard_errors']
        assert not repeat['heard_as_said']
    finally:w.close()


@pytest.mark.skipif(not (ASSETS/'speech-bench/src/measure/quality-heard.ts').exists(),reason='prepare sources first')
def test_report_rebuilds_saved_reading_speaker_and_practical_metrics(tmp_path,monkeypatch):
    import report as reporting
    import soundfile as sf
    raw=json.loads((ASSETS/'jkyb/data/common_kanji_source.jsonl').read_text().splitlines()[0])
    embedding=[1.]+[0.]*191
    monkeypatch.setattr(reporting,'centroids',lambda path:dict(centroids={'jvs001':embedding}))
    (tmp_path/'splits.json').write_text(json.dumps(dict(diagnostic_jkyb=['j'],diagnostic_jvs=['jvs001'])))
    audio=(.1*np.sin(np.arange(96000)*2*np.pi*200/48000)).astype(np.float32)
    wav=tmp_path/'example.wav';sf.write(wav,audio,48000,subtype='FLOAT')
    s=Store(tmp_path)
    for engine,precision in [('official','bf16'),('onnx','fp16')]:
        case=dict(id=f'{engine}-mf-{precision}',engine=engine,precision=precision,profile='practical',variant='mf')
        for name in ['jkyb','jvs','practical']:
            item=dict(id='j' if name=='jkyb' else name,dataset=name,kind='test',cluster='cluster',
                      text=raw['text'] if name=='jkyb' else '今日は晴れです。',references=[],caption='')
            if name=='jkyb':item['benchmark']=raw
            if name=='jvs':item['speaker']='jvs001'
            spec=dict(item=item,case=case,seed=0,mode='native')
            key=engine+name;s.populate([(key,spec,True)])
            metrics=audio_metrics(audio);metrics.update(frames=50,seconds_per_character=2/len(item['text']))
            s.outcome(key,'ok',metrics,wav)
            h=s.db.execute('SELECT wav_sha256 FROM tasks WHERE id=?',(key,)).fetchone()[0]
            if name=='jkyb':
                s.score(key,'kana','ok',dict(transcript=raw['yomi']),h)
                s.score(key,'whisper','ok',dict(transcript=raw['text']),h)
            if name=='jvs':
                s.score(key,'campplus','ok',dict(embedding=embedding),h)
                s.score(key,'whisper','ok',dict(transcript=item['text']),h)
            if name=='practical':
                for scorer in ['qwen','reazon']:s.score(key,scorer,'ok',dict(transcript=item['text']),h)
                s.score(key,'eres2net','ok',dict(embedding=embedding),h)
    s.db.close()
    reporting.report(tmp_path)
    first=(tmp_path/'report/summary.csv').read_bytes()
    reporting.report(tmp_path)
    assert (tmp_path/'report/summary.csv').read_bytes()==first
    assert 'camplus_cosine' in first.decode() and 'heard_error_rate' in first.decode()
    assert len((tmp_path/'report/per-input.jsonl').read_text().splitlines())==6
    assert (tmp_path/'report/comparisons.csv').stat().st_size>0
