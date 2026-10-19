"""Download pinned evaluation inputs. No synthesis and no changes to existing engines."""
from __future__ import annotations

import io
import argparse
import fcntl
import json
import subprocess
from pathlib import Path
import tarfile
import urllib.request
import zipfile

from common import ASSETS, CONFIG, DATASETS, ROOT, atomic_json, sha256


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.download')
        print('Download', url, flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def repository(repo, commit, name):
    dest = ASSETS / name
    if not (dest / '.quality-revision').exists():
        archive = fetch(f'https://codeload.github.com/{repo}/tar.gz/{commit}', ASSETS / f'{name}.tar.gz')
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            prefix = tf.getmembers()[0].name.split('/')[0] + '/'
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                rel = Path(m.name.removeprefix(prefix))
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe archive member')
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())
        (dest / '.quality-revision').write_text(commit + '\n')
    if (dest / '.quality-revision').read_text().strip() != commit:
        raise ValueError(f'Revision mismatch: {dest}')
    return dest


def speaker_record(sp, gender=None):
    """Index released recordings; JVS transcripts can include unrecorded prompts."""
    import soundfile as sf
    record = {'id': sp.name, 'gender': gender, 'parallel': [], 'references': [],
              'transcript_only': {}}
    for subset, field in [('parallel100', 'parallel'), ('nonpara30', 'references')]:
        transcripts = dict(line.split(':', 1) for line in
                           (sp / subset / 'transcripts_utf8.txt').read_text().splitlines() if line)
        recordings = {wav.stem: wav for wav in (sp / subset / 'wav24kHz16bit').glob('*.wav')}
        unknown = sorted(recordings.keys() - transcripts.keys())
        if unknown:
            raise ValueError(f'Recordings without transcripts: {sp.name}/{subset}: {unknown}')
        absent = sorted(transcripts.keys() - recordings.keys())
        record['transcript_only'][subset] = absent
        if field == 'parallel':
            # Missing later recordings must never shift the fixed targets/held-out set.
            required = sorted(transcripts)[:15]
            missing_required = sorted(set(required) - recordings.keys())
            if len(required) < 15 or missing_required:
                raise FileNotFoundError(f'Missing fixed target/held-out recordings: {sp.name}: {missing_required}')
        for key, wav in sorted(recordings.items()):
            info = sf.info(wav)
            record[field].append(dict(id=key, text=transcripts[key], path=str(wav.relative_to(ROOT)),
                                      seconds=info.duration, sha256=sha256(wav)))
    if len(record['parallel']) < 15 or not record['references']:
        raise ValueError(f'Incomplete speaker: {sp.name}')
    return record


def main(datasets=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    ASSETS.mkdir(parents=True, exist_ok=True)
    repository('Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition', CONFIG['jkyb_evaluator'], 'jkyb-evaluator')
    repository('nyosegawa/speech-bench', CONFIG['speech_bench'], 'speech-bench')
    hf_hub_download('Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet', 'data/common_kanji_source.jsonl',
                    repo_type='dataset', revision=CONFIG['jkyb_revision'], local_dir=ASSETS / 'jkyb')
    pinned = {}
    for name, m in CONFIG['models'].items():
        dest = ASSETS / 'models' / name
        if 'file' in m:
            path = Path(hf_hub_download(m['repo'], m['file'], revision=m['revision'], local_dir=dest))
            h = sha256(path)
            if m.get('sha256') and h != m['sha256']:
                raise ValueError(f'Bad model hash: {name}')
            files = {m['file']: h}
        else:
            snapshot_download(m['repo'], revision=m['revision'], local_dir=dest,
                              allow_patterns=['*.json', '*.safetensors', 'tokenizer.*', '*.txt', '*.model'],
                              ignore_patterns=['*msgpack*', '*flax*', '*tf_model*'])
            files = {str(p.relative_to(dest)): sha256(p) for p in dest.rglob('*') if p.is_file() and '.cache' not in p.parts}
        pinned[name] = dict(**m, files=files)
    atomic_json(ASSETS / 'models.lock.json', pinned)
    packages={}
    for name in ['Irodori-TTS','onnx']:
        code='import json; from importlib.metadata import distributions; print(json.dumps({d.metadata["Name"].lower():d.version for d in distributions()}))'
        packages[name]=json.loads(subprocess.check_output([str(ROOT/f'.venvs/{name}/bin/python'),'-c',code],text=True))
    if packages['Irodori-TTS'].get('torch')!='2.10.0+cu128' or packages['onnx'].get('onnxruntime-gpu')!='1.23.2':
        raise ValueError('Existing primary-comparison runtime versions are required')
    atomic_json(ASSETS/'runtime-packages.json',packages)
    if datasets is not None and 'jvs' not in datasets:
        print('Quality inputs ready (JVS excluded)',flush=True)
        return
    # Official distribution; retain original archive hash and never redistribute reference recordings.
    jvs = ASSETS / 'jvs'
    if not list(jvs.glob('**/jvs001/parallel100/transcripts_utf8.txt')):
        import gdown
        archive = ASSETS / 'jvs_ver1.zip'
        if not archive.exists():
            gdown.download(CONFIG['jvs_url'], str(archive), quiet=False, resume=True)
        jvs.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                rel = Path(member.filename)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe JVS archive')
            z.extractall(jvs)
    paths = sorted(p for p in jvs.glob('**/jvs[0-9][0-9][0-9]')
                   if (p/'parallel100/transcripts_utf8.txt').is_file())
    if len(paths) != 100:
        raise ValueError(f'Expected 100 JVS speakers, got {len(paths)}')
    jvs_index = []
    genders = {}
    for metadata in jvs.glob('**/gender_f0range.txt'):
        for line in metadata.read_text().splitlines():
            fields=line.split()
            if len(fields)>=2 and fields[0].startswith('jvs'):
                genders[fields[0]]=fields[1]
    for sp in paths:
        jvs_index.append(speaker_record(sp, genders.get(sp.name)))
    atomic_json(ASSETS / 'jvs-index.json', jvs_index)
    atomic_json(ASSETS / 'jvs-index-audit.json', dict(
        speakers=len(jvs_index),
        parallel_recordings=sum(len(s['parallel']) for s in jvs_index),
        reference_recordings=sum(len(s['references']) for s in jvs_index),
        fixed_target_and_held_out_recordings=len(jvs_index)*15,
        transcript_only_entries={s['id']:s['transcript_only'] for s in jvs_index},
        policy='Use available recording IDs in sorted order; require the first 15 parallel transcript IDs without substitution'))
    atomic_json(ASSETS / 'models.lock.json', pinned)
    atomic_json(ASSETS / 'download-manifest.json', dict(jvs_archive=sha256(ASSETS / 'jvs_ver1.zip'),
        jkyb_sha256=sha256(ASSETS / 'jkyb/data/common_kanji_source.jsonl'),
        speech_bench=CONFIG['speech_bench'], jkyb_evaluator=CONFIG['jkyb_evaluator']))
    print('Quality inputs ready', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    args=parser.parse_args()
    ASSETS.mkdir(parents=True,exist_ok=True)
    with (ASSETS/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main(args.datasets)
