"""Prepare identical inputs without modifying the original reference recording."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    cfg = json.loads((ROOT / 'benchmark/config.json').read_text())
    source = ROOT / cfg['reference']
    info = sf.info(source)
    audio, sr = sf.read(source, dtype='float32', always_2d=True)
    assert sr == 48000 and audio.shape[1] == 1, 'Reference must be mono 48 kHz'
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < 1
    dest = ROOT / 'artifacts/benchmark/inputs'
    dest.mkdir(parents=True, exist_ok=True)
    reference = dest / 'reference.wav'
    # Write ordinary RIFF WAV: some native readers do not accept WAVEX.
    sf.write(reference, audio, sr, subtype='PCM_24', format='WAV')
    check, _ = sf.read(reference, dtype='float32', always_2d=True)
    assert np.array_equal(audio, check), 'PCM samples changed during container conversion'
    metadata = dict(source=str(source), source_sha256=sha256(source), prepared=str(reference),
                    prepared_sha256=sha256(reference), sample_rate=sr, channels=1,
                    samples=len(audio), seconds=len(audio)/sr, subtype=info.subtype,
                    preprocessing='WAVEX to RIFF PCM_24 container only; samples unchanged')
    (dest / 'reference.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    tok = Tokenizer.from_file(str(ROOT / 'models/rf/tokenizer/tokenizer.json'))
    count = lambda s: 1 + len(tok.encode(s, add_special_tokens=False).ids)
    seeds = [
        '今日はよく晴れています。',
        '次の電車は三番線に到着します。',
        '水は温度によって姿を変えます。',
        '小さな猫が窓の外を眺めていました。',
    ]
    additions = [
        ['朝は窓を開けて、新鮮な空気を部屋に入れました。', '昼休みには友人と近くの公園を歩き、帰りにパンを買いました。', '夕方になったら、家族と今日あったことを話そうと思います。'],
        ['ご乗車のお客様は黄色い線の内側でお待ちください。', '車内では携帯電話を静かな設定にして、周りの方への配慮をお願いします。', 'お忘れ物のないよう、お降りの際には座席の周辺をご確認ください。'],
        ['冷やすと氷になり、温めると蒸気になります。', 'このような変化は身近な生活でも観察でき、料理や天気を理解する手がかりになります。', '条件を一つずつ変えて観察すると、原因と結果の関係が分かりやすくなります。'],
        ['遠くの丘から、ゆっくりと風が吹いてきました。', '猫は耳を動かすと、机の上に置かれた古い地図を見つめ、まだ知らない町の景色を想像しました。', 'やがて玄関の扉が開き、旅から戻った主人の声を聞くと、軽い足取りで迎えに走りました。'],
    ]
    bounds = [(8,15),(16,31),(32,63),(64,95),(96,127)]
    endings = [
        ['帰りに買い物をします。','夕食が楽しみです。','気持ちの良い一日です。'],
        ['足元にご注意ください。','どうぞお待ちください。','ご協力をお願いします。'],
        ['変化の様子を観察します。','温度を測ります。','結果を記録します。'],
        ['猫は静かに待っていました。','空は青く澄んでいました。','穏やかな朝でした。'],
    ]
    prompts = []
    for bucket, (lo, hi) in enumerate(bounds):
        for genre in range(4):
            text = seeds[genre]
            k = 0
            target = (lo + hi)//2
            while count(text) < target:
                fragment = additions[genre][k % len(additions[genre])]
                if count(text+fragment) <= hi:
                    text += fragment
                else:
                    for ending in endings[genre]:
                        candidate = text + ending
                        if lo <= count(candidate) <= hi:
                            text = candidate
                            break
                    else: break
                    break
                k += 1
            n = count(text)
            assert lo <= n <= hi, (text, n, lo, hi)
            prompts.append(dict(id=f'p{len(prompts)+1:02}', bucket=f'{lo}-{hi}',
                                genre=['conversation','announcement','explanation','story'][genre],
                                text=text, normalized_text=text, characters=len(text), tokens=n))
    (ROOT/'benchmark/prompts.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False))
    print('tokens:', [p['tokens'] for p in prompts])

if __name__ == '__main__': main()
