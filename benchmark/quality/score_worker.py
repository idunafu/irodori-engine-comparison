"""One resident scorer per process. GPU scorers never overlap synthesis."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import sys
import time
import signal
import numpy as np
import soundfile as sf
import soxr
from common import ASSETS, BASE, CONFIG, ROOT, environment
from worker import Process


def mono(path):
    audio,sr=sf.read(path,dtype='float32',always_2d=True)
    audio=audio.mean(axis=1)
    if sr!=16000:
        audio=soxr.resample(audio,sr,16000,quality='HQ')
    return np.asarray(audio,dtype=np.float32)


def emit(value):
    print('QUALITY '+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def main():
    def stop(sig,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    scorer=sys.argv[1]
    native=None
    if scorer in ('kana','whisper'):
        sys.path.insert(0,str(ASSETS/'jkyb-evaluator/src'))
        import torch
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        from jkyb_eval.asr import AsrConfiguration,WhisperTranscriber
        transcriber=WhisperTranscriber(AsrConfiguration(model=str(ASSETS/'models'/scorer),
            device='cuda:0',batch_size=1,num_workers=0,audio_workers=1))
    elif scorer in ('campplus','eres2net'):
        import sherpa_onnx
        config=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(ASSETS/'models'/scorer/CONFIG['models'][scorer]['file']),
            num_threads=8,debug=False,provider='cpu')
        extractor=sherpa_onnx.SpeakerEmbeddingExtractor(config)
    elif scorer in ('qwen','reazon'):
        cmd=[str(ROOT/'artifacts/benchmark/build/speech.cpp/vulkan/speech'),'worker',
             str(ASSETS/'models'/scorer/CONFIG['models'][scorer]['file']),
             '--device','gpu','--threads','8','--no-warmup']
        native=Process(cmd,environment(),BASE/f'{scorer}-native.log',prefix='')
        if native.ready.get('protocol')!=3:
            raise RuntimeError('Unsupported ASR worker protocol')
    else:
        raise ValueError(scorer)
    emit(dict(type='ready',scorer=scorer))
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('type')=='stop':
                break
            try:
                start=time.perf_counter()
                if scorer in ('kana','whisper'):
                    result=dict(transcript=next(iter(transcriber.transcribe([Path(request['wav'])]))))
                elif scorer in ('campplus','eres2net'):
                    stream=extractor.create_stream()
                    stream.accept_waveform(16000,mono(request['wav']))
                    stream.input_finished()
                    if not extractor.is_ready(stream):
                        raise ValueError('Insufficient audio for speaker embedding')
                    embedding=np.asarray(extractor.compute(stream),dtype=np.float32)
                    norm=np.linalg.norm(embedding)
                    if not np.isfinite(norm) or norm==0:
                        raise ValueError('Invalid speaker embedding')
                    result=dict(embedding=(embedding/norm).tolist())
                else:
                    audio=np.clip(mono(request['wav']),-1,1)
                    pcm=np.round(audio*32767).astype('<i2').tobytes()
                    for seq,offset in enumerate(range(0,len(pcm),32000)):
                        native.send(dict(type='chunk',id=request['id'],seq=seq,
                            pcm=base64.b64encode(pcm[offset:offset+32000]).decode()))
                    args=dict(type='transcribe',id=request['id'],sample_rate=16000)
                    if scorer=='qwen':
                        args['language']='ja'
                    native.send(args)
                    while True:
                        out=native.receive()
                        if out.get('id')!=request['id']:
                            raise RuntimeError('ASR response ID mismatch')
                        if out['type']=='progress':
                            continue
                        if out['type']!='end' or out.get('stop')!='complete':
                            raise RuntimeError(str(out))
                        result=dict(transcript=out['text'])
                        break
                result.update(type='result',id=request['id'],ms=(time.perf_counter()-start)*1000)
                emit(result)
            except Exception as error:
                import traceback
                traceback.print_exc(file=sys.stderr)
                emit(dict(type='error',id=request.get('id'),error=f'{type(error).__name__}: {error}'))
    finally:
        if native:
            native.close()


if __name__=='__main__':
    main()
