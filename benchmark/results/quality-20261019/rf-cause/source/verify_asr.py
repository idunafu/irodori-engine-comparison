"""A bounded third-recognizer check on eight saved waveforms; run in detached tmux."""
import fcntl
import json
from pathlib import Path
import signal
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[5]
sys.path.insert(0,str(ROOT/'benchmark/quality'))
from common import HERE,PYTHON,atomic_json,environment,sha256,utc
from worker import Process


def main():
    state=dict(status='running',stage='waiting-for-gpu-lock',completed=0,total=8,started_at=utc())
    target=OUT/'whisper-check.jsonl'
    worker=None
    def save():atomic_json(OUT/'whisper-status.json',dict(state,updated_at=utc()))
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        with (ROOT/'artifacts/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['stage']='whisper-check';save()
            examples=[r for r in json.loads((OUT/'transcript-examples.json').read_text()) if r['mode']=='native']
            assert len(examples)==8
            prior={r['id']:r for r in map(json.loads,target.read_text().splitlines())} if target.exists() else {}
            with target.open('a',buffering=1) as output:
                for row in examples:
                    assert sha256(Path(row['wav']))==row['wav_sha256']
                    if row['id'] not in prior:
                        if worker is None:worker=Process([str(PYTHON),str(HERE/'score_worker.py'),'whisper'],environment(),OUT/'whisper-worker.log')
                        result=worker.request(dict(id=row['id'],wav=row['wav']))
                        if result.get('type')!='result':raise RuntimeError(result)
                        output.write(json.dumps(dict(**row,result=result),ensure_ascii=False)+'\n')
                    else:assert prior[row['id']]['wav_sha256']==row['wav_sha256']
                    state['completed']+=1;save()
        state.update(status='complete',exit_code=0,finished_at=utc());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',exit_code=1,finished_at=utc());save()
        raise
    finally:
        if worker:worker.close()


if __name__=='__main__':main()
