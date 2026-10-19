"""Autonomous asset preparation with bounded retries for an unavailable public corpus."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import json
from common import ASSETS, BASE, DATASETS, HERE, PYTHON, ROOT, atomic_json, utc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    directory=BASE/'runs'/args.run_id
    directory.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),stage='preparing-inputs',run_id=args.run_id,
               started=utc(),phase_total=0,phase_done=0,phase_failed=0,last_progress_at=utc())
    child=None
    stop=threading.Event()
    def save():
        atomic_json(directory/'status.json',dict(state,heartbeat_at=utc()))
    def heartbeat():
        while not stop.wait(30):save()
    def terminate(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        while True:
            state.update(stage='preparing-inputs',last_progress_at=utc());save()
            log_path=directory/'prepare.log'
            offset=log_path.stat().st_size if log_path.exists() else 0
            with (directory/'prepare.log').open('a',buffering=1) as log:
                child=subprocess.Popen([str(PYTHON),str(HERE/'prepare.py'),'--datasets',*args.datasets],cwd=ROOT,
                                       stdout=log,stderr=log,start_new_session=True)
                code=child.wait();child=None
            if code==0:break
            with log_path.open() as log:
                log.seek(offset)
                text=log.read()[-6000:]
            if 'Too many users' not in text and 'downloadQuotaExceeded' not in text:
                raise RuntimeError('Asset preparation failed; inspect prepare.log')
            state.update(stage='waiting-for-jvs-download-quota',
                reason='Official public JVS Google Drive download quota exceeded',
                next_attempt_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(time.time()+6*3600)))
            save()
            print(utc(),'JVS download quota exceeded; next attempt in six hours',flush=True)
            # Script-owned backoff, not an agent monitor. SIGTERM interrupts the wait.
            stop.wait(6*3600)
        stop.set();thread.join(timeout=2)
        argv=[str(PYTHON),str(HERE/'run.py'),'--run-id',args.run_id]
        argv+=['--datasets',*args.datasets]
        if (directory/'manifest.json').exists():argv.append('--resume')
        if args.retry_failed:argv.append('--retry-failed')
        os.execv(PYTHON,argv)
    except KeyboardInterrupt:
        state.update(status='interrupted',finished_at=utc(),exit_code=130)
        return 130
    except BaseException as error:
        state.update(status='failed',error=str(error),finished_at=utc(),exit_code=1)
        raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        stop.set();thread.join(timeout=2);save()


if __name__=='__main__':raise SystemExit(main())
