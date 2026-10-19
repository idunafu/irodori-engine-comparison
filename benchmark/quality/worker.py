"""Bounded subprocess protocol, with complete process-group cleanup."""
from __future__ import annotations
import json
import os
import queue
import signal
import subprocess
import threading
from common import BASE, CONFIG, HERE, ROOT, environment


def command(case):
    env = environment()
    if case['engine'] == 'official':
        return [str(ROOT / '.venvs/Irodori-TTS/bin/python'), str(HERE/'python_worker.py'),
                case['variant'],case['precision']],env
    if case['engine'] == 'speech':
        folder = ROOT / 'artifacts/benchmark/models'
        model = (next((folder/f"speech-{case['variant']}").glob('*.gguf')) if case['precision']=='fp32'
                 else folder/f"speech-{case['variant']}-f16.gguf")
        return [str(BASE/'build/speech-quality'),str(model),str(CONFIG['steps'][case['variant']])],env
    site = ROOT / '.venvs/onnx/lib/python3.10/site-packages'
    libs = [str(p) for p in (site/'nvidia').glob('*/lib')]
    libs += [str(site/'onnxruntime/capi'),env.get('LD_LIBRARY_PATH','')]
    env['LD_LIBRARY_PATH'] = ':'.join(libs)
    env['IRODORI_ORT_DLL'] = str(site/'onnxruntime/capi/libonnxruntime.so.1.23.2')
    return [str(BASE/'build/onnx-quality'),str(ROOT),case['precision']],env


class Process:
    def __init__(self, cmd, env, log, prefix='QUALITY ', timeout=None):
        self.timeout = timeout or CONFIG['request_timeout']
        self.queue = queue.Queue()
        self.log = open(log,'a',buffering=1)
        self.prefix = prefix
        self.proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,start_new_session=True)
        self.thread = threading.Thread(target=self.read,daemon=True)
        self.thread.start()
        try:
            self.ready = self.receive()
            if self.ready.get('type') != 'ready':
                raise RuntimeError(f'Worker startup failed: {self.ready}')
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            for line in self.proc.stdout:
                if self.prefix and not line.startswith(self.prefix):
                    self.log.write(line)
                    continue
                self.queue.put(json.loads(line[len(self.prefix):]))
        except BaseException as e:
            self.queue.put(dict(type='error',error='Invalid worker protocol: '+str(e)))
        finally:
            self.queue.put(dict(type='error',error='Worker exited',infrastructure=True))

    def receive(self):
        try:
            return self.queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError('Worker timeout') from None

    def send(self,value):
        self.proc.stdin.write(json.dumps(value,ensure_ascii=False)+'\n')
        self.proc.stdin.flush()

    def request(self,value):
        self.send(value)
        response=self.receive()
        if response.get('id') != value['id']:
            raise RuntimeError('Worker response ID mismatch: '+str(response))
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid,signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid,signal.SIGKILL)
                    self.proc.wait()
        self.thread.join(timeout=2)
        self.log.close()
