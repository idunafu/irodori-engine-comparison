"""Record GPU clocks, temperature and all compute/graphics processes without GPU work."""
import json
import time
from pathlib import Path
import pynvml as nv

ROOT=Path(__file__).resolve().parents[1]

def main():
    nv.nvmlInit()
    dev=nv.nvmlDeviceGetHandleByIndex(0)
    path=ROOT/'artifacts/benchmark/telemetry.jsonl'
    try:
        with path.open('a',buffering=1) as out:
            while True:
                processes={}
                for kind,query in (('compute',nv.nvmlDeviceGetComputeRunningProcesses),('graphics',nv.nvmlDeviceGetGraphicsRunningProcesses)):
                    for p in query(dev):
                        if p.pid not in processes:
                            try:name=nv.nvmlSystemGetProcessName(p.pid)
                            except nv.NVMLError:name=None
                            if isinstance(name,bytes):name=name.decode(errors='replace')
                            processes[p.pid]=dict(pid=p.pid,name=name,kinds=[],vram=None)
                        item=processes[p.pid];item['kinds'].append(kind)
                        if isinstance(p.usedGpuMemory,int) and p.usedGpuMemory<1<<60:item['vram']=max(item['vram'] or 0,p.usedGpuMemory)
                utilization=nv.nvmlDeviceGetUtilizationRates(dev)
                point=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),at=time.perf_counter(),
                    temperature=nv.nvmlDeviceGetTemperature(dev,nv.NVML_TEMPERATURE_GPU),
                    sm_clock=nv.nvmlDeviceGetClockInfo(dev,nv.NVML_CLOCK_SM),gpu_utilization=utilization.gpu,
                    memory_utilization=utilization.memory,gpu_used=nv.nvmlDeviceGetMemoryInfo(dev).used,
                    processes=list(processes.values()))
                out.write(json.dumps(point)+'\n')
                time.sleep(1)
    except KeyboardInterrupt:pass
    finally:nv.nvmlShutdown()

if __name__=='__main__':main()
