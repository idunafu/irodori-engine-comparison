"""Record hardware, exact source commits, model hashes, and dependency locks."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_toolchain import cuda_architectures, cuda_compiler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmark'
def command(cmd):return subprocess.check_output(cmd,text=True,cwd=ROOT,stderr=subprocess.DEVNULL).strip()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    from run import matrix, command as worker_command
    model_paths=[ROOT/f'models/{v}/model.safetensors' for v in ('rf','mf')]+[ROOT/'models/codec/weights.pth']
    model_paths.append(ROOT/'artifacts/codec/dacvae.safetensors')
    model_paths+=list((OUT/'models').glob('*.gguf'))+list((OUT/'models').glob('speech-*/*.gguf'))
    model_paths+=list((ROOT/'artifacts/onnx/export/ref').glob('*.onnx'))+list((ROOT/'artifacts/onnx/export/ref').glob('*.onnx.data'))
    hashes={}
    for p in model_paths:hashes[str(p.relative_to(ROOT))]=dict(sha256=digest(p),bytes=p.stat().st_size)
    sources={p.name:command(['git','-C',str(p),'rev-parse','HEAD']) for p in (ROOT/'engines').iterdir() if p.is_dir()}
    data=dict(recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),system=platform.platform(),cpu=command(['lscpu']),
        gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv,noheader']),
        cuda=command([cuda_compiler() or 'nvcc','--version']),uv=command(['uv','--version']),rust=command(['rustc','--version']),
        cmake=command(['cmake','--version']),sources=sources,
        nested_speech_ggml=command(['git','-C','engines/speech.cpp/ggml','rev-parse','HEAD']),
        models_lock=json.loads((ROOT/'models.lock.json').read_text()),model_hashes=hashes,
        reference=json.loads((OUT/'inputs/reference.json').read_text()),
        input_signature=dict(config_sha256=digest(ROOT/'benchmark/config.json'),prompts_sha256=digest(ROOT/'benchmark/prompts.json')),
        measurement=dict(cuda_arch=cuda_architectures(),cpu_threads=8,nvml_interval_ms=10,clock_changes=False,power_limit_changes=False,
            disk_caches='retained',primary_timing='request processing to host PCM, file I/O excluded',
            profiler='synchronized wall time, run separately from primary timing'),
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING','NVIDIA_TF32_OVERRIDE','VULKAN_SDK')})
    packages={}
    for env in ('Irodori-TTS','Irodori-TTS-Optimiz','onnx'):
        packages[env]=command(['uv','pip','freeze','--python',str(ROOT/f'.venvs/{env}/bin/python')]).splitlines()
    packages['benchmark']=command(['uv','pip','freeze','--python',str(ROOT/'benchmark/.venv/bin/python')]).splitlines()
    data['python_packages']=packages
    env_keys=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS',
              'HF_HUB_OFFLINE','WGPU_BACKEND','WGPU_BACKENDS','NVIDIA_TF32_OVERRIDE',
              'TOKENIZERS_PARALLELISM','IRODORI_ORT_DLL','LD_LIBRARY_PATH')
    data['workers']={}
    for case in matrix():
        cmd,env=worker_command(case)
        data['workers'][case['case']]=dict(command=cmd,environment={k:env[k] for k in env_keys if k in env})
    data['build_manifests']={str(p.relative_to(ROOT)):json.loads(p.read_text())
        for p in (OUT/'source').rglob('.bench-built-*.json')}
    data['cmake_settings']={}
    for p in (OUT/'build').glob('*/*/CMakeCache.txt'):
        settings={}
        for line in p.read_text().splitlines():
            if '=' not in line or line.startswith(('#','//')):continue
            key,value=line.split('=',1)
            name=key.split(':',1)[0]
            if name.startswith(('GGML_','ENGINE_','CUDAToolkit_','Vulkan_')) or name in (
                'CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER',
                'CMAKE_CXX_COMPILER','CMAKE_CUDA_FLAGS_RELEASE','CMAKE_CXX_FLAGS_RELEASE'):
                settings[name]=value
        data['cmake_settings'][str(p.relative_to(ROOT))]=settings
    data['rust_cargo_locks']={str(p.relative_to(ROOT)):digest(p)
        for engine in ('sttts-gui','Irodori-TTS-burn','irodori-onnx')
        for p in [OUT/'source'/engine/'Cargo.lock']}
    (OUT/'provenance.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Recorded',len(hashes),'model files and',len(sources),'engine commits')

if __name__=='__main__':main()
