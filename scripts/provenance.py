#!/usr/bin/env python3
"""Record source pins, tool versions, build configuration, and Python packages."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from native_toolchain import cuda_compiler

ROOT = Path(__file__).resolve().parents[1]

def output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

def main():
    data = {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "engines": json.loads((ROOT / "engines.lock.json").read_text()),
            "models": json.loads((ROOT / "models.lock.json").read_text()),
            "tools": {}, "python": {}, "cmake": {},
            "runtime_configuration": {"onnx": {"provider": "CUDAExecutionProvider",
                "model_precision": "fp16", "codec_encoder_precision": "fp32",
                "NVIDIA_TF32_OVERRIDE": "0", "cudnn_conv_algo_search": "HEURISTIC"}}}
    for name, cmd in {"gpu": ["nvidia-smi"], "cpu": ["lscpu", "--json"], "uv": ["uv", "--version"],
                      "cmake": ["cmake", "--version"], "gcc": ["gcc", "--version"],
                      "rustc": ["rustc", "--version"], "cargo": ["cargo", "--version"],
                      "cuda": [cuda_compiler() or "nvcc", "--version"]}.items():
        data["tools"][name] = output(cmd)
    data["nested_submodules"] = output(["git", "submodule", "status", "--recursive"])
    for name in ("Irodori-TTS", "Irodori-TTS-Optimiz", "onnx"):
        python = ROOT / f".venvs/{name}/bin/python"
        data["python"][name] = {"version": output([str(python), "--version"]),
                                 "packages": output(["uv", "pip", "freeze", "--python", str(python)])}
    for cache in (ROOT / "artifacts/build").glob("*/*/CMakeCache.txt"):
        entries = [line for line in cache.read_text().splitlines()
                   if not line.startswith(("#", "//")) and ":" in line and "=" in line]
        data["cmake"][str(cache.relative_to(ROOT))] = entries
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    (folder / "environment.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
