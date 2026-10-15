#!/usr/bin/env python3
"""Generate one short WAV per supported GPU/model combination and validate it."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
UNSUPPORTED = {("Irodori-TTS-Optimiz", "mf"): "Pinned source lacks MeanFlow support; checkpoint flow_parameterization is rejected"}
CASES = [
    ("Irodori-TTS", "cuda", "rf"), ("Irodori-TTS", "cuda", "mf"),
    ("Irodori-TTS-Optimiz", "cuda", "rf"), ("Irodori-TTS-Optimiz", "cuda", "mf"),
    ("audio.cpp", "cuda", "rf"), ("audio.cpp", "vulkan", "rf"),
    ("speech.cpp", "cuda", "rf"), ("speech.cpp", "cuda", "mf"),
    ("speech.cpp", "vulkan", "rf"), ("speech.cpp", "vulkan", "mf"),
    ("Irodori-TTS-burn", "vulkan", "rf"), ("Irodori-TTS-burn", "vulkan", "mf"),
    ("sttts-gui", "vulkan", "rf"), ("sttts-gui", "vulkan", "mf"),
    ("irodori-onnx", "cuda", "mf"),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engines", nargs="*", metavar="ENGINE", help="Optional engine names; omitted means all engines")
    parser.add_argument("--model", choices=("rf", "mf"))
    args = parser.parse_args()
    unknown = set(args.engines) - {case[0] for case in CASES}
    if unknown:
        parser.error("Unknown engines: " + ", ".join(sorted(unknown)))
    path = ROOT / "artifacts/check-results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    failures = []
    selected = 0
    for engine, backend, model in CASES:
        if args.engines and engine not in args.engines:
            continue
        if args.model and model != args.model:
            continue
        selected += 1
        key = f"{engine}-{model}-{backend}"
        log = ROOT / "logs" / f"check-{key}.log"
        out = ROOT / "artifacts/smoke" / f"{key}.wav"
        cmd = ["bash", str(ROOT / "scripts/run.sh"), engine, "--backend", backend,
               "--model", model, "--output", str(out)]
        print(f"Checking {key}", flush=True)
        record = {"engine": engine, "backend": backend, "model": model,
                  "text": "こんにちは。音声合成の動作確認です。", "seed": 42,
                  "reference": "artifacts/voices/ref1.wav", "log": str(log.relative_to(ROOT)),
                  "command": cmd, "checked_at": datetime.now(timezone.utc).isoformat()}
        if engine == "irodori-onnx":
            record["runtime_configuration"] = {"model_precision": "fp16", "codec_encoder_precision": "fp32",
                                               "NVIDIA_TF32_OVERRIDE": "0"}
        if (engine, model) in UNSUPPORTED:
            record.update(status="upstream_unsupported", reason=UNSUPPORTED[(engine, model)])
            results[key] = record
            path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            print(f"UNSUPPORTED {key}: {record['reason']}", flush=True)
            continue
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=1200)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            audio, rate = sf.read(out, dtype="float32", always_2d=True)
            if rate != 48000 or len(audio) == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"Invalid audio: {rate=} samples={len(audio)}")
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if rms < 1e-5:
                raise RuntimeError(f"Silent output: {rms=}")
            record.update(status="audio_verified", sample_rate=rate, channels=audio.shape[1],
                          seconds=len(audio) / rate, rms=rms, output=str(out.relative_to(ROOT)))
            print(f"PASS {key}: {len(audio) / rate:.2f}s, RMS {rms:.4f}", flush=True)
        except (subprocess.SubprocessError, RuntimeError, OSError) as error:
            record.update(status="failed", error=str(error))
            failures.append(key)
            print(f"FAIL {key}: see {log}", flush=True)
        results[key] = record
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if not selected:
        parser.error("No prepared cases match the requested engine/model")
    if failures:
        raise SystemExit("Failed cases: " + ", ".join(failures))

if __name__ == "__main__":
    main()
