#!/usr/bin/env python3
"""Launch a pinned engine with workspace-local models, caches, and GPU locking."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("Irodori-TTS", "Irodori-TTS-Optimiz", "audio.cpp", "speech.cpp",
           "sttts-gui", "Irodori-TTS-burn", "irodori-onnx")
TEXT = "こんにちは。音声合成の動作確認です。"

def wait_sttts_release(tmp):
    # The upstream stop reply precedes model destruction. Wait for the actual
    # process locks so the next model cannot race the previous GPU owner.
    deadline = time.monotonic() + 30
    for name in ("sttts-gpu.lock", "sttts-say-server.lock"):
        with (tmp / name).open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("sttts-say did not release its process/GPU locks within 30 seconds")
                    time.sleep(0.05)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=ENGINES)
    parser.add_argument("--model", choices=("rf", "mf"), default="rf")
    parser.add_argument("--backend", choices=("cuda", "vulkan", "cpu"))
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/voices/ref1.wav")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    engine, model = args.engine, args.model
    backend = args.backend or ("vulkan" if engine in ("sttts-gui", "Irodori-TTS-burn") else "cuda")
    if engine == "audio.cpp" and model == "mf":
        parser.error("audio.cpp at the pinned commit supports RF only")
    if engine == "irodori-onnx" and model != "mf":
        parser.error("irodori-onnx supports MeanFlow only")
    if engine == "Irodori-TTS-Optimiz" and model == "mf":
        parser.error("The pinned Optimiz commit has no MeanFlow implementation (flow_parameterization is unsupported)")
    supported = {"Irodori-TTS": ("cuda", "cpu"), "Irodori-TTS-Optimiz": ("cuda", "cpu"),
                 "audio.cpp": ("cuda", "vulkan"), "speech.cpp": ("cuda", "vulkan"),
                 "sttts-gui": ("vulkan",), "Irodori-TTS-burn": ("vulkan",),
                 "irodori-onnx": ("cuda",)}
    if backend not in supported[engine]:
        parser.error(f"{engine}: prepared backends are {supported[engine]}")
    if engine == "irodori-onnx" and args.no_reference:
        parser.error("The upstream synth example requires a registered reference voice")
    out = (args.output or ROOT / f"artifacts/smoke/{engine}-{model}-{backend}.wav").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = None if args.no_reference else args.reference.resolve()
    if ref is not None and not ref.is_file():
        parser.error(f"Reference WAV is missing: {ref}. Run convert-models.sh or use --no-reference.")
    env = dict(os.environ, HF_HUB_OFFLINE="1", WGPU_BACKEND="vulkan", RUST_LOG="info")
    tmp = ROOT / ".cache/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    if engine in ("Irodori-TTS", "Irodori-TTS-Optimiz"):
        env["PYTHONPATH"] = str(ROOT / "engines" / engine)
        cmd = [str(ROOT / f".venvs/{engine}/bin/python"), str(ROOT / f"engines/{engine}/infer.py"),
               "--checkpoint", str(ROOT / f"models/{model}/model.safetensors"),
               "--codec-repo", str(ROOT / "models/codec/weights.pth"),
               "--model-device", backend, "--model-precision", "bf16" if backend == "cuda" else "fp32",
               "--codec-device", backend, "--codec-precision", "fp32",
               "--text", args.text, "--seed", str(args.seed), "--output-wav", str(out)]
        cmd += ["--ref-wav", str(ref)] if ref else ["--no-ref"]
    elif engine == "audio.cpp":
        cmd = [str(ROOT / f"artifacts/build/audio.cpp/{backend}/bin/audiocpp_cli"),
               "--task", "clon" if ref else "tts", "--family", "irodori_tts",
               "--model", str(ROOT / "models/audio-rf-runtime/irodori-tts-v4-small-f16.gguf"),
               "--backend", backend, "--language", "ja", "--text", args.text,
               "--request-option", f"seed={args.seed}", "--out", str(out)]
        cmd += ["--voice-ref", str(ref)] if ref else ["--request-option", "no_ref=true"]
    elif engine == "speech.cpp":
        name = "Irodori-TTS-859M-v4.1-F16.gguf" if model == "rf" else "Irodori-TTS-866M-MF-v4.1-F16.gguf"
        cmd = [str(ROOT / f"artifacts/build/speech.cpp/{backend}/speech"), "tts",
               str(ROOT / f"models/speech-{model}/{name}"), "--device", "gpu",
               "--seed", str(args.seed), "-o", str(out)]
        cmd += ["--add-voice", f"reference={ref}", "--voice", "reference"] if ref else ["--voice", "none"]
        cmd += [args.text]
    elif engine == "Irodori-TTS-burn":
        cmd = [str(ROOT / "artifacts/build/Irodori-TTS-burn/release/irodori"),
               "--checkpoint", str(ROOT / f"models/{model}/model.safetensors"),
               "--codec", str(ROOT / "artifacts/codec/dacvae.safetensors"),
               "--precision", "fp32", "--wgpu-device", "discrete:0",
               "--cache-dir", str(ROOT / ".cache/burn")]
        if ref:
            cmd += ["--add-voice", f"reference={ref}"]
        cmd += ["tts", "--text", args.text, "--seed", str(args.seed), "--output", str(out)]
        if ref:
            cmd += ["--voice", "reference"]
    else:
        cmd = []
    lock_path = ROOT / "artifacts/gpu.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if engine == "sttts-gui":
            binary = str(ROOT / "artifacts/build/sttts-gui/release/sttts-say")
            try:
                subprocess.run([binary, "model", "use", "v4.1-small" + ("-mf" if model == "mf" else "")], env=env, check=True)
                cmd = [binary, "speak", "--text", args.text, "--seed", str(args.seed), "--out", str(out)]
                if ref:
                    voices = Path(env["STTTS_ROOT"]) / "data/voices"
                    voices.mkdir(parents=True, exist_ok=True)
                    link = voices / "comparison-reference.wav"
                    if link.exists() and link.resolve() != ref:
                        raise RuntimeError(f"Unexpected reference voice: {link}")
                    if not link.exists():
                        link.symlink_to(ref)
                    cmd += ["--voice", "comparison-reference"]
                subprocess.run(cmd, env=env, check=True)
            finally:
                subprocess.run([binary, "stop"], env=env, check=True)
                wait_sttts_release(tmp)
        elif engine == "irodori-onnx":
            site = ROOT / ".venvs/onnx/lib/python3.10/site-packages"
            libs = [str(p) for p in sorted((site / "nvidia").glob("*/lib"))]
            libs += [str(site / "onnxruntime/capi"), env.get("LD_LIBRARY_PATH", "")]
            env.update(LD_LIBRARY_PATH=":".join(libs), IRODORI_ORT_DLL=str(site / "onnxruntime/capi/libonnxruntime.so.1.23.2"),
                       IRODORI_MODEL_DIR=str(ROOT / "artifacts/onnx/export/ref"),
                       IRODORI_TOKENIZER_DIR=str(ROOT / "models/mf/tokenizer"), IRODORI_CUDA="0", IRODORI_FP16="1",
                       NVIDIA_TF32_OVERRIDE="0")
            binary = str(ROOT / "artifacts/build/irodori-onnx/release/examples/synth")
            voice = ROOT / "artifacts/voices/onnx-reference.irvc"
            subprocess.run([binary, "register", str(voice), str(ref)], env=env, check=True)
            subprocess.run([binary, "say", str(voice), args.text, str(out), "--seed", str(args.seed)], env=env, check=True)
        else:
            subprocess.run(cmd, env=env, check=True)
    print(f"Output: {out}", flush=True)

if __name__ == "__main__":
    main()
