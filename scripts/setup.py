#!/usr/bin/env python3
"""Rebuild the pinned workspace and retain the result of every setup stage."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    results = {}
    status_path = ROOT / "artifacts/setup-results.json"
    status_path.parent.mkdir(exist_ok=True)
    def stage(name, command):
        print(f"Setup: {name}", flush=True)
        log = ROOT / "logs" / f"setup-{name}.log"
        with log.open("w") as stream:
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        results[name] = {"returncode": done.returncode, "log": str(log.relative_to(ROOT)),
                         "completed_at": datetime.now(timezone.utc).isoformat()}
        status_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"{'PASS' if done.returncode == 0 else 'FAIL'} {name}: {log}", flush=True)
        return done.returncode == 0

    if not stage("submodules", ["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
                               "submodule", "update", "--init", "--recursive"]):
        raise SystemExit(1)
    pins = json.loads((ROOT / "engines.lock.json").read_text())
    for engine, pin in pins.items():
        actual = subprocess.check_output(["git", "-C", str(ROOT / "engines" / engine), "rev-parse", "HEAD"], text=True).strip()
        if actual != pin["commit"]:
            raise SystemExit(f"Unexpected commit for {engine}: {actual}")
        clean = subprocess.run(["git", "-C", str(ROOT / "engines" / engine), "diff", "--quiet", "HEAD", "--"])
        if clean.returncode:
            raise SystemExit(f"Tracked source changes in {engine}; preserve them and restore the pinned source before setup")
    stage("vulkan", ["bash", "scripts/setup-vulkan.sh"])
    python_ok = stage("python", ["bash", "scripts/setup-python.sh"])
    onnx_ok = stage("onnx-python", ["bash", "scripts/setup-onnx-python.sh"])
    models_ok = stage("models", ["python3", "scripts/setup-models.py"])
    for engine in ("audio.cpp", "speech.cpp", "sttts-gui", "Irodori-TTS-burn", "irodori-onnx"):
        stage(f"build-{engine}", ["bash", "scripts/setup-native.sh", engine])
    if python_ok and models_ok:
        reference_ok = stage("reference", ["bash", "scripts/run.sh", "Irodori-TTS", "--model", "rf",
                                            "--no-reference", "--output", "artifacts/smoke/original-rf.wav"])
        if reference_ok and onnx_ok:
            stage("conversion", ["bash", "scripts/convert-models.sh"])
        stage("audio-check", ["bash", "scripts/check.sh"])
    stage("provenance", ["python3", "scripts/provenance.py"])
    if any(row["returncode"] for row in results.values()):
        raise SystemExit("Some stages failed; see artifacts/setup-results.json and logs. No system packages were installed.")

if __name__ == "__main__":
    main()
