#!/usr/bin/env python3
"""Download only the pinned model files and expose workspace-local aliases."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "rf": ("Aratako/Irodori-TTS-v4.1-Small", "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b", ["model.safetensors", "tokenizer/*", "README.md"]),
    "mf": ("Aratako/Irodori-TTS-v4.1-Small-MF", "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e", ["model.safetensors", "tokenizer/*", "README.md"]),
    "codec": ("Aratako/Semantic-DACVAE-Japanese-32dim", "47376ee24834d7a05a48ebabfe3cde29b3c5e214", ["weights.pth", "README.md"]),
    "speech-rf": ("sakasegawa/Irodori-TTS-v4.1-Small-GGUF", "66fe2cde1973fd44e7202fbe36df722103958188", ["Irodori-TTS-859M-v4.1-F16.gguf", "Irodori-TTS-859M-v4.1-F16.gguf.json", "README.md"]),
    "speech-mf": ("sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF", "9db7da1bec3d1069ef8e29732e0497be60c7b047", ["Irodori-TTS-866M-MF-v4.1-F16.gguf", "Irodori-TTS-866M-MF-v4.1-F16.gguf.json", "README.md"]),
    "audio-rf": ("audio-cpp/audio.cpp-gguf", "f861648c0c1f4da9f18e9a0ce5e217b0679bb2f5", ["Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-f16.gguf"]),
    "watermark": ("sony/silentcipher", "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1", ["*"]),
}

def main():
    cache = ROOT / ".cache/huggingface/hub"
    records = {}
    (ROOT / "models").mkdir(exist_ok=True)
    env = dict(os.environ, HF_HOME=str(cache.parent), HF_HUB_CACHE=str(cache))
    aliases = sys.argv[1:] or list(MODELS)
    for alias in aliases:
        repo, revision, files = MODELS[alias]
        include_args = [arg for pattern in files for arg in ("--include", pattern)]
        subprocess.run(["hf", "download", repo, "--revision", revision,
                        "--cache-dir", str(cache), *include_args], env=env, check=True)
        repo_cache = cache / ("models--" + repo.replace("/", "--"))
        snapshot = repo_cache / "snapshots" / revision
        # Rust's HF client resolves refs/main. This workspace pins that ref locally.
        (repo_cache / "refs").mkdir(exist_ok=True)
        (repo_cache / "refs/main").write_text(revision)
        link = ROOT / "models" / alias
        if link.is_symlink():
            if link.resolve() != snapshot.resolve():
                raise RuntimeError(f"Unexpected model alias: {link}")
        elif link.exists():
            raise RuntimeError(f"Model alias is not a symlink: {link}")
        else:
            link.symlink_to(os.path.relpath(snapshot, link.parent), target_is_directory=True)
        if alias == "audio-rf":
            # audio.cpp resolves symlinks before detecting the source format.
            # A hard link preserves the .gguf filename without duplicating weights.
            folder = ROOT / "models/audio-rf-runtime"
            folder.mkdir(exist_ok=True)
            source = snapshot / "Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-f16.gguf"
            target = folder / source.name
            if target.exists():
                if not os.path.samefile(source, target):
                    raise RuntimeError(f"Unexpected materialized GGUF: {target}")
            else:
                os.link(source.resolve(), target)
    records = {alias: {"repository": repo, "revision": revision, "files": files}
               for alias, (repo, revision, files) in MODELS.items()}
    (ROOT / "models.lock.json").write_text(json.dumps(records, indent=2) + "\n")

if __name__ == "__main__":
    main()
