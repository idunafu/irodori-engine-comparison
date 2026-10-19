"""Build quality workers from isolated sources, reusing the pinned native dependencies."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from common import ASSETS, BASE, HERE, ROOT, atomic_json, run, sha256


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'Expected one patch anchor: {path}: {old[:80]}')
    path.write_text(text.replace(old, new))


def main():
    dest = BASE / 'build'
    dest.mkdir(parents=True, exist_ok=True)
    heard = ASSETS / 'speech-bench/src/measure/heard.ts'
    if heard.exists():
        heard.with_name('quality-heard.ts').write_text(heard.read_text()+'\nexport { heardUnits };\n')
    build = ROOT / 'artifacts/benchmark/build/speech.cpp/vulkan'
    source = ROOT / 'engines/speech.cpp'
    link = shlex.split((build / 'CMakeFiles/comparison_bench.dir/link.txt').read_text())
    obj = next(x for x in link if x.endswith('.cpp.o'))
    link[link.index(obj):link.index(obj)+1] = [str(HERE / 'speech_worker.cpp'),
        str(source / 'tools/common/json-reader.cpp'), '-std=c++17',
        '-I'+str(source / 'src/families'), '-I'+str(source / 'src/common'),
        '-I'+str(source / 'tools/common'), '-I'+str(source / 'ggml/include')]
    link[link.index('-o')+1] = str(dest / 'speech-quality')
    print('Compile speech.cpp quality worker', flush=True)
    subprocess.run(link, cwd=build, check=True)
    # Same pinned source, separate ASR task; never route TTS through worker sentence splitting.
    run(['cmake', '--build', build, '--target', 'speech-cli', '--parallel', '8'])
    overlay = BASE / 'source/irodori-onnx'
    if overlay.exists():
        shutil.rmtree(overlay)
    shutil.copytree(ROOT / 'engines/irodori-onnx', overlay,
                    ignore=shutil.ignore_patterns('.git', '.venv', 'target', '__pycache__'))
    crate = overlay / 'crates/irodori-tts'
    engine = crate / 'src/engine.rs'
    replace(engine, 'pub steps: usize,', 'pub steps: usize,\n    pub seconds: Option<f64>,')
    replace(engine, 'steps: 4,', 'steps: 4,\n            seconds: None,')
    replace(engine, 'let frames = self.predict_frames(',
            'let frames = if let Some(seconds) = opts.seconds { (seconds * 25.0).round() as usize } else { self.predict_frames(')
    replace(engine, 'has_caption, opts)?;', 'has_caption, opts)? };')
    replace(engine, 'text::duration_features(normalized, tokens, MAX_TEXT_LEN, true)',
            'text::duration_features(normalized, tokens, MAX_TEXT_LEN, voice.mask.iter().any(|&v| v))')
    replace(engine, 'ndarray::arr1(&[true])', 'ndarray::arr1(&[voice.mask.iter().any(|&v| v)])')
    replace(crate / 'src/session.rs', 'let mut builder = Session::builder()?;',
            'let mut builder = Session::builder()?.with_intra_threads(8)?.with_inter_threads(1)?\n'
            '            .with_config_entry("session.intra_op.allow_spinning", "0")?;')
    shutil.copyfile(HERE / 'onnx_worker.rs', crate / 'examples/quality_worker.rs')
    run(['cargo', 'build', '--manifest-path', overlay / 'Cargo.toml', '--release', '--locked',
         '--target-dir', BASE / 'rust-target', '-p', 'irodori-tts', '--example', 'quality_worker'])
    shutil.copy2(BASE / 'rust-target/release/examples/quality_worker', dest / 'onnx-quality')
    atomic_json(dest / 'manifest.json', dict(worker_sources={p.name: sha256(p) for p in HERE.glob('*worker*')},
        speech_binary=sha256(dest / 'speech-quality'), onnx_binary=sha256(dest / 'onnx-quality'),
        asr_binary=sha256(build / 'speech'), onnx_engine_patch=sha256(engine),
        onnx_behavior='has_speaker and duration features follow the voice mask; fixed seconds only when requested'))


if __name__ == '__main__':
    main()
