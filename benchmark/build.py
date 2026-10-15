"""Create disposable source overlays and reproducible instrumented binaries."""
import argparse
import difflib
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_toolchain import cmake_arguments

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/benchmark/source'

def replace(path, old, new):
    s=path.read_text()
    assert old in s, (path,old[:100])
    path.write_text(s.replace(old,new,1))

def rust_scope(path,name,stage):
    s=path.read_text()
    m=re.search(r'\bfn '+re.escape(name)+r'\s*(?:<[^\n]*>)?\s*\(',s)
    assert m,(path,name)
    p=s.index('{',m.end())+1
    path.write_text(s[:p]+f'\n        let _bench_scope = crate::bench_profile::Scope::new("{stage}");\n'+s[p:])

def run(cmd):
    print(' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)

def build_stamp(engine,d,backend,cmd):
    data=dict(engine=engine,backend=backend,command=list(map(str,cmd)),cpu_threads=8,
        instrumentation_version=4,patch_sha256=hashlib.sha256((ROOT/'benchmark/patches'/f'{engine}.patch').read_bytes()).hexdigest())
    if engine=='irodori-onnx':data['ort_intra_threads']=8
    if engine=='audio.cpp':data['text_chunk_size']=4096
    (d/f'.bench-built-{backend}.json').write_text(json.dumps(data,indent=2)+'\n')

def overlay(engine):
    dest=SOURCE/engine
    if dest.exists():shutil.rmtree(dest)
    shutil.copytree(ROOT/'engines'/engine,dest,ignore=shutil.ignore_patterns('.git','target','__pycache__','.venv'))
    return dest

def patches(engine,dest):
    changes=[]
    for p in dest.rglob('*'):
        if not p.is_file() or p.suffix not in ('.cpp','.h','.hpp','.rs','.toml','.txt'):continue
        original=ROOT/'engines'/engine/p.relative_to(dest)
        old=original.read_text() if original.exists() else ''
        new=p.read_text()
        if old!=new:changes.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+str(p.relative_to(dest)),tofile='b/'+str(p.relative_to(dest))))
    path=ROOT/'benchmark/patches'/f'{engine}.patch';path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(changes))

def rust(engine):
    d=overlay(engine)
    if engine=='sttts-gui':
        crate=d/'crates/irodori'; name='irodori'; worker='sttts_worker.rs'
        funcs={'src/dit.rs':[('forward_prepared','dit_forward'),('encode_speaker','speaker_encode')],
               'src/condition.rs':[('encode_text','text_encode'),('encode_caption','caption_encode')],
               'src/pipeline.rs':[('encode_reference','reference_encode')],
               'src/codec.rs':[('decode_latent_windowed','codec_decode'),('encode_waveform','codec_encode')],
               'src/sampler.rs':[('sample_euler_meanflow_padded','sampling'),('sample_euler_rf_cfg_padded','sampling')]}
    elif engine=='Irodori-TTS-burn':
        crate=d;name='irodori-tts-burn';worker='burn_worker.rs'
        funcs={'src/model/wgsl.rs':[('forward_with_cond_embed_wgsl','dit_forward')],
               'src/model/dit/aux_conditioner.rs':[('encode_caption','caption_encode'),('encode_speaker','speaker_encode')],
               'src/codec/model.rs':[('encode_wgsl','codec_encode')],
               'src/codec/streaming.rs':[('decode_stream','codec_decode')],
               'src/inference.rs':[('sample_conditioned','sampling')],
               'src/synthesis/engine.rs':[('create_voice','reference_encode')]}
        # Voice creation's encoder is lazy-loaded by upstream; its cost remains part of registration.
    else:
        crate=d/'crates/irodori-tts';name='irodori-tts';worker='onnx_worker.rs'
        replace(crate/'src/session.rs','let mut builder = Session::builder()?;',
            'let mut builder = Session::builder()?.with_intra_threads(8)?.with_inter_threads(1)?\n            .with_config_entry("session.intra_op.allow_spinning", "0")?;')
        funcs={'src/engine.rs':[('encode_text','text_encode'),('encode_caption','caption_encode'),
               ('predict_frames','duration_predict'),('run_dit_loop','sampling'),('decode','codec_decode')],
               'src/registrar.rs':[('encode_clip','codec_encode')]}
        replace(crate/'src/engine.rs','pub steps: usize,','pub steps: usize,\n    pub seconds: Option<f64>,')
        replace(crate/'src/engine.rs','steps: 4,','steps: 4,\n            seconds: None,')
        replace(crate/'src/engine.rs','let frames = self.predict_frames(',
                'let frames = if let Some(seconds) = opts.seconds { (seconds * 25.0).round() as usize } else { self.predict_frames(')
        replace(crate/'src/engine.rs','has_caption, opts)?;', 'has_caption, opts)? };')
        replace(crate/'src/engine.rs','let out = self.dit.run_binding_with_options(&dit_binding, &self.run_options)?;',
                'let _dit_timer = crate::bench_profile::Scope::new("dit_forward");\n            let out = self.dit.run_binding_with_options(&dit_binding, &self.run_options)?;')
        replace(crate/'src/registrar.rs','let out = self.speaker_encoder.run_with_options(',
                'let _speaker_timer = crate::bench_profile::Scope::new("speaker_encode");\n        let out = self.speaker_encoder.run_with_options(')
    helper=(ROOT/'benchmark/rust_profile.rs').read_text()
    if engine=='irodori-onnx':
        helper=re.sub(r'    static DEVICE:.*\n','',helper)
        helper=re.sub(r'pub fn device[^\n]*\n','',helper)
        helper=re.sub(r'fn sync\(\).*\n','fn sync() {} // ORT outputs are read back synchronously by the pinned engine.\n',helper)
    (crate/'src/bench_profile.rs').write_text(helper)
    with (crate/'src/lib.rs').open('a') as f:f.write('\npub mod bench_profile;\n')
    for path,items in funcs.items():
        for method,stage in items:rust_scope(crate/path,method,stage)
    if engine!='irodori-onnx':
        path=crate/('src/dit.rs' if engine=='sttts-gui' else 'src/model/wgsl.rs')
        replace(path,'let _bench_scope = crate::bench_profile::Scope::new("dit_forward");',
            'let mut _bench_scope = crate::bench_profile::Scope::new("dit_forward");\n        _bench_scope.shape(x_t.dims().to_vec());')
    if engine=='sttts-gui':
        p=crate/'src/sampler.rs'
        replace(p,'    for i in 0..steps {','    let _loop_scope=crate::bench_profile::Scope::new("sampling_loop");\n    for i in 0..steps {')
        replace(p,'    for i in 0..opts.steps {','    let _loop_scope=crate::bench_profile::Scope::new("sampling_loop");\n    for i in 0..opts.steps {')
    if engine=='Irodori-TTS-burn':
        p=crate/'src/rf/euler_sampler.rs'
        replace(p,'        for (step, interval) in schedule.windows(2).enumerate() {','        let _loop_scope=crate::bench_profile::Scope::new("sampling_loop");\n        for (step, interval) in schedule.windows(2).enumerate() {')
        replace(p,'    for i in 0..params.num_steps {','    let _loop_scope=crate::bench_profile::Scope::new("sampling_loop");\n    for i in 0..params.num_steps {')
        p=crate/'src/model/dit/aux_conditioner.rs'
        replace(p,'                let text_state = frontend.text_norm.forward(\n                    frontend\n                        .shared\n                        .encode_text(text_input_ids, text_mask.clone()),\n                );',
                '                let text_state = {let _p=crate::bench_profile::Scope::new("text_encode"); frontend.text_norm.forward(\n                    frontend.shared.encode_text(text_input_ids, text_mask.clone()))};')
    if engine=='irodori-onnx':
        replace(crate/'src/engine.rs','        for step in 0..opts.steps {','        let _loop_scope=crate::bench_profile::Scope::new("sampling_loop");\n        for step in 0..opts.steps {')
        replace(crate/'src/engine.rs','let _dit_timer = crate::bench_profile::Scope::new("dit_forward");',
                'let mut _dit_timer = crate::bench_profile::Scope::new("dit_forward");\n            _dit_timer.shape(vec![1,frames,LATENT_DIM]);')
        replace(crate/'src/engine.rs','let (_, velocity) = out[0].try_extract_tensor::<f32>()?;',
                'let (_, velocity) = out[0].try_extract_tensor::<f32>()?;\n            drop(_dit_timer);')
    examples=crate/'examples';examples.mkdir(exist_ok=True)
    cpu_components(engine,crate)
    shutil.copyfile(ROOT/'benchmark'/worker,examples/'comparison_bench.rs')
    if engine=='sttts-gui':
        with (crate/'Cargo.toml').open('a') as f:f.write('\n[[example]]\nname = "comparison_bench"\nrequired-features = ["gpu"]\n')
    patches(engine,d)
    cmd=['cargo','build','--manifest-path',d/'Cargo.toml','--release','--locked',
         '--target-dir',ROOT/f'artifacts/build/{engine}','-p',name,'--example','comparison_bench']
    if engine=='sttts-gui':cmd+=['--features','irodori/gpu']
    if engine=='Irodori-TTS-burn':cmd+=['--features','cli,synthesis']
    run(cmd)
    build_stamp(engine,d,'rust',cmd)

def cpp(engine,backend):
    d=SOURCE/engine
    if not d.exists():
        d=overlay(engine)
        shutil.copyfile(ROOT/'benchmark/profile.hpp',d/'profile.hpp')
        if engine=='speech.cpp':
            shutil.copyfile(ROOT/'benchmark/speech_worker.cpp',d/'comparison_bench.cpp')
            for name in ('synthesizer.cpp','sampler.cpp','reference.cpp'):
                p=d/'src/families/irodori-tts'/name
                p.write_text('#include "profile.hpp"\n'+p.read_text())
            p=d/'src/families/irodori-tts/sampler.cpp'
            replace(p,'    Graph graph;','    bench::Scope forward_timer("dit_forward"); forward_timer.batch=(int)b.size(); forward_timer.frames=frames;\n    Graph graph;')
            replace(p,'    const std::vector<float> times = schedule(steps, g.sway);','    bench::Scope sample_timer("sampling");\n    const std::vector<float> times = schedule(steps, g.sway);')
            p=d/'src/families/irodori-tts/synthesizer.cpp'
            replace(p,'    Voice v;\n    v.frames', '    bench::Scope speaker_timer("speaker_encode");\n    Voice v;\n    v.frames')
            probe='''        if(bench::active) {
            {bench::Scope p("text_encode");Graph probe;auto st=text_.build(probe,ids);probe.output(st);probe.compute(backend_,allocr_);Graph::read(st);}
            if(!caption_ids.empty()){bench::Scope p("caption_encode");Graph probe;auto st=text_.build(probe,caption_ids,Condition::Caption);probe.output(st);probe.compute(backend_,allocr_);Graph::read(st);}
        }
'''
            replace(p,'        Graph g;\n        ggml_tensor * state = text_.build(g, ids);',probe+'        Graph g;\n        ggml_tensor * state = text_.build(g, ids);')
            p=d/'src/families/irodori-tts/reference.cpp'
            replace(p,'    const Wav wav = read_wav(wav_path);','    bench::Scope reference_timer("reference_encode");\n    const Wav wav = read_wav(wav_path);')
            p=d/'src/families/irodori-tts/codec.cpp'
            p.write_text('#include "profile.hpp"\n'+p.read_text())
            s=p.read_text();m=re.search(r'Codec::encode\([^\n]*\)\s*\{',s)
            assert m
            s=s[:m.end()]+'\n    bench::Scope p("codec_encode");'+s[m.end():];p.write_text(s)
            with (d/'CMakeLists.txt').open('a') as f:f.write('\ninclude_directories(${CMAKE_CURRENT_SOURCE_DIR})\nadd_executable(comparison_bench comparison_bench.cpp)\ntarget_link_libraries(comparison_bench PRIVATE irodoritts)\n')
        else:
            p=d/'tests/irodori_tts/irodori_tts_warm_bench.cpp'
            s=p.read_text();s=s[:s.index('int main(int argc')]
            s='#include "profile.hpp"\n#include "engine/models/irodori_tts/session.h"\n#include <fstream>\n#include <iomanip>\n#include <cmath>\n'+s+(ROOT/'benchmark/audio_main.cpp').read_text()
            s=s.replace('throw std::runtime_error("Irodori-TTS warmbench is CUDA-only");','if (value == "vulkan") return engine::core::BackendType::Vulkan;\n    throw std::runtime_error("Unknown benchmark backend");')
            (d/'comparison_bench.cpp').write_text(s)
            replace(d/'include/engine/models/irodori_tts/session.h','public:\n', 'public:\n  void bench_clear_reference_cache() { reference_speaker_cache_.clear(); }\n')
            for name in ('codec.cpp','rf_dit.cpp','condition_encoder.cpp'):
                p=d/'src/models/irodori_tts'/name
                p.write_text('#include "profile.hpp"\n'+p.read_text())
            p=d/'src/models/irodori_tts/codec.cpp'
            replace(p,'  return impl_->encode_reference(audio, latent_steps_out);','  bench::Scope codec_timer("codec_encode");\n  return impl_->encode_reference(audio, latent_steps_out);')
            replace(p,'  return impl_->decode(latent, latent_steps, target_samples, on_audio, stream_frames);','  bench::Scope codec_timer("codec_decode");\n  return impl_->decode(latent, latent_steps, target_samples, on_audio, stream_frames);')
            p=d/'src/models/irodori_tts/condition_encoder.cpp'
            replace(p,'  return impl_->encode_speaker_reference(ref_latent, ref_tokens);','  bench::Scope speaker_timer("speaker_encode");\n  return impl_->encode_speaker_reference(ref_latent, ref_tokens);')
            replace(p,'  return impl_->run(token_ids, token_mask, caption, speaker);','  bench::Scope conditions_timer("text_caption_duration_fused");\n  return impl_->run(token_ids, token_mask, caption, speaker);')
            p=d/'src/models/irodori_tts/rf_dit.cpp'
            replace(p,'  return impl_->sample(request, timing);','  bench::Scope sampling_timer("sampling");\n  return impl_->sample(request, timing);')
            replace(p,'          auto velocity = run_step_velocity(','          bench::Scope forward_timer("dit_forward");\n          auto velocity = run_step_velocity(')
            with (d/'CMakeLists.txt').open('a') as f:f.write('\ninclude_directories(${CMAKE_CURRENT_SOURCE_DIR})\nadd_executable(comparison_bench comparison_bench.cpp)\ntarget_link_libraries(comparison_bench PRIVATE engine_runtime ggml)\nif(ENGINE_ENABLE_OPENMP)\ntarget_link_libraries(comparison_bench PRIVATE OpenMP::OpenMP_CXX)\nendif()\n')
        patches(engine,d)
    refresh_cpp(engine,d)
    patches(engine,d)
    build=ROOT/f'artifacts/benchmark/build/{engine}/{backend}'
    cmd=['cmake','-S',d,'-B',build,'-DCMAKE_BUILD_TYPE=Release']
    if engine=='speech.cpp':cmd+=['-DGGML_CUDA='+('ON' if backend=='cuda' else 'OFF'),'-DGGML_VULKAN='+('ON' if backend=='vulkan' else 'OFF'),'-DGGML_METAL=OFF']
    else:cmd+=['-DAUDIOCPP_MODEL_SET=custom','-DAUDIOCPP_MODELS=irodori_tts','-DENGINE_ENABLE_CUDA='+('ON' if backend=='cuda' else 'OFF'),'-DENGINE_ENABLE_VULKAN='+('ON' if backend=='vulkan' else 'OFF'),'-DENGINE_ENABLE_METAL=OFF','-DENGINE_ENABLE_HIP=OFF']
    cmd+=cmake_arguments(backend)
    run(cmd);run(['cmake','--build',build,'--target','comparison_bench','--parallel','8'])
    build_stamp(engine,d,backend,cmd)

def refresh_cpp(engine,d):
    shutil.copyfile(ROOT/'benchmark/profile.hpp',d/'profile.hpp')
    if engine=='speech.cpp':
        p=d/'src/families/irodori-tts/codec.cpp';s=p.read_text()
        if 'bench::Scope p("codec_encode")' not in s:
            s='#include "profile.hpp"\n'+s
            m=re.search(r'Codec::encode\([^\n]*\)\s*\{',s);assert m
            s=s[:m.end()]+'\n    bench::Scope p("codec_encode");'+s[m.end():];p.write_text(s)
        p=d/'src/families/irodori-tts/sampler.cpp';s=p.read_text()
        if 'loop_timer("sampling_loop")' not in s:
            s=s.replace('    for (int i = 0; i < steps; i++) {','    bench::Scope loop_timer("sampling_loop");\n    for (int i = 0; i < steps; i++) {',1);p.write_text(s)
        p=d/'src/families/irodori-tts/synthesizer.cpp';s=p.read_text()
        if 'actual_encoder_timer' not in s:
            s=s.replace('bench::Scope p("text_encode")','bench::Scope p("text_encode_probe")')
            s=s.replace('bench::Scope p("caption_encode")','bench::Scope p("caption_encode_probe")')
            s=s.replace('        Graph g;\n        ggml_tensor * state = text_.build(g, ids);','        bench::Scope actual_encoder_timer("text_caption_duration_fused");\n        Graph g;\n        ggml_tensor * state = text_.build(g, ids);',1)
            p.write_text(s)
        cpu_components(engine,d)
        return
    s=(ROOT/'engines/audio.cpp/tests/irodori_tts/irodori_tts_warm_bench.cpp').read_text()
    s=s[:s.index('int main(int argc')]
    s='#include "profile.hpp"\n#include "engine/models/irodori_tts/session.h"\n#include <fstream>\n#include <iomanip>\n#include <cmath>\n'+s+(ROOT/'benchmark/audio_main.cpp').read_text()
    s=s.replace('throw std::runtime_error("Irodori-TTS warmbench is CUDA-only");','if (value == "vulkan") return engine::core::BackendType::Vulkan;\n    throw std::runtime_error("Unknown benchmark backend");')
    (d/'comparison_bench.cpp').write_text(s)
    p=d/'src/models/irodori_tts/condition_encoder.cpp'
    s='#include "profile.hpp"\n'+(ROOT/'engines/audio.cpp/src/models/irodori_tts/condition_encoder.cpp').read_text()
    s=s.replace('  return impl_->encode_speaker_reference(ref_latent, ref_tokens);','  bench::Scope speaker_timer("speaker_encode");\n  return impl_->encode_speaker_reference(ref_latent, ref_tokens);',1)
    s=s.replace('  return impl_->run(token_ids, token_mask, caption, speaker);','  bench::Scope conditions_timer("condition_call_including_probes");\n  return impl_->run(token_ids, token_mask, caption, speaker);',1)
    if 'bench_component_probe' not in s:
        anchor='      core::write_tensor_i32(positions_, positions(tokens_));'
        probe='''
      if (bench::active) {
        bench::encoder_probe("text_encode_probe",output_text_.tensor,owner_->backend_);
        if(config.use_caption_condition)bench::encoder_probe("caption_encode_probe",output_caption_.tensor,owner_->backend_);
      }
'''
        pos=s.rfind(anchor);assert pos>=0;s=s[:pos]+s[pos:].replace(anchor,anchor+probe,1);p.write_text(s)
        marker='      core::set_backend_threads(owner_->backend_, owner_->threads_);'
        pos=s.rfind(marker);assert pos>=0
        s=s[:pos]+s[pos:].replace(marker,'      bench::Scope actual_encoder_timer("text_caption_duration_fused");\n'+marker,1)
        p.write_text(s)
    p=d/'src/framework/sampling/flow_sampler_runtime.cpp';s=p.read_text()
    if 'bench::Scope loop_timer("sampling_loop")' not in s:
        s='#include "profile.hpp"\n'+s
        s=s.replace('        for (int64_t i = 0; i < static_cast<int64_t>(config_.schedule.size()); ++i) {',
                    '        bench::Scope loop_timer("sampling_loop");\n        for (int64_t i = 0; i < static_cast<int64_t>(config_.schedule.size()); ++i) {',1)
        p.write_text(s)
    p=d/'src/models/irodori_tts/rf_dit.cpp';s=p.read_text()
    if 'forward_timer.frames=shape.latent_steps' not in s:
        s=s.replace('          bench::Scope forward_timer("dit_forward");',
                    '          bench::Scope forward_timer("dit_forward");\n          forward_timer.frames=shape.latent_steps;\n          forward_timer.batch=(int)step_batch(selection.text_cfg_enabled,selection.speaker_cfg_enabled,selection.caption_cfg_enabled);',1)
        p.write_text(s)
    cpu_components(engine,d)

def cpu_components(engine,d):
    """Instrument CPU preparation separately without changing calculations."""
    if engine=='speech.cpp':
        p=d/'src/families/irodori-tts/synthesizer.cpp';s=p.read_text()
        if 'text_tokenize' not in s:
            s=s.replace('const std::vector<int32_t> ids = tokenizer_.encode(text);',
                'const std::vector<int32_t> ids = [&]{bench::Scope t("text_tokenize");return tokenizer_.encode(text);}();')
            s=s.replace('caption_ids = tokenizer_.encode(caption);',
                'caption_ids = [&]{bench::Scope t("caption_tokenize");return tokenizer_.encode(caption);}();')
            p.write_text(s)
        p=d/'src/families/irodori-tts/reference.cpp';s=p.read_text()
        if 'reference_preprocess' not in s:
            s=s.replace('    const Wav wav = read_wav(wav_path);','    bench::Scope prep("reference_preprocess");\n    const Wav wav = read_wav(wav_path);')
            s=s.replace('    return {codec.encode(audio),','    prep.finish();\n    return {codec.encode(audio),');p.write_text(s)
    elif engine=='audio.cpp':
        p=d/'src/models/irodori_tts/session.cpp';s=p.read_text()
        if 'text_tokenize' not in s:
            s='#include "profile.hpp"\n'+s
            s=s.replace('    const auto caption_start = Clock::now();','    bench::Scope caption_tokenizer("caption_tokenize");\n    const auto caption_start = Clock::now();')
            s=s.replace('    const auto tokenized = tokenizer_.encode_padded(\n        irodori_request.text, assets_->config.max_text_len);',
                '    const auto tokenized = [&]{bench::Scope t("text_tokenize");return tokenizer_.encode_padded(\n        irodori_request.text, assets_->config.max_text_len);}();')
            p.write_text(s)
        p=d/'src/models/irodori_tts/codec.cpp';s=p.read_text()
        if 'reference_preprocess' not in s:
            s=s.replace('    auto mono = engine::audio::mixdown_interleaved_to_mono_average(',
                '    bench::Scope prep("reference_preprocess_in_codec");\n    auto mono = engine::audio::mixdown_interleaved_to_mono_average(',1)
            s=s.replace('    const int64_t padded_samples = static_cast<int64_t>(mono.size());',
                '    prep.finish();\n    const int64_t padded_samples = static_cast<int64_t>(mono.size());',1);p.write_text(s)
        p=d/'comparison_bench.cpp';s=p.read_text()
        s=s.replace('    const auto wav = engine::audio::read_wav_f32(path);','    bench::Scope file("reference_file_read");\n    const auto wav = engine::audio::read_wav_f32(path);');p.write_text(s)
    elif engine=='sttts-gui':
        p=d/'src/pipeline.rs'
        replace(p,'let (ids, mask) = self.tokenizer.batch_encode(std::slice::from_ref(&text), self.cfg.max_text_len, self.cfg.text_add_bos)?;',
            'let (ids, mask) = {let _t=crate::bench_profile::Scope::cpu("text_tokenize"); self.tokenizer.batch_encode(std::slice::from_ref(&text), self.cfg.max_text_len, self.cfg.text_add_bos)?};')
        replace(p,'let (channels, sr) = load_audio(path)?;',
            'let (channels, sr) = {let _t=crate::bench_profile::Scope::cpu("reference_file_read");load_audio(path)?};')
        rust_scope(d/'src/condition.rs','caption_inputs','caption_tokenize')
        p=d/'src/codec.rs'
        replace(p,'let [b, c, n] = wav.dims();','let _prep=crate::bench_profile::Scope::new("reference_preprocess_in_codec");\n        let [b, c, n] = wav.dims();')
        replace(p,'        let t = Tensor::<3>::from_data(TensorData::new(all, [b, 1, len]), &self.device);',
            '        drop(_prep);\n        let t = Tensor::<3>::from_data(TensorData::new(all, [b, 1, len]), &self.device);')
    elif engine=='Irodori-TTS-burn':
        p=d/'src/synthesis/text.rs'
        replace(p,'        let mut ids = self.ids(text)?;',
            '        let _tok=crate::bench_profile::Scope::cpu(if caption {"caption_tokenize"} else {"text_tokenize"});\n        let mut ids = self.ids(text)?;')
        replace(p,'        Ok(Tokens {','        drop(_tok);\n        Ok(Tokens {')
        p=d/'src/synthesis/engine.rs'
        replace(p,'        let mut prepared = Vec::with_capacity(recordings.len());',
            '        let _prep=crate::bench_profile::Scope::cpu("reference_preprocess");\n        let mut prepared = Vec::with_capacity(recordings.len());')
        replace(p,'        let mut encoder =\n', '        drop(_prep);\n        let _codec_load=crate::bench_profile::Scope::new("reference_codec_load");\n        let mut encoder =\n')
        replace(p,'        encoder.prepare_encoder_for_wgsl();',
            '        encoder.prepare_encoder_for_wgsl();\n        drop(_codec_load);')
    else:
        p=d/'src/engine.rs'
        replace(p,'let ids = self.tokenizer.encode(normalized, MAX_TEXT_LEN)?;',
            'let ids = {let _t=crate::bench_profile::Scope::cpu("text_tokenize");self.tokenizer.encode(normalized, MAX_TEXT_LEN)?};')
        replace(p,'let cap_ids = self.tokenizer.encode(caption, MAX_CAPTION_LEN)?;',
            'let cap_ids = {let _t=crate::bench_profile::Scope::cpu("caption_tokenize");self.tokenizer.encode(caption, MAX_CAPTION_LEN)?};')
        p=d/'src/registrar.rs'
        replace(p,'let mut samples = audio::resample_to_codec_rate(clip.samples, clip.sample_rate)?;',
            'let _prep=crate::bench_profile::Scope::cpu("reference_preprocess_in_codec");\n        let mut samples = audio::resample_to_codec_rate(clip.samples, clip.sample_rate)?;')
        replace(p,'        let out = self.codec_encoder.run_with_options(',
            '        drop(_prep);\n        let out = self.codec_encoder.run_with_options(')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('engine');parser.add_argument('--backend',default='cuda');a=parser.parse_args()
    SOURCE.mkdir(parents=True,exist_ok=True)
    (rust(a.engine) if a.engine not in ('speech.cpp','audio.cpp') else cpp(a.engine,a.backend))
