use std::{io::{self,BufRead,Write},path::PathBuf,time::Instant};
use serde_json::{Value,json};
use irodori_tts::{Clip,Engine,EngineConfig,SynthOptions,VoiceRegistrar};
fn emit(v:Value){println!("BENCH {v}");io::stdout().flush().unwrap();}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let t=Instant::now();
    let config=EngineConfig{model_dir:root.join("artifacts/onnx/export/ref"),tokenizer_dir:root.join("models/mf/tokenizer"),
        ort_dylib:std::env::var("IRODORI_ORT_DLL").ok().map(PathBuf::from),fp16:a[2]=="fp16",cuda_device:Some(0)};
    let mut engine=Engine::new(&config)?;let mut registrar=VoiceRegistrar::new(&config)?;
    emit(json!({"type":"ready","load_ms":t.elapsed().as_secs_f64()*1000.,"precision":a[2],"device":"CUDA:0"}));
    let mut reader=hound::WavReader::open(root.join("artifacts/benchmark/inputs/reference.wav"))?;
    let samples:Vec<f32>=reader.samples::<i32>().map(|x|x.unwrap() as f32/8388608.).collect();
    let mut voice=None;
    for line in io::stdin().lock().lines(){let r:Value=serde_json::from_str(&line?)?;if r["type"]=="stop"{break;}
        irodori_tts::bench_profile::begin(r["profile"].as_bool().unwrap_or(false));let t=Instant::now();
        let hit=voice.is_some() && r["cache"]=="native_cache";
        if !hit{let _p=irodori_tts::bench_profile::Scope::new("reference_total");voice=Some(registrar.register(&[Clip{samples:&samples,sample_rate:48000}])?);}
        let opts=SynthOptions{steps:4,seed:42,trim_tail:false,caption:Some("自然な日本語で、落ち着いた口調で話す。".into()),
            seconds:r["seconds"].as_f64(),..SynthOptions::default()};
        let out=engine.synthesize(r["text"].as_str().unwrap(),voice.as_ref().unwrap(),&opts)?;
        let ms=t.elapsed().as_secs_f64()*1000.;
        if let Some(path)=r["output"].as_str(){let mut f=std::fs::File::create(path)?;for v in &out.samples{f.write_all(&v.to_le_bytes())?;}}
        let rms=(out.samples.iter().map(|v|v*v).sum::<f32>()/out.samples.len() as f32).sqrt();
        emit(json!({"type":"result","id":r["id"],"ms":ms,"samples":out.samples.len(),"sample_rate":out.sample_rate,
            "frames":out.frames,"events":irodori_tts::bench_profile::take(),"finite":out.samples.iter().all(|v|v.is_finite()),
            "rms":rms,"reference_cache_hit":hit}));
    }Ok(())
}
