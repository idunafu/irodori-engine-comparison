use std::{io::{self,BufRead,Write},path::PathBuf,time::Instant};
use serde_json::{Value,json};
use irodori::pipeline::{Tts,TtsPaths,SamplingRequest};
fn emit(v:Value){println!("BENCH {v}");io::stdout().flush().unwrap();}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let variant=&a[2];
    let t=Instant::now();let device=irodori::try_gpu_device()?;
    irodori::bench_profile::device(device.clone());
    let tts=Tts::load(&TtsPaths {model_weights:root.join(format!("models/{variant}/model.safetensors")),
        tokenizer_dir:root.join(format!("models/{variant}/tokenizer")),codec_weights:root.join("models/codec/weights.pth"),watermark_dir:None},&device)?;
    device.sync().map_err(|e|anyhow::anyhow!("{e:?}"))?;
    emit(json!({"type":"ready","load_ms":t.elapsed().as_secs_f64()*1000.,"precision":"fp32","device":"Vulkan discrete:0"}));
    let mut cached=None;
    for line in io::stdin().lock().lines(){let r:Value=serde_json::from_str(&line?)?;if r["type"]=="stop"{break;}
        let profile=r["profile"].as_bool().unwrap_or(false);irodori::bench_profile::begin(profile);
        let mut req=SamplingRequest {text:r["text"].as_str().unwrap().into(),
            caption:Some("自然な日本語で、落ち着いた口調で話す。".into()),
            ref_wav:Some(root.join("artifacts/benchmark/inputs/reference.wav")),no_ref:false,
            num_steps:Some(if variant=="rf"{40}else{4}),seed:Some(42),seconds:r["seconds"].as_f64(),
            watermark:false,trim_tail:false,cfg_scale_text:3.,cfg_scale_caption:3.,cfg_scale_speaker:5.,
            cfg_min_t:0.5,cfg_max_t:1.,..SamplingRequest::default()};
        let t=Instant::now();let mut hit=false;
        if r["cache"]=="native_cache" {
            if cached.is_some(){hit=true;}else{cached=Some(tts.encode_reference(req.ref_wav.as_ref().unwrap(),&req,&mut Vec::new())?);}
            req.ref_latent=cached.clone();
        }
        let out=tts.synthesize(&req)?;
        let ms=t.elapsed().as_secs_f64()*1000.;
        if let Some(path)=r["output"].as_str(){let mut f=std::fs::File::create(path)?;for v in &out.audio{f.write_all(&v.to_le_bytes())?;}}
        let rms=(out.audio.iter().map(|v|v*v).sum::<f32>()/out.audio.len() as f32).sqrt();
        emit(json!({"type":"result","id":r["id"],"ms":ms,"samples":out.audio.len(),"sample_rate":out.sample_rate,
            "frames":out.audio.len().div_ceil(1920),"events":irodori::bench_profile::take(),"stages":out.timings,
            "finite":out.audio.iter().all(|v|v.is_finite()),"rms":rms,"reference_cache_hit":hit}));
    }Ok(())
}
