use std::{io::{self,BufRead,Write},path::PathBuf,time::Instant};
use serde_json::{Value,json};
use irodori_tts_burn::{backend_config,WgpuFloatPrecision,synthesis::{SynthesisEngine,SynthesisModelPaths,SynthesisRequest}};
fn emit(v:Value){println!("BENCH {v}");io::stdout().flush().unwrap();}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let variant=&a[2];let t=Instant::now();
    let precision=WgpuFloatPrecision::Fp32;
    let selected="discrete:0".parse::<backend_config::DeviceSelector>()?;
    let device=backend_config::wgpu_device_with_precision(&selected.device(),precision)?;
    irodori_tts_burn::bench_profile::device(device.clone());
    let mut engine=SynthesisEngine::load(SynthesisModelPaths{checkpoint:root.join(format!("models/{variant}/model.safetensors")),
        codec:root.join("artifacts/codec/dacvae.safetensors"),tokenizer:None},device.clone(),precision)?;
    device.sync().map_err(|e|anyhow::anyhow!("{e:?}"))?;
    emit(json!({"type":"ready","load_ms":t.elapsed().as_secs_f64()*1000.,"precision":"fp32","device":"Vulkan discrete:0"}));
    let mut cached=false;
    for line in io::stdin().lock().lines(){let r:Value=serde_json::from_str(&line?)?;if r["type"]=="stop"{break;}
        irodori_tts_burn::bench_profile::begin(r["profile"].as_bool().unwrap_or(false));
        let t=Instant::now();let hit=cached && r["cache"]=="native_cache";
        if !hit {let _p=irodori_tts_burn::bench_profile::Scope::new("reference_total");
            let voice=engine.create_voice(&[root.join("artifacts/benchmark/inputs/reference.wav")],Some(-16.))?;
            engine.add_voice("reference",voice)?;cached=true;}
        let mut req=SynthesisRequest {text:r["text"].as_str().unwrap().into(),instructions:"自然な日本語で、落ち着いた口調で話す。".into(),
            voice:"reference".into(),steps:Some(if variant=="rf"{40}else{4}),seed:Some(42),seconds:r["seconds"].as_f64(),keep_tail:true,
            ..SynthesisRequest::default()};
        if variant=="rf"{req.cfg_scale_text=Some(3.);req.cfg_scale_instructions=Some(3.);req.cfg_scale_speaker=Some(5.);
           req.cfg_guidance_mode=Some(irodori_tts_burn::CfgGuidanceMode::Independent);req.cfg_min_t=Some(0.5);req.cfg_max_t=Some(1.);req.sway_coeff=Some(0.);}
        let (audio,report)=engine.synthesize(&req)?;device.sync().map_err(|e|anyhow::anyhow!("{e:?}"))?;
        let ms=t.elapsed().as_secs_f64()*1000.;
        if let Some(path)=r["output"].as_str(){let mut f=std::fs::File::create(path)?;for v in &audio{f.write_all(&v.to_le_bytes())?;}}
        let rms=(audio.iter().map(|v|v*v).sum::<f32>()/audio.len() as f32).sqrt();
        emit(json!({"type":"result","id":r["id"],"ms":ms,"samples":audio.len(),"sample_rate":report.sample_rate,
            "frames":audio.len().div_ceil(1920),"events":irodori_tts_burn::bench_profile::take(),"finite":audio.iter().all(|v|v.is_finite()),
            "rms":rms,"reference_cache_hit":hit}));
    }Ok(())
}
