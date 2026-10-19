use std::{io::{self,BufRead,Write},path::PathBuf,time::Instant};
use serde_json::{Value,json};
use irodori_tts::{Clip,Engine,EngineConfig,SynthOptions,VoiceRegistrar};
use ndarray::Array3;
fn emit(v:Value){println!("QUALITY {v}");io::stdout().flush().unwrap();}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);
 let config=EngineConfig{model_dir:root.join("artifacts/onnx/export/ref"),tokenizer_dir:root.join("models/mf/tokenizer"),
     ort_dylib:std::env::var("IRODORI_ORT_DLL").ok().map(PathBuf::from),fp16:a[2]=="fp16",cuda_device:Some(0)};
 let mut engine=Engine::new(&config)?;let mut registrar=VoiceRegistrar::new(&config)?;
 emit(json!({"type":"ready","device":"CUDA:0","precision":a[2]}));
 let mut key=String::new();let mut voice=None;
 for line in io::stdin().lock().lines(){let r:Value=serde_json::from_str(&line?)?;if r["type"]=="stop"{break;}
  let result=(||->Result<Value,Box<dyn std::error::Error>>{
   let t=Instant::now();let refs=r["references"].as_array().unwrap();let k=serde_json::to_string(refs)?;
   if voice.is_none() || k!=key {
    voice=Some(if refs.is_empty(){registrar.register_no_reference()?}else{
      let mut recordings=Vec::new();
      for p in refs {let mut wav=hound::WavReader::open(p.as_str().unwrap())?;let spec=wav.spec();
       if spec.channels!=1{return Err("Reference must be mono".into());}
       let samples:Vec<f32>=match spec.sample_format {
        hound::SampleFormat::Float=>wav.samples::<f32>().collect::<Result<_,_>>()?,
        _=>wav.samples::<i32>().map(|x|x.map(|s|s as f32/(1u64<<(spec.bits_per_sample-1)) as f32)).collect::<Result<_,_>>()?
       };recordings.push((samples,spec.sample_rate));}
      let clips:Vec<Clip>=recordings.iter().map(|(s,sr)|Clip{samples:s,sample_rate:*sr}).collect();registrar.register(&clips)?
    });key=k;
   }
   let noise=if let Some(path)=r["noise"].as_str(){let bytes=std::fs::read(path)?;
      let v:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
      Some(Array3::from_shape_vec((1,v.len()/32,32),v)?)}else{None};
   let opts=SynthOptions{steps:4,seed:r["seed"].as_u64().unwrap(),trim_tail:false,
       caption:r["caption"].as_str().map(str::to_string),seconds:r["seconds"].as_f64(),noise,..SynthOptions::default()};
   let out=engine.synthesize(r["text"].as_str().unwrap(),voice.as_ref().unwrap(),&opts)?;
   let mut f=std::fs::File::create(r["output"].as_str().unwrap())?;
   for v in &out.samples{f.write_all(&v.to_le_bytes())?;}
   Ok(json!({"type":"result","id":r["id"],"samples":out.samples.len(),"frames":out.frames,
      "sample_rate":out.sample_rate,"ms":t.elapsed().as_secs_f64()*1000.,"raw_float":true}))
  })();match result{Ok(v)=>emit(v),Err(e)=>emit(json!({"type":"error","id":r["id"],"error":e.to_string()}))}
 }Ok(())
}
