#include "irodori-tts/synthesizer.h"
#include "ggml-backend.h"
#include "profile.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
int main(int argc,char ** argv){
 try {
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if(!dev)throw std::runtime_error("No GPU backend");
    auto t=std::chrono::steady_clock::now();
    auto backend=ggml_backend_dev_init(dev,nullptr);
    irodori::Synthesizer synth(argv[1],backend);
    std::cout<<"BENCH {\"type\":\"ready\",\"load_ms\":"<<bench::elapsed(t)<<",\"device\":"<<std::quoted(ggml_backend_dev_description(dev))<<"}"<<std::endl;
    irodori::Voice voice; bool has_voice=false;
    std::string line;
    while(std::getline(std::cin,line)){
      if(line=="stop")break;
      std::vector<std::string> fields;std::stringstream ss(line);std::string f;
      while(std::getline(ss,f,'\t'))fields.push_back(f);
      if(fields.size()<7)throw std::runtime_error("Invalid benchmark request");
      bench::active=fields[1]=="1";bench::events.clear();
      irodori::Request r;r.text=fields[4];r.caption="自然な日本語で、落ち着いた口調で話す。";
      r.seed=42;r.steps=std::stoi(argv[3]);r.length.seconds=std::stod(fields[3]);
      auto tail=synth.default_tail();tail.keep=true;r.tail=tail;
      if(r.steps==40){auto g=synth.default_guidance();g.text=3;g.caption=3;g.speaker=5;
         g.mode=irodori::GuidanceMode::Independent;g.min_t=.5;g.max_t=1.;g.sway=0;r.guidance=g;}
      auto started=std::chrono::steady_clock::now();
      bool hit=fields[2]=="native_cache" && has_voice;
      if(!hit){bench::Scope p("reference_total");voice=synth.load_voice(argv[2]);has_voice=true;}
      std::vector<float> audio;irodori::Stats stats;
      synth.synthesize(r,voice,[&](const float *p,size_t n){audio.insert(audio.end(),p,p+n);return true;},&stats);
      const double ms=bench::elapsed(started);
      double sq=0;bool finite=true;for(float v:audio){sq+=v*v;finite&=std::isfinite(v);}
      if(fields[5]!="-"){std::ofstream out(fields[5],std::ios::binary);out.write((char*)audio.data(),audio.size()*sizeof(float));}
      std::cout<<"BENCH {\"type\":\"result\",\"id\":"<<std::quoted(fields[0])<<",\"ms\":"<<ms
        <<",\"samples\":"<<audio.size()<<",\"sample_rate\":48000,\"frames\":"<<stats.frames
        <<",\"events\":"<<bench::take()<<",\"reference_cache_hit\":"<<(hit?"true":"false")
        <<",\"finite\":"<<(finite?"true":"false")<<",\"rms\":"<<std::sqrt(sq/audio.size())
        <<",\"stages\":{\"text_caption_duration_fused\":"<<stats.text<<",\"sample_rf\":"<<stats.sampling<<",\"decode_latent\":"<<stats.codec<<"}}"<<std::endl;
    }
 }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
