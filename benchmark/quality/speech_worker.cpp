#include "irodori-tts/synthesizer.h"
#include "irodori-tts/reference.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>

static std::string str(const JsonValue & j, const char * k) {
    auto p=j.member(k); return p && p->kind==JsonValue::Kind::String ? p->text : "";
}
static std::vector<float> floats(const std::string & path) {
    if(path.empty()) return {};
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("Cannot open float data "+path);
    auto n=f.tellg();if(n<0 || n%4)throw std::runtime_error("Invalid float file");
    std::vector<float> v((size_t)n/4);f.seekg(0);f.read((char*)v.data(),n);return v;
}
int main(int argc,char **argv) {
 try {
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if(!dev)throw std::runtime_error("Vulkan GPU unavailable");
    auto backend=ggml_backend_dev_init(dev,nullptr);
    irodori::Synthesizer synth(argv[1],backend);
    // The public accessor is const; encoding changes only the codec's scratch allocator.
    auto & codec=const_cast<irodori::Codec &>(synth.codec());
    irodori::ReferenceRules rules(synth.model());
    std::map<std::string,std::vector<float>> clips;
    std::string voice_key;
    irodori::Voice voice;
    std::cout<<"QUALITY {\"type\":\"ready\",\"device\":"<<json_string(ggml_backend_dev_description(dev))<<"}"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
      std::string id;
      try {
        auto j=parse_json(line);if(str(j,"type")=="stop")break;id=str(j,"id");
        auto start=std::chrono::steady_clock::now();
        const auto * refs=j.member("references");
        std::string key=refs?to_json(*refs):"[]";
        if(key!=voice_key) {
          std::vector<float> latent;
          if(refs)for(const auto & p:refs->items) {
            auto it=clips.find(p.text);
            if(it==clips.end())it=clips.emplace(p.text,irodori::encode_reference(codec,p.text,rules).latent).first;
            latent.insert(latent.end(),it->second.begin(),it->second.end());
            if(latent.size()>=3000*32)break;
          }
          if(latent.size()>3000*32)latent.resize(3000*32);
          voice=latent.empty()?irodori::Voice{}:synth.voice_from_latent(std::move(latent));voice_key=key;
        }
        irodori::Request r;r.text=str(j,"text");r.caption=str(j,"caption");
        r.seed=std::stoull(j.member("seed")->text);r.steps=std::stoi(argv[2]);
        auto seconds=j.member("seconds");r.length.seconds=seconds && seconds->kind==JsonValue::Kind::Number?std::stod(seconds->text):0;
        auto tail=synth.default_tail();tail.keep=true;r.tail=tail;
        if(r.steps==40){auto g=synth.default_guidance();g.text=3;g.caption=3;g.speaker=5;g.mode=irodori::GuidanceMode::Independent;g.min_t=.5;g.max_t=1;g.sway=0;r.guidance=g;}
        r.noise=floats(str(j,"noise"));
        std::vector<float> audio;irodori::Stats stats;
        synth.synthesize(r,voice,[&](const float*p,size_t n){audio.insert(audio.end(),p,p+n);return true;},&stats);
        std::ofstream f(str(j,"output"),std::ios::binary);f.write((char*)audio.data(),audio.size()*4);
        if(!f)throw std::runtime_error("Cannot save audio");
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"QUALITY {\"type\":\"result\",\"id\":"<<json_string(id)<<",\"samples\":"<<audio.size()<<",\"frames\":"<<stats.frames<<",\"sample_rate\":48000,\"ms\":"<<ms<<",\"raw_float\":true}"<<std::endl;
      } catch(const std::exception&e){std::cout<<"QUALITY {\"type\":\"error\",\"id\":"<<json_string(id)<<",\"error\":"<<json_string(e.what())<<"}"<<std::endl;}
    }
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
