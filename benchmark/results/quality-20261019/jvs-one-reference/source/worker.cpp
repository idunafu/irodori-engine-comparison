#include "irodori-tts/synthesizer.h"
#include "irodori-tts/reference.h"
#include "irodori-tts/loudness.h"
#include "resample.h"
#include "wav.h"
#include "ggml-backend.h"
#include "json-reader.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
static std::string str(const JsonValue&j,const char*k){auto*p=j.member(k);return p&&p->kind==JsonValue::Kind::String?p->text:"";}
static std::vector<float> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("read "+p);auto n=f.tellg();if(n<0||n%4)throw std::runtime_error("bad float file");std::vector<float>x(n/4);f.seekg(0);f.read((char*)x.data(),n);return x;}
static void save(const std::string&p,const std::vector<float>&x){std::ofstream f(p,std::ios::binary);f.write((char*)x.data(),x.size()*4);if(!f)throw std::runtime_error("write "+p);}
int main(int argc,char**argv){try{
 ggml_backend_load_all();auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);if(!dev)throw std::runtime_error("no Vulkan GPU");
 auto backend=ggml_backend_dev_init(dev,nullptr);irodori::Synthesizer synth(argv[1],backend);
 auto&codec=const_cast<irodori::Codec&>(synth.codec());irodori::ReferenceRules rules(synth.model());std::map<std::string,irodori::Voice>voices;
 std::cout<<"QUALITY {\"type\":\"ready\"}"<<std::endl;std::string line;
 while(std::getline(std::cin,line)){std::string id;try{auto j=parse_json(line);if(str(j,"type")=="stop")break;id=str(j,"id");
  if(str(j,"op")=="dump"){
   std::vector<float>latent;
   if(!str(j,"latent_in").empty())latent=read(str(j,"latent_in"));
   else{std::vector<float>pcm;if(!str(j,"pcm_in").empty())pcm=read(str(j,"pcm_in"));else{
     auto wav=read_wav(str(j,"reference"));auto raw=Resampler(wav.sample_rate,codec.sample_rate())(wav.mono());
     if(!str(j,"resampled_out").empty())save(str(j,"resampled_out"),raw);
     pcm=rules.lufs?irodori::normalize_loudness(raw,codec.sample_rate(),*rules.lufs):irodori::bound_peak(std::move(raw));}
    if(!str(j,"pcm_out").empty())save(str(j,"pcm_out"),pcm);latent=codec.encode(pcm);}
   auto voice=synth.voice_from_latent(latent);save(str(j,"latent_out"),latent);save(str(j,"state_out"),voice.speaker);
   std::cout<<"QUALITY {\"type\":\"result\",\"id\":"<<json_string(id)<<",\"frames\":"<<voice.frames<<",\"speaker_tokens\":"<<voice.speaker_tokens<<",\"speaker_dim\":"<<synth.model().u32("irodori-tts.speaker.dim")<<",\"lufs\":"<<(rules.lufs?*rules.lufs:0)<<"}"<<std::endl;
  }else{
   std::string key=str(j,"state_in")+"|"+str(j,"latent_in");auto it=voices.find(key);
   if(it==voices.end()){irodori::Voice voice;if(!str(j,"state_in").empty()){voice.speaker=read(str(j,"state_in"));voice.speaker_tokens=voice.speaker.size()/synth.model().u32("irodori-tts.speaker.dim");}else voice=synth.voice_from_latent(read(str(j,"latent_in")));it=voices.emplace(key,std::move(voice)).first;}
   irodori::Request r;r.text=str(j,"text");r.steps=std::stoi(argv[2]);r.seed=std::stoull(j.member("seed")->text);r.length.seconds=std::stod(j.member("seconds")->text);r.noise=read(str(j,"noise"));auto tail=synth.default_tail();tail.keep=true;r.tail=tail;
   if(r.steps==40){auto g=synth.default_guidance();g.text=3;g.caption=0;g.speaker=5;g.mode=irodori::GuidanceMode::Independent;g.min_t=.5;g.max_t=1;g.sway=0;r.guidance=g;}
   std::vector<float>audio;irodori::Stats stats;synth.synthesize(r,it->second,[&](const float*p,size_t n){audio.insert(audio.end(),p,p+n);return true;},&stats);save(str(j,"output"),audio);
   std::cout<<"QUALITY {\"type\":\"result\",\"id\":"<<json_string(id)<<",\"samples\":"<<audio.size()<<",\"frames\":"<<stats.frames<<",\"raw_float\":true}"<<std::endl;
  }
 }catch(const std::exception&e){std::cout<<"QUALITY {\"type\":\"error\",\"id\":"<<json_string(id)<<",\"error\":"<<json_string(e.what())<<"}"<<std::endl;}}
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
