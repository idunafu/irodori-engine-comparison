#include "irodori-tts/codec.h"
#include "irodori-tts/layout.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <string>
static std::string str(const JsonValue&j,const char*k){auto*p=j.member(k);return p&&p->kind==JsonValue::Kind::String?p->text:"";}
static std::vector<float> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("read "+p);auto n=f.tellg();std::vector<float>x(n/4);f.seekg(0);f.read((char*)x.data(),n);return x;}
static void save(const std::string&p,const std::vector<float>&x){std::ofstream f(p,std::ios::binary);f.write((char*)x.data(),x.size()*4);if(!f)throw std::runtime_error("write "+p);}
int main(int argc,char**argv){try{
 ggml_backend_load_all();auto dev=ggml_backend_dev_by_type(std::string(argv[2])=="cpu"?GGML_BACKEND_DEVICE_TYPE_CPU:GGML_BACKEND_DEVICE_TYPE_GPU);
 if(!dev)throw std::runtime_error("backend unavailable");auto backend=ggml_backend_dev_init(dev,nullptr);
 if(std::string(argv[2])=="cpu")ggml_backend_cpu_set_n_threads(backend,8);
 const ModelFile model(argv[1],backend,irodori::model_layout);irodori::Codec codec(model,backend);
 std::cout<<"QUALITY {\"type\":\"ready\"}"<<std::endl;std::string line;
 while(std::getline(std::cin,line)){std::string id;try{auto j=parse_json(line);if(str(j,"type")=="stop")break;id=str(j,"id");auto audio=read(str(j,"pcm"));
  auto a=codec.encode(audio,100);save(str(j,"windowed"),a);auto b=codec.encode(audio,(int)(audio.size()/codec.hop()+1));save(str(j,"whole"),b);
  std::cout<<"QUALITY {\"type\":\"result\",\"id\":"<<json_string(id)<<",\"codec_input_dtype\":"<<(int)model.tensor("codec.enc.conv_in.weight")->type<<"}"<<std::endl;
 }catch(const std::exception&e){std::cout<<"QUALITY {\"type\":\"error\",\"id\":"<<json_string(id)<<",\"error\":"<<json_string(e.what())<<"}"<<std::endl;}}
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
