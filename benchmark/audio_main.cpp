int main(int argc,char **argv){
 try {
    engine::runtime::ModelLoadRequest load_request;load_request.model_path=argv[1];load_request.family_hint="irodori_tts";
    auto t=std::chrono::steady_clock::now();auto registry=engine::runtime::make_default_registry();auto model=registry.load(load_request);
    engine::runtime::SessionOptions options;options.backend.type=parse_backend(argv[3]);options.backend.device=0;options.backend.threads=8;
    options.options["irodori_tts.reference_cache_slots"]="1";
    auto session_base=model->create_task_session({engine::runtime::VoiceTaskKind::Tts,engine::runtime::RunMode::Offline},options);
    auto *session=dynamic_cast<engine::runtime::IOfflineVoiceTaskSession*>(session_base.get());
    auto *irodori=dynamic_cast<engine::models::irodori_tts::IrodoriTTSSession*>(session_base.get());
    if(!session || !irodori)throw std::runtime_error("Session creation failed");
    session->prepare(engine::runtime::SessionPreparationRequest{});
    std::cout<<"BENCH {\"type\":\"ready\",\"load_ms\":"<<bench::elapsed(t)<<",\"device\":"<<std::quoted(argv[3])<<"}"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)){
        const auto object=engine::io::json::parse(line);
        if(optional_string(object,"type")=="stop")break;
        const auto *p=object.find("profile");bench::active=p && p->as_bool();bench::events.clear();
        if(optional_string(object,"cache")=="reencode")irodori->bench_clear_reference_cache();
        auto t=std::chrono::steady_clock::now();auto request=make_request(object);
        request.options["text_chunk_size"]="4096";
        auto result=session->run(request);const double ms=bench::elapsed(t);
        if(!result.audio_output)throw std::runtime_error("No PCM returned");
        const auto &audio=*result.audio_output;double sq=0;bool finite=true;
        for(float v:audio.samples){sq+=v*v;finite&=std::isfinite(v);}
        auto output=optional_string(object,"output");
        if(!output.empty()){std::ofstream out(output,std::ios::binary);out.write((char*)audio.samples.data(),audio.samples.size()*sizeof(float));}
        std::cout<<"BENCH {\"type\":\"result\",\"id\":"<<std::quoted(optional_string(object,"id"))<<",\"ms\":"<<ms
          <<",\"samples\":"<<audio.samples.size()<<",\"sample_rate\":"<<audio.sample_rate
          <<",\"frames\":"<<(audio.samples.size()+1919)/1920<<",\"events\":"<<bench::take()
          <<",\"finite\":"<<(finite?"true":"false")<<",\"rms\":"<<std::sqrt(sq/audio.samples.size())<<"}"<<std::endl;
    }
 }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
