#pragma once
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <functional>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
namespace bench {
inline thread_local bool active=false;
inline thread_local std::vector<std::string> events;
struct Scope {
    const char * name; bool enabled; std::chrono::steady_clock::time_point start;
    int batch=0, frames=0;
    Scope(const char * n):name(n),enabled(active),start(std::chrono::steady_clock::now()){}
    ~Scope(){ finish(); }
    void finish(){ if(!enabled)return;
        enabled=false;
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::ostringstream s; s << "{\"stage\":" << std::quoted(name) << ",\"ms\":" << ms
          << ",\"batch\":" << batch << ",\"frames\":" << frames << "}";
        events.push_back(s.str());
    }
};
inline std::string take(){std::string s="[";for(auto & e:events){if(s.size()>1)s+=",";s+=e;}return s+"]";}
inline double elapsed(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
// Separate graph nodes and allocations: backend graph optimizers mutate graph metadata.
// Only leaves (parameters and prepared inputs) share read-only source buffers.
inline void encoder_probe(const char *name,ggml_tensor *source,ggml_backend_t backend){
    if(!active || !source)return;
    const size_t capacity=65536;
    auto *ctx=ggml_init({ggml_graph_overhead_custom(capacity,false)+capacity*ggml_tensor_overhead(),nullptr,true});
    std::unordered_map<ggml_tensor*,ggml_tensor*> copies;
    std::function<ggml_tensor*(ggml_tensor*)> clone=[&](ggml_tensor *t)->ggml_tensor*{
        if(!t)return nullptr;
        if(copies.count(t))return copies[t];
        auto *copy=ggml_new_tensor(ctx,t->type,4,t->ne);
        *copy=*t;copies[t]=copy;
        copy->extra=nullptr;
        for(int i=0;i<GGML_MAX_SRC;i++)copy->src[i]=clone(t->src[i]);
        copy->view_src=clone(t->view_src);
        if(t->op!=GGML_OP_NONE){copy->data=nullptr;copy->buffer=nullptr;}
        return copy;
    };
    auto *output=clone(source);auto *graph=ggml_new_graph_custom(ctx,capacity,false);
    ggml_build_forward_expand(graph,output);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(alloc,graph))throw std::runtime_error("Encoder probe allocation failed");
    {
        Scope timer(name);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Encoder probe compute failed");
        ggml_backend_synchronize(backend);
        std::vector<char> bytes(ggml_nbytes(output));ggml_backend_tensor_get(output,bytes.data(),0,bytes.size());
    }
    ggml_gallocr_free(alloc);ggml_free(ctx);
}
}
