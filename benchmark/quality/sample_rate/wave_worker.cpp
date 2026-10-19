// Exercise the pinned C++ WAV reader and resampler without loading a GPU model.
#include "wav.h"
#include "resample.h"
#include "json-reader.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <string>
static std::string get(const JsonValue &j, const char *key) {
    const auto *p = j.member(key);
    return p && p->kind == JsonValue::Kind::String ? p->text : "";
}
int main() {
    std::cout << "QUALITY {\"type\":\"ready\"}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string id;
        try {
            auto j = parse_json(line);
            if (get(j, "type") == "stop") break;
            id = get(j, "id");
            auto wav = read_wav(get(j, "wav"));
            auto audio = Resampler(wav.sample_rate, 48000)(wav.mono());
            std::ofstream out(get(j, "output"), std::ios::binary);
            out.write(reinterpret_cast<const char *>(audio.data()), audio.size() * sizeof(float));
            if (!out) throw std::runtime_error("cannot write output");
            std::cout << "QUALITY {\"type\":\"result\",\"id\":" << json_string(id)
                      << ",\"samples\":" << audio.size() << ",\"sample_rate\":48000}" << std::endl;
        } catch (const std::exception &e) {
            std::cout << "QUALITY {\"type\":\"error\",\"id\":" << json_string(id)
                      << ",\"error\":" << json_string(e.what()) << "}" << std::endl;
        }
    }
}
