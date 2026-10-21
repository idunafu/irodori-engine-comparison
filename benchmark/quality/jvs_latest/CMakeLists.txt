cmake_minimum_required(VERSION 3.20)
project(irodori-latest-quality LANGUAGES C CXX)

if(NOT SPEECH_SOURCE)
    message(FATAL_ERROR "Set SPEECH_SOURCE to the pinned evaluation checkout")
endif()
add_subdirectory("${SPEECH_SOURCE}" speech)
add_executable(speech-latest-quality
    "${CMAKE_CURRENT_SOURCE_DIR}/../speech_worker.cpp"
    "${SPEECH_SOURCE}/tools/common/json-reader.cpp")
target_compile_features(speech-latest-quality PRIVATE cxx_std_17)
target_include_directories(speech-latest-quality PRIVATE
    "${SPEECH_SOURCE}/tools/common"
    "${SPEECH_SOURCE}/ggml/include")
target_link_libraries(speech-latest-quality PRIVATE irodoritts)
