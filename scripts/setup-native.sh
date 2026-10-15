#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
engine="${1:?Specify audio.cpp, speech.cpp, sttts-gui, Irodori-TTS-burn, or irodori-onnx}"
case "$engine" in
audio.cpp|speech.cpp)
    failed=0
    for backend in cuda vulkan; do
        build="$WORKSPACE_ROOT/artifacts/build/$engine/$backend"
        args=(-S "engines/$engine" -B "$build" -DCMAKE_BUILD_TYPE=Release)
        if [[ "$engine" == audio.cpp ]]; then
            args+=(-DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=irodori_tts
                   -DENGINE_ENABLE_METAL=OFF -DENGINE_ENABLE_HIP=OFF
                   -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF)
            [[ "$backend" != cuda ]] || args+=(-DENGINE_ENABLE_CUDA=ON)
            [[ "$backend" != vulkan ]] || args+=(-DENGINE_ENABLE_VULKAN=ON)
            target=audiocpp_cli
        else
            args+=(-DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF)
            [[ "$backend" != cuda ]] || args+=(-DGGML_CUDA=ON)
            [[ "$backend" != vulkan ]] || args+=(-DGGML_VULKAN=ON)
            target=speech-cli
        fi
        if ! toolchain_options="$(python3 scripts/native_toolchain.py "$backend")"; then
            echo "Build skipped: $engine/$backend (native toolchain unavailable)" >&2
            failed=1
            continue
        fi
        if [[ -n "$toolchain_options" ]]; then
            mapfile -t toolchain_args <<< "$toolchain_options"
            args+=("${toolchain_args[@]}")
        fi
        if ! cmake "${args[@]}" || ! cmake --build "$build" --target "$target" --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"; then
            echo "Build failed: $engine/$backend (see the preceding error; system installation is skipped)" >&2
            failed=1
        fi
    done
    exit "$failed"
    ;;
sttts-gui)
    cargo build --manifest-path engines/sttts-gui/Cargo.toml --release --locked \
        --target-dir artifacts/build/sttts-gui -p sttts-say
    ;;
Irodori-TTS-burn)
    cargo build --manifest-path engines/Irodori-TTS-burn/Cargo.toml --release --locked \
        --target-dir artifacts/build/Irodori-TTS-burn --features cli,synthesis --bin irodori
    ;;
irodori-onnx)
    cargo build --manifest-path engines/irodori-onnx/Cargo.toml --release --locked \
        --target-dir artifacts/build/irodori-onnx -p irodori-tts --example synth
    ;;
*) echo "Unknown engine: $engine" >&2; exit 2 ;;
esac
