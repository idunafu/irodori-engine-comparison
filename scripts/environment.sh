#!/usr/bin/env bash
# Source this file from a workspace script; no shell profile is modified.
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKSPACE_ROOT
export UV_CACHE_DIR="$WORKSPACE_ROOT/.cache/uv"
export HF_HOME="$WORKSPACE_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export CARGO_HOME="$WORKSPACE_ROOT/.cache/cargo"
export XDG_CACHE_HOME="$WORKSPACE_ROOT/.cache"
export XDG_DATA_HOME="$WORKSPACE_ROOT/.cache/data"
export STTTS_ROOT="$WORKSPACE_ROOT/artifacts/sttts"
export IRODORI_TTS_BURN_CACHE_DIR="$WORKSPACE_ROOT/.cache/burn"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
mkdir -p "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$CARGO_HOME" "$XDG_DATA_HOME" "$WORKSPACE_ROOT/logs"
if [[ -z "${VULKAN_SDK:-}" && -f "$WORKSPACE_ROOT/toolchains/1.4.357.0/setup-env.sh" ]]; then
    VULKAN_SDK="$WORKSPACE_ROOT/toolchains/1.4.357.0/x86_64"
fi
if [[ -n "${VULKAN_SDK:-}" ]]; then
    export VULKAN_SDK
    export PATH="$VULKAN_SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib/VulkanLoader/lib:$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PKG_CONFIG_PATH="$VULKAN_SDK/share/pkgconfig:$VULKAN_SDK/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export CMAKE_PREFIX_PATH="$VULKAN_SDK:$VULKAN_SDK/lib/VulkanLoader${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export VK_ADD_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}"
fi
