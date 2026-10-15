#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
mkdir -p toolchains
archive=toolchains/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
if [[ ! -f toolchains/1.4.357.0/setup-env.sh ]]; then
    curl --fail --location --retry 3 -o "$archive" \
        https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
    echo '0f09bf6a0625e346bf004be70b92907e934a4c76606b323441b2baf3a5a0e66d  toolchains/vulkansdk-linux-x86_64-1.4.357.0.tar.xz' | sha256sum --check
    tar -xJf "$archive" -C toolchains
fi
source "$WORKSPACE_ROOT/scripts/environment.sh"
vulkaninfo --summary
