"""Resolve native build tools without assuming their installation directories."""
import argparse
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess


def cuda_architectures():
    # Preserve the published RTX 5090 configuration unless explicitly overridden.
    return os.environ.get('CMAKE_CUDA_ARCHITECTURES') or os.environ.get('CUDAARCHS') or '120'


def cuda_compiler():
    configured = os.environ.get('CMAKE_CUDA_COMPILER') or os.environ.get('CUDACXX')
    if configured:
        return shutil.which(configured) or configured
    root = os.environ.get('CUDAToolkit_ROOT') or os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if root:
        return str(Path(root).expanduser() / 'bin/nvcc')
    return shutil.which('nvcc')


def vulkan_library():
    configured = os.environ.get('Vulkan_LIBRARY') or os.environ.get('VULKAN_LIBRARY')
    if configured:
        return configured
    # Find the installed loader even when only its versioned runtime .so exists.
    soname = ctypes.util.find_library('vulkan')
    if soname and Path(soname).is_absolute() and Path(soname).is_file():
        return soname
    ldconfig = shutil.which('ldconfig')
    if soname and ldconfig:
        result = subprocess.run([ldconfig, '-p'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            entry, separator, location = line.strip().partition(' => ')
            if separator and entry.split()[0] == soname and Path(location).is_file():
                return location
    sdk = os.environ.get('VULKAN_SDK')
    if sdk:
        for relative in ('lib/libvulkan.so', 'lib/VulkanLoader/lib/libvulkan.so'):
            library = Path(sdk) / relative
            if library.is_file():
                return str(library)
    return None  # Let CMake's FindVulkan use its normal search paths.


def cmake_arguments(backend):
    if backend == 'cuda':
        compiler = cuda_compiler()
        if not compiler:
            raise RuntimeError('nvcc not found: add it to PATH or set CUDACXX / CUDAToolkit_ROOT')
        args = [f'-DCMAKE_CUDA_COMPILER={compiler}',
                f'-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures()}']
        root = os.environ.get('CUDAToolkit_ROOT') or os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if not root:
            candidate = Path(compiler).resolve().parent.parent
            if (candidate / 'include/cuda.h').is_file():
                root = str(candidate)
        if root:
            args.append(f'-DCUDAToolkit_ROOT={root}')
        return args
    if backend != 'vulkan':
        raise ValueError(f'Unknown backend: {backend}')
    args = []
    library = vulkan_library()
    if library:
        args.append(f'-DVulkan_LIBRARY={library}')
    sdk = os.environ.get('VULKAN_SDK')
    include = os.environ.get('Vulkan_INCLUDE_DIR') or (str(Path(sdk) / 'include') if sdk else None)
    glslc = os.environ.get('Vulkan_GLSLC_EXECUTABLE') or (str(Path(sdk) / 'bin/glslc') if sdk else None)
    if include:
        args.append(f'-DVulkan_INCLUDE_DIR={include}')
    if glslc:
        args.append(f'-DVulkan_GLSLC_EXECUTABLE={glslc}')
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('cuda', 'vulkan'))
    arguments = parser.parse_args()
    for option in cmake_arguments(arguments.backend):
        print(option)
