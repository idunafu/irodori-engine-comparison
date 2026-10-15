"""Check toolchain selection independently of the machine's installation paths."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from native_toolchain import cmake_arguments, cuda_compiler, vulkan_library


class NativeToolchainTest(unittest.TestCase):
    def test_relocated_cuda_toolkit_and_architecture(self):
        with tempfile.TemporaryDirectory(prefix='cuda toolkit ') as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            compiler = root / 'bin/nvcc'
            compiler.touch()
            with patch.dict(os.environ, {'CUDA_HOME': str(root), 'CUDAARCHS': '90;120'}, clear=True):
                args = cmake_arguments('cuda')
            self.assertIn(f'-DCMAKE_CUDA_COMPILER={compiler}', args)
            self.assertIn(f'-DCUDAToolkit_ROOT={root}', args)
            self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=90;120', args)

    def test_explicit_compiler_overrides_toolkit_and_path(self):
        with patch.dict(os.environ, {'CUDACXX': 'chosen-nvcc', 'CUDA_HOME': 'other-toolkit'}, clear=True), \
                patch('native_toolchain.shutil.which', return_value='chosen-nvcc'):
            self.assertEqual(cuda_compiler(), 'chosen-nvcc')

    def test_path_discovery_keeps_published_default_architecture(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('native_toolchain.shutil.which', return_value='nvcc'):
            args = cmake_arguments('cuda')
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=120', args)

    def test_missing_cuda_does_not_select_a_cpu_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('native_toolchain.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'nvcc not found'):
                cmake_arguments('cuda')

    def test_external_vulkan_sdk_and_loader_override(self):
        with tempfile.TemporaryDirectory(prefix='vulkan sdk ') as sdk, \
                patch.dict(os.environ, {'VULKAN_SDK': sdk, 'VULKAN_LIBRARY': 'chosen-loader'}, clear=True):
            args = cmake_arguments('vulkan')
        self.assertIn('-DVulkan_LIBRARY=chosen-loader', args)
        self.assertIn(f'-DVulkan_INCLUDE_DIR={sdk}/include', args)
        self.assertIn(f'-DVulkan_GLSLC_EXECUTABLE={sdk}/bin/glslc', args)

    def test_sdk_loader_is_used_without_system_search_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            library = sdk / 'lib/VulkanLoader/lib/libvulkan.so'
            library.parent.mkdir(parents=True)
            library.touch()
            with patch.dict(os.environ, {'VULKAN_SDK': str(sdk)}, clear=True), \
                    patch('native_toolchain.ctypes.util.find_library', return_value=None):
                self.assertEqual(vulkan_library(), str(library))


if __name__ == '__main__':
    unittest.main()
