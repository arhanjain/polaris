#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = "Differentiable rasterizer for 2D Gaussian Splatting"

setup(
    name="diff_surfel_rasterization",
    version="0.1.0",
    author="Binbin Huang",
    description="Differentiable rasterizer for 2D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tordjx/diff-surfel-rasterization",
    # Find the diff_surfel_rasterization package
    packages=find_packages(),
    # Include all source files (.cu, .cpp, .h) for JIT compilation
    package_data={
        "": [
            "*.cu",
            "*.cpp",
            "*.h",
            "*.cuh",
            "cuda_rasterizer/*.cu",
            "cuda_rasterizer/*.h",
            "cuda_rasterizer/*.cuh",
        ],
    },
    # Also include source files at repo root level
    # These will be accessible from Path(__file__).parent.parent in __init__.py
    data_files=[
        (
            ".",
            [
                "ext.cpp",
                # Add any root-level .cu files here
            ],
        ),
        (
            "cuda_rasterizer",
            [
                # List cuda_rasterizer files if they're at repo root
            ],
        ),
    ],
    include_package_data=True,
    # Dependencies
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.13.0",
        "ninja",  # For faster JIT compilation
    ],
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# ==============================================================================
# NOTES:
# ==============================================================================
#
# This setup.py does NOT use CUDAExtension to pre-compile the extension.
# Instead, it ships source files which are JIT compiled on first import.
#
# OLD APPROACH (removed):
#   from torch.utils.cpp_extension import CUDAExtension, BuildExtension
#   ext_modules=[
#       CUDAExtension(
#           name='diff_surfel_rasterization._C',
#           sources=[...],
#           extra_compile_args={...}
#       )
#   ],
#   cmdclass={'build_ext': BuildExtension}
#
# NEW APPROACH:
#   - Ship source files via package_data
#   - JIT compile in __init__.py using torch.utils.cpp_extension.load()
#   - First import takes 2-5 minutes, subsequent imports are instant
#
# BENEFITS:
#   ✓ pip install is instant (no compilation)
#   ✓ Works on any CUDA version (compiles for user's hardware)
#   ✓ No need to build wheels for every Python/CUDA combination
#   ✓ Smaller package size (source only, no binaries)
#
# ==============================================================================
