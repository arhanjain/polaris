#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = "Simple KNN for 3D Gaussian Splatting"

setup(
    name="simple_knn",
    version="0.1.0",
    author="Bernhard Kerbl",
    description="Simple KNN CUDA implementation for 3D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/bkerbl/simple-knn",
    # Find the simple_knn package
    packages=find_packages(),
    # Include all source files (.cu, .cpp, .h) for JIT compilation
    package_data={
        "": ["*.cu", "*.cpp", "*.h", "*.cuh"],
    },
    # Also include source files at repo root
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
# BENEFITS:
#   ✓ pip install is instant (no compilation)
#   ✓ Works on any CUDA version (compiles for user's hardware)
#   ✓ No need to build wheels for every Python/CUDA combination
#   ✓ Smaller package size (source only, no binaries)
#
# ==============================================================================
