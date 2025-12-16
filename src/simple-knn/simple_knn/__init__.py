#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch

# ==============================================================================
# JIT COMPILATION SUPPORT
# ==============================================================================
# Try to import pre-compiled _C module, if not available, compile via JIT
# ==============================================================================

try:
    # Try to import pre-compiled extension (from setup.py build)
    from . import _simple_knn as _C
except ImportError:
    # If not available, compile via JIT on first import
    import os
    from pathlib import Path

    def _load_extension_jit():
        """JIT compile the CUDA extension if pre-built version not available."""
        from torch.utils.cpp_extension import load

        # Get source directory
        # simple-knn typically has structure:
        # simple-knn/
        #   simple_knn/
        #     __init__.py  (this file)
        #   spatial.cu
        #   simple_knn.cu
        #   ext.cpp

        _pkg_path = Path(__file__).parent
        _src_path = _pkg_path.parent  # Go up to repo root where .cu files are

        # Find all source files
        sources = []

        # Common simple-knn source files
        potential_files = [
            _src_path / "ext.cpp",
            _src_path / "spatial.cu",
            _src_path / "simple_knn.cu",
        ]

        for f in potential_files:
            if f.exists():
                sources.append(str(f))

        # Also search recursively for any .cu/.cpp files we might have missed
        for ext in ["*.cu", "*.cpp"]:
            for p in _src_path.rglob(ext):
                p_str = str(p)
                if p_str not in sources and "test" not in p_str.lower():
                    sources.append(p_str)

        if not sources:
            raise FileNotFoundError(
                f"No source files found in {_src_path}. "
                "Make sure simple-knn is properly installed.\n"
                f"Package path: {_pkg_path}\n"
                f"Source path: {_src_path}"
            )

        # Compilation settings
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]

        extra_cflags = ["-O3", "-std=c++17"]

        # Include directories
        include_dirs = [str(_src_path)]

        # Build directory
        cuda_ver = (
            torch.version.cuda.replace(".", "_") if torch.cuda.is_available() else "cpu"
        )
        build_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "torch_extensions",
            f"simple_knn_cu{cuda_ver}",
        )

        # Create build directory if it doesn't exist
        os.makedirs(build_dir, exist_ok=True)

        is_first_build = not os.path.exists(os.path.join(build_dir, "build.ninja"))

        if is_first_build:
            print("\n" + "=" * 70)
            print("Compiling simple-knn (first time only)...")
            print("This will take 1-2 minutes.")
            print("=" * 70 + "\n")

        try:
            extension = load(
                name="simple_knn_cuda",
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=include_dirs,
                build_directory=build_dir,
                verbose=is_first_build,
                with_cuda=True,
            )

            if is_first_build:
                print("\n✓ Compilation successful! Cached for future use.\n")

            return extension

        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR: Failed to compile simple-knn")
            print("=" * 70)
            print(f"\n{e}\n")
            print("Requirements:")
            print("  - CUDA toolkit installed")
            print("  - Compatible C++ compiler (gcc 7-12)")
            print("  - PyTorch with CUDA support")
            print("=" * 70 + "\n")
            raise

    # Load via JIT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. simple-knn requires CUDA.\n"
            f"PyTorch version: {torch.__version__}"
        )

    _C = _load_extension_jit()

# ==============================================================================
# Export the distCUDA2 function (main API)
# ==============================================================================


def distCUDA2(points):
    """
    Compute KNN distances for points using CUDA.

    Args:
        points: Tensor of shape (N, 3) containing 3D points

    Returns:
        Tensor of shape (N,) containing squared distances to nearest neighbors
    """
    return _C.distCUDA2(points)
