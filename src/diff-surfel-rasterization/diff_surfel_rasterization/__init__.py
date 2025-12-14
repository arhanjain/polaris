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

from typing import NamedTuple
import torch.nn as nn
import torch

# ==============================================================================
# JIT COMPILATION SUPPORT
# ==============================================================================
# Try to import pre-compiled _C module, if not available, compile via JIT
# ==============================================================================

try:
    # Try to import pre-compiled extension (from setup.py build)
    from . import _C
except ImportError:
    # If not available, compile via JIT on first import
    import os
    import warnings
    from pathlib import Path

    def _load_extension_jit():
        """JIT compile the CUDA extension if pre-built version not available."""
        from torch.utils.cpp_extension import load

        # Get source directory (parent of this __init__.py)
        # _src_path = Path(__file__).parent.parent
        _src_path = Path(__file__).parent / "csrc"

        # Find all source files
        sources = []
        for pattern in ["ext.cpp", "cuda_rasterizer/*.cu", "*.cu"]:
            sources.extend([str(p) for p in _src_path.glob(pattern)])

        if not sources:
            raise FileNotFoundError(
                f"No source files found in {_src_path}. "
                "Make sure diff-surfel-rasterization is properly installed."
            )

        # Compilation settings
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]

        extra_cflags = ["-O3", "-std=c++17"]

        # Include directories
        include_dirs = [
            str(_src_path),
            str(_src_path / "cuda_rasterizer"),
            str(_src_path.parent.parent / "third_party" / "glm"),
        ]

        # Build directory
        cuda_ver = (
            torch.version.cuda.replace(".", "_") if torch.cuda.is_available() else "cpu"
        )
        build_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "torch_extensions",
            f"diff_surfel_rasterization_cu{cuda_ver}",
        )

        # Create build directory if it doesn't exist
        os.makedirs(build_dir, exist_ok=True)

        is_first_build = not os.path.exists(os.path.join(build_dir, "build.ninja"))
        if is_first_build:
            print("\n" + "=" * 70)
            print("Compiling diff-surfel-rasterization (first time only)...")
            print("This will take 2-5 minutes.")
            print("=" * 70 + "\n")

        try:
            extension = load(
                name="diff_surfel_rasterization_cuda",
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
            print("ERROR: Failed to compile diff-surfel-rasterization")
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
            "CUDA not available. diff-surfel-rasterization requires CUDA.\n"
            f"PyTorch version: {torch.__version__}"
        )

    _C = _load_extension_jit()

# ==============================================================================
# REST OF ORIGINAL CODE (unchanged)
# ==============================================================================


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.near_n,
            raster_settings.far_n,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    depth,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.rasterize_gaussians(*args)
            )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            raster_settings.near_n,
            raster_settings.far_n,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    near_n: float
    far_n: float


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.near_n,
                raster_settings.far_n,
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
