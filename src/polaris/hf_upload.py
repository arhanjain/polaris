"""
Utilities to validate and upload a PolaRiS environment folder to Hugging Face.

Default target dataset: `PolaRiS-Evals/PolaRiS-Hub`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from huggingface_hub import HfApi  # type: ignore

ALLOWED_MESH_SUFFIXES = {".usdz", ".usd", ".glb", ".ply"}


def _is_numeric_sequence(value: Iterable[object], expected_len: int = 7) -> bool:
    try:
        items = list(value)
    except TypeError:
        return False
    if len(items) != expected_len:
        return False
    return all(isinstance(v, (int, float)) for v in items)


def _validate_assets(assets_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    asset_names: List[str] = []

    if not assets_dir.exists():
        errors.append(f"Missing assets directory: {assets_dir}")
        return errors, warnings, asset_names
    if not assets_dir.is_dir():
        errors.append(f"`assets` is not a directory: {assets_dir}")
        return errors, warnings, asset_names

    for asset_dir in sorted(p for p in assets_dir.iterdir() if p.is_dir()):
        asset_names.append(asset_dir.name)
        mesh_candidates = [
            p for p in asset_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_MESH_SUFFIXES
        ]
        if not mesh_candidates:
            errors.append(
                f"Asset `{asset_dir.name}` is missing any mesh file "
                f"(expected one of {sorted(ALLOWED_MESH_SUFFIXES)})"
            )
    if not asset_names:
        errors.append(f"No asset subfolders found in {assets_dir}")
    return errors, warnings, asset_names


def _objects_match_assets(obj_name: str, asset_names: Iterable[str]) -> bool:
    normalized = obj_name.lower().rstrip("0123456789_")
    for asset in asset_names:
        asset_norm = asset.lower().rstrip("0123456789_")
        if (
            normalized.startswith(asset_norm)
            or asset_norm.startswith(normalized)
            or normalized in asset_norm
            or asset_norm in normalized
        ):
            return True
    return False


def _validate_initial_conditions(ic_path: Path, asset_names: Iterable[str]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not ic_path.exists():
        errors.append(f"Missing initial_conditions.json at {ic_path}")
        return errors, warnings
    try:
        with ic_path.open("r") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - surfaced to user
        errors.append(f"Failed to parse {ic_path}: {exc}")
        return errors, warnings

    if not isinstance(data, dict):
        errors.append("initial_conditions.json must be a JSON object")
        return errors, warnings

    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("`instruction` must be a non-empty string")

    poses = data.get("poses")
    if not isinstance(poses, list) or not poses:
        errors.append("`poses` must be a non-empty list")
        return errors, warnings

    for idx, pose in enumerate(poses):
        if not isinstance(pose, dict):
            errors.append(f"Pose {idx} is not an object")
            continue
        for obj_name, obj_pose in pose.items():
            if not _is_numeric_sequence(obj_pose, expected_len=7):
                errors.append(f"Pose {idx} for `{obj_name}` is not a 7-element numeric sequence")
            elif not _objects_match_assets(obj_name, asset_names):
                warnings.append(
                    f"Pose {idx} references `{obj_name}` which does not obviously map to an asset "
                    f"({', '.join(asset_names)})"
                )
    return errors, warnings


def _validate_usd_files(env_dir: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    usd_files = list(env_dir.glob("*.usd"))
    if not usd_files:
        errors.append(f"No stage .usd file found in {env_dir}")
        return errors, warnings

    try:
        from pxr import Usd  # type: ignore
    except Exception:  # noqa: BLE001 - library is optional
        # pxr is not installed; skip the deeper USD check but do not block validation.
        return errors, warnings

    for usd_file in usd_files:
        stage = Usd.Stage.Open(str(usd_file))
        if stage is None:
            errors.append(f"Failed to open USD stage: {usd_file}")
            continue
        if stage.GetDefaultPrim() is None:
            warnings.append(f"USD stage has no default prim set: {usd_file}")
    return errors, warnings


def validate_environment(env_dir: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not env_dir.exists():
        return [f"Environment path does not exist: {env_dir}"], warnings
    if not env_dir.is_dir():
        return [f"Environment path is not a directory: {env_dir}"], warnings

    assets_errors, assets_warnings, asset_names = _validate_assets(env_dir / "assets")
    errors.extend(assets_errors)
    warnings.extend(assets_warnings)

    ic_errors, ic_warnings = _validate_initial_conditions(env_dir / "initial_conditions.json", asset_names)
    errors.extend(ic_errors)
    warnings.extend(ic_warnings)

    usd_errors, usd_warnings = _validate_usd_files(env_dir)
    errors.extend(usd_errors)
    warnings.extend(usd_warnings)

    return errors, warnings


def upload_environment(
    env_dir: Path,
    repo_id: str,
    token: str | None,
    branch: str,
    commit_message: str | None,
) -> None:
    env_name = env_dir.name
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(env_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=env_name,
        revision=branch,
        commit_message=commit_message or f"Add environment `{env_name}`",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and upload a PolaRiS environment to Hugging Face.")
    parser.add_argument(
        "env_dir",
        type=Path,
        help="Path to the environment folder (e.g., /home/mingtong/polaris/PolaRiS-environments/food_bussing)",
    )
    parser.add_argument(
        "--repo-id",
        default="PolaRiS-Evals/PolaRiS-Hub",
        help="Target Hugging Face dataset repository",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch on the dataset repository",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN env var if omitted)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Upload without running local validation (not recommended)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as errors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run validation; do not upload",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for the upload",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env_dir: Path = args.env_dir.resolve()

    if not args.skip_validation:
        errors, warnings = validate_environment(env_dir)
        for warn in warnings:
            print(f"[WARN] {warn}")
        if errors:
            for err in errors:
                print(f"[ERROR] {err}")
            sys.exit(1)
        if args.strict and warnings:
            print("[ERROR] Warnings treated as errors because --strict is set")
            sys.exit(1)
    else:
        print("Skipping validation as requested.")

    if args.dry_run:
        print("Dry run complete; nothing uploaded.")
        return

    upload_environment(
        env_dir=env_dir,
        repo_id=args.repo_id,
        token=args.token,
        branch=args.branch,
        commit_message=args.commit_message,
    )
    print(f"Uploaded `{env_dir.name}` to {args.repo_id} (branch: {args.branch}).")


if __name__ == "__main__":
    main()

