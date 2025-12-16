"""
Utilities to validate and upload a PolaRiS environment folder to Hugging Face.
Default target dataset: `PolaRiS-Evals/PolaRiS-Hub`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import re
import tyro

from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError  # type: ignore

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
            p
            for p in asset_dir.rglob("*")
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


def _validate_initial_conditions(
    ic_path: Path, asset_names: Iterable[str]
) -> Tuple[List[str], List[str]]:
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
                errors.append(
                    f"Pose {idx} for `{obj_name}` is not a 7-element numeric sequence"
                )
            elif not _objects_match_assets(obj_name, asset_names):
                warnings.append(
                    f"Pose {idx} references `{obj_name}` which does not obviously map to an asset "
                    f"({', '.join(asset_names)})"
                )
    return errors, warnings


def _validate_usd_files(
    env_dir: Path, require_pxr: bool = False
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    usd_files = list(env_dir.glob("*.usda"))
    if not usd_files:
        errors.append(f"No stage .usda file found in {env_dir}")
        return errors, warnings

    try:
        from pxr import Usd  # type: ignore
    except Exception as exc:  # noqa: BLE001 - library is optional
        if require_pxr:
            errors.append(f"pxr.Usd not available; cannot open USD files ({exc})")
        # when not required, stay quiet to keep dry-runs clean
        return errors, warnings

    for usd_file in usd_files:
        stage = Usd.Stage.Open(str(usd_file))
        if stage is None:
            errors.append(f"Failed to open USD stage: {usd_file}")
            continue
        if stage.GetDefaultPrim() is None:
            warnings.append(f"USD stage has no default prim set: {usd_file}")
    return errors, warnings


def validate_environment(
    env_dir: Path, require_pxr: bool = False
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not env_dir.exists():
        return [f"Environment path does not exist: {env_dir}"], warnings
    if not env_dir.is_dir():
        return [f"Environment path is not a directory: {env_dir}"], warnings

    assets_errors, assets_warnings, asset_names = _validate_assets(env_dir / "assets")
    errors.extend(assets_errors)
    warnings.extend(assets_warnings)

    ic_errors, ic_warnings = _validate_initial_conditions(
        env_dir / "initial_conditions.json", asset_names
    )
    errors.extend(ic_errors)
    warnings.extend(ic_warnings)

    usd_errors, usd_warnings = _validate_usd_files(env_dir, require_pxr=require_pxr)
    errors.extend(usd_errors)
    warnings.extend(usd_warnings)

    return errors, warnings


def upload_environment(
    env_dir: Path,
    repo_id: str,
    token: str | None,
    branch: str,
    pr_branch: str | None,
    commit_message: str | None,
    pr_title: str | None,
    pr_description: str | None,
) -> None:
    env_name = env_dir.name
    api = HfApi(token=token)
    commit_message = commit_message or f"Add environment `{env_name}`"
    if pr_title:
        commit_message = pr_title
    if pr_description:
        commit_message = f"{commit_message}\n\n{pr_description}"

    operations = []
    for file in env_dir.rglob("*"):
        if not file.is_file():
            continue
        rel_path = file.relative_to(env_dir).as_posix()
        path_in_repo = f"{env_name}/{rel_path}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(file),
            )
        )
    pr_title = pr_title or f"Add environment `{env_name}`"
    pr_description = pr_description or ""
    revision = pr_branch or branch
    try:
        commit_info = api.create_commit(  # type: ignore[arg-type]
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            revision=revision,
            commit_message=commit_message,
            create_pr=True,
        )
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            f"Repository `{repo_id}` not found or unauthorized. "
            "Ensure the dataset exists and your HF token has write access."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Hugging Face API error while creating PR: {exc}") from exc
    pr_url = getattr(commit_info, "pr_url", None)
    pr_num = getattr(commit_info, "pr_num", None)

    # Try to extract PR number from URL if not directly available
    if pr_url and not pr_num:
        match = re.search(r"/(?:pull|pulls|discussions)/(\d+)", pr_url)
        if match:
            pr_num = match.group(1)

    if pr_url:
        print(f"Pull request opened: {pr_url}")
    elif pr_num:
        pr_url = f"https://huggingface.co/datasets/{repo_id}/discussions/{pr_num}"
        print(f"Pull request opened: {pr_url}")
    else:
        discussions_page = f"https://huggingface.co/datasets/{repo_id}/discussions"
        print(
            f"Pull request created (URL not returned by API). Check: {discussions_page}"
        )

    if pr_num:
        repo_name = repo_id.split("/")[-1]
        print("\nTo check out and update this PR locally:")
        print(f"  git clone https://huggingface.co/datasets/{repo_id}")
        print(f"  cd {repo_name} && git fetch origin refs/pr/{pr_num}:pr/{pr_num}")
        print(f"  git checkout pr/{pr_num}")
        print("  # make edits, then:")
        print(f"  git push origin pr/{pr_num}:refs/pr/{pr_num}")
    print(f"PR source revision: {revision} -> target: {branch}")


@dataclass
class Args:
    """Validate and upload a PolaRiS environment to Hugging Face."""

    env_dir: Path
    """Path to the environment folder (e.g., /home/mingtong/polaris/PolaRiS-environments/food_bussing)"""

    repo_id: str = "owhan/PolaRiS-environments"
    """Target Hugging Face dataset repository"""

    branch: str = "main"
    """Target branch on the dataset repository"""

    pr_branch: str | None = None
    """Optional source branch/ref for the PR (e.g., refs/pr/104); defaults to --branch"""

    token: str | None = None
    """Hugging Face token (defaults to HF_TOKEN env var if omitted)"""

    skip_validation: bool = False
    """Upload without running local validation (not recommended)"""

    strict: bool = False
    """Treat validation warnings as errors"""

    require_pxr: bool = False
    """Fail validation if pxr (USD) is unavailable for stage open checks"""

    dry_run: bool = False
    """Only run validation; do not upload"""

    commit_message: str | None = None
    """Optional commit message for the upload"""

    pr_title: str | None = None
    """Pull request title"""

    pr_description: str | None = None
    """Pull request description/body"""


def main(args: Args | None = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    env_dir: Path = args.env_dir.resolve()

    if not args.skip_validation:
        errors, warnings = validate_environment(env_dir, require_pxr=args.require_pxr)
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
        pr_branch=args.pr_branch,
        commit_message=args.commit_message,
        pr_title=args.pr_title,
        pr_description=args.pr_description,
    )
    print(
        f"Prepared PR for `{env_dir.name}` to {args.repo_id} (target branch: {args.branch})."
    )


if __name__ == "__main__":
    main()
