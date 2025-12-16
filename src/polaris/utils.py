import os
import torch
import json
from datetime import datetime
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry

DATA_PATH = (
    Path("./PolaRiS-environments").resolve()
    if "POLARIS_DATA_PATH" not in os.environ
    else Path(os.environ["POLARIS_DATA_PATH"]).resolve()
)


def load_eval_initial_conditions(
    usd: str, initial_conditions_file: str | None = None, rollouts: int | None = None
) -> tuple[str, dict]:
    """
    If initial_conditions_file is provided, load the initial conditions from the file.
    Otherwise, load the initial conditions from the USD file. If neither exist, raise an error.
    """
    if initial_conditions_file is None:
        initial_conditions_file_path = Path(usd).parent / "initial_conditions.json"
    else:
        initial_conditions_file_path = Path(initial_conditions_file)

    if not initial_conditions_file_path.exists():
        raise FileNotFoundError(
            "Either USD directory must have an initial_conditions.json file, or a custom initial_conditions_file must be provided."
        )
    with open(initial_conditions_file_path, "r") as f:
        initial_conditions = json.load(f)

    # will have initial conditions and language instruction
    if "instruction" not in initial_conditions or "poses" not in initial_conditions:
        raise ValueError(
            "Initial conditions ill formated. Must contain 'instruction' and 'poses' keys."
        )
    instruction = initial_conditions["instruction"]
    initial_conditions = (
        initial_conditions["poses"]
        if rollouts is None
        else initial_conditions["poses"][:rollouts]
    )
    return instruction, initial_conditions


def run_folder_path(run_folder: str | None, usd: str, policy: str) -> Path:
    """
    If run_folder is not provided, create a new run folder in the runs directory with the current date and time.
    Otherwise, use the provided run folder.
    """
    if not run_folder:
        run_folder_path = f"runs/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%I:%M:%S %p')}"
    else:
        run_folder_path = run_folder

    run_folder_path = Path(run_folder_path) / Path(usd).stem / policy
    print(f" >>> Saving to {run_folder_path} <<< ")
    run_folder_path.mkdir(parents=True, exist_ok=True)
    return run_folder_path


def parse_env_cfg(
    task_name: str,
    usd_file: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
) -> ManagerBasedRLEnvCfg:
    """
    Parse configuration for an environment and override based on inputs.
    Adapted from isaaclab_tasks.utils.parse_env_cfg.

    New Parameters
    --------------
    usd_file: str
        Path to USD file we want to use
    """
    # load the default configuration
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")

    # check that it is not a dict
    if isinstance(cfg, dict):
        raise RuntimeError(
            f"Configuration for the task: '{task_name}' is not a class. Please provide a class."
        )

    cfg.dynamic_setup(usd_file)

    # simulation device
    cfg.sim.device = device
    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return cfg


def rotate_vector_by_quaternion(q, v):
    """Rotate vectors by quaternions using the fast Hamilton product.

    Args:
        q: (4) tensor of quaternions in [w,x,y,z] format
        v: (..., 3) tensor of vectors to rotate
    Returns:
        (..., 3) tensor of rotated vectors
    """
    q = q.repeat(v.shape[:-1] + (1,))
    # Extract quaternion components
    qw = q[..., 0]
    qv = q[..., 1:]

    # uv = 2 * cross(qv, v)
    uv = 2 * torch.cross(qv, v, dim=-1)

    # return v + qw * uv + cross(qv, uv)
    return v + qw[..., None] * uv + torch.cross(qv, uv, dim=-1)


def multiply_quaternions(q1, q2):
    """Fast quaternion multiplication using PyTorch.
    Assumes quaternions are in [w,x,y,z] format.

    Args:
        q1: (N, 4) tensor of quaternions
        q2: (N, 4) tensor of quaternions
    Returns:
        (N, 4) tensor of resulting quaternions
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    # Compute components directly
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack((w, x, y, z), dim=-1)
