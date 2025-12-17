# PolaRiS

PolaRiS is a evaluation framework for generalist policies. It provides tooling for reconstructing environments, evaluating models, and running experiments with minimal setup.

## Installation

### Clone the repository (recursively)

```bash
git clone --recursive git@github.com:arhanjain/polaris.git
cd PolaRiS
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Setup environment with uv
If you don't have UV installed, see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)


By default we support CUDA 13. If you have an older version of CUDA installed, please downgrade the torch and torchvision version and index to be compatible in the [pyproject.toml](pyproject.toml).
```bash
uv sync 
```

<!-- ### HuggingFace Datasets
For using our evaluation DROID environments or simulation cotraining data, clone the datasets below.
```bash
# Environments
uvx hf download owhan/PolaRiS-environments --repo-type=dataset --local-dir ./PolaRiS-environments
# Cotrain Datasets
uvx hf download owhan/PolaRis-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets
``` -->

## Getting Started
First, download the PolaRiS environments (<2GB)
```bash
uvx hf download owhan/PolaRiS-environments --repo-type=dataset --local-dir ./PolaRiS-environments
```

### Minimal Code Example
Next let's test a simple random action policy in a PolaRiS environment.
```python
import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
# This must be done before importing anything with dependency on Isaaclab
# >>>> Isaac Sim App Launcher <<<<
parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# >>>> Isaac Sim App Launcher <<<<

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import MangerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions

env_cfg = parse_env_cfg(
    "DROID-FoodBussing",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)
env: MangerBasedRLSplatEnv = gym.make("DROID-FoodBussing", cfg=env_cfg)   # type: ignore
language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions = initial_conditions[0])

while True:
    action = torch.tensor(env.action_space.sample())
    obs, rew, term, trunc, info = env.step(action, expensive=True)

    if term[0] or trunc[0]:
        break

print(f"Episode Finished. Success: {info['rubric']['success']}, Progress: {info['rubric']['progress']}")
```

### Run a π0.5 Policy in PolaRiS
*Note: First run may take longer due to JIT compilation of the splat rasterization kernels. Ensure you have NVIDIA Drivers and CUDA Toolkit (nvcc) properly configured.*

Both the policy server and evaluation process should fit onto a single GPU (tested on RTX 3090, 24 GB). 
```bash
# Starting from the root of this repo. This will setup openpi and host a pi05 policy.
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_droid_jointpos_fullfinetune --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris

# In a separate process, start evaluation process
sudo apt install ffmpeg # for saving videos
uv run scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 --run-folder runs/test
```
Results include rollout videos, and a CSV summarizing success and normalized progress of each episode.

For the full list of all checkpoints and environments we provide for evaluation, see [checkpoints_and_envs.md](docs/checkpoints_and_envs.md)

## Evaluating Your Policies In PolaRiS
See [docs/custom_policies.md](docs/custom_policies.md)

## Creating Custom Evaluation Environments 
Time Estimate: 20 Minutes Human Time + 40 Minutes Offline Training

Coming soon - 12/22
<!-- See [docs/custom_environments.md](docs/custom_environments.md) -->

## Issues
This codebase has been tested on CUDA 13 and CUDA 12 with NVIDIA 5090 and 3090 GPUs. Please raise an issue if you run into any issues.
